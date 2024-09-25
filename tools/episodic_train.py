import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
#from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from nmc.models import *
from nmc.datasets import * 
from nmc.augmentations import get_train_augmentation, get_val_augmentation
from nmc.losses import get_loss
from nmc.schedulers import get_scheduler
from nmc.optimizers import get_optimizer
from nmc.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate_epi
from nmc.utils.episodic_utils import * 

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mf1 = 0.0
    device = torch.device(cfg['DEVICE'])
    train_cfg, sched_cfg = cfg['TRAIN'], cfg['SCHEDULER']
    num_episodes = train_cfg['NUM_EPISODES']
    dataset_cfg =  cfg['DATASET']
    
    image_dir = Path(dataset_cfg['ROOT']) / 'train_images'
    transformations = get_train_augmentation(train_cfg['IMAGE_SIZE'])
    
    episodic_dataset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], dataset_cfg['N_WAY'], dataset_cfg['K_SHOT'], dataset_cfg['Q_QUERY'], train_cfg['NUM_EPISODES'], dataset_cfg['SPLIT_RATIO'], dataset_cfg['MINOR_CLS'], transformations)
    # print("Episodic dataset is generated")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], episodic_dataset.n_classes)
    model.init_pretrained(cfg['MODEL']['PRETRAINED'])
    model.unfreezing_layer(cfg['MODEL']['UNFREEZE']) 
    model = model.to(device)
    
    print("Model is initialized")
    
    if train_cfg['DDP']:
        model = DDP(model, device_ids=[gpu])
        sampler = DistributedSampler(episodic_dataset_train, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        sampler = None
    optimizer = get_optimizer(model, cfg['OPTIMIZER']['NAME'], cfg['OPTIMIZER']['LR'], cfg['OPTIMIZER']['WEIGHT_DECAY'])
    criterion_cls = get_loss(cfg['LOSS_CLS']['NAME'])
    criterion_dist_loss = get_loss('DistContrastive')
    
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, num_episodes, sched_cfg['POWER'], num_episodes * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])

   
    pbar = tqdm(total=num_episodes, desc=f"Episode: [{0}/{num_episodes}] Loss: {0:.8f}")
    
    #eval roc curve
    adaptive_threshold = AdaptiveROCThreshold(episodic_dataset.n_classes, momentum=0.9)
    
    print("Start Training ...")
    epoch = 3
    for _ in range(epoch):
        for episode_idx in range(num_episodes):
            model.train()
            # print(f"Episode index: {episode_idx}")
            if train_cfg['DDP']:
                sampler.set_epoch(episode_idx)

            # support_x, support_y, query_x, query_y = episodic_dataset_train.create_episode()
            support_x, support_y, query_x, query_y = episodic_dataset.create_episode(is_train=True)
            support_x, support_y = support_x.to(device), support_y.to(device)
            # support_y = [batch,n_class]
            query_x, query_y = query_x.to(device), query_y.to(device)
            
            #loss 추적
            class_losses = {f"class_{i}": 0 for i in range(support_y.size(1))}
            query_losses = {f"query_{i}": 0 for i in range(query_y.size(1))}
            
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=train_cfg['AMP']):
                support_pred = model(support_x)
                #support_pred = [batch,n_class,embedding]
                    
                similarity_matrix = dot_similarity(support_pred)
                #dot_similarity = [n_class,batch,batch]
                support_y_t = support_y.t()
                
                #각 클래스별 loss 생성이 끝난 후 negative prototype을 만들어 각 임베딩별 loss 생성
                #negative_prototype = calculate_negative_prototypes(support_pred,support_y).detach()

                class_exists = (support_y.sum(dim=0) > 0)
                for c in range(similarity_matrix.size(0)):
                    total_loss =0
                    class_loss=0
                    class_similarities = similarity_matrix[c]
                    class_labels = support_y_t[c]
                    
                    if class_labels.sum() >= 2:  # skip if less than 2 samples for this class
                        class_loss = criterion_cls(class_similarities, class_labels) #contrastive loss
                        total_loss += class_loss
                        class_losses[f"class_{c}"] = class_loss.item()
                    
                    # 역전파
                    # 계산에 참여한 head만 자동으로 계산됨(디버깅함)
                    # retain_traph를 통해 batch단위 loss 역전파동안 계산그래프 유지
                    if class_loss is not 0:
                        scaler.scale(total_loss).backward(retain_graph=True)
                    
            # opt step은 한번만
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()
            
            optimizer.zero_grad(set_to_none=True)      
            
            with autocast(enabled=train_cfg['AMP']):
                with torch.no_grad():
                    support_pred = model(support_x)
                    prototypes, pos_dist, neg_dist = compute_prototypes_dist(support_pred,support_y)
                query_pred = model(query_x)
                for c in range(query_pred.size(1)):  # 클래스 수만큼 반복
                    total_loss =0
                    query_class_loss =0
                    if ~torch.isnan(pos_dist[c]):
                        query_class_loss = criterion_dist_loss(query_pred[:,c,:],prototypes[c], query_y[:,c])
                        total_loss += query_class_loss
                        query_losses[f"query_{c}"] = query_class_loss.item()
                    # 현재 클래스에 대한 loss 계산
                    if query_class_loss != 0:
                        # 개별 클래스에 대한 backward 수행
                        scaler.scale(total_loss).backward(retain_graph=True)
                    
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()
            
            # Create a formatted string for the losses
            loss_str = " ".join([f"{k}: {v:.4f}" for k, v in {**class_losses, **query_losses}.items()])

            pbar.set_description(f"Episode: [{episode_idx+1}/{num_episodes}]")
            pbar.update(1)
            # 세로로 loss 출력
            print("\nLosses:")
            num_classes = len(class_losses)
            max_class_key_length = max(len(key) for key in class_losses.keys())
            max_query_key_length = max(len(key) for key in query_losses.keys())
            max_value_length = 8  # ".4f" 형식으로 출력할 때의 길이

            print(f"{'Class'.ljust(max_class_key_length)} {'Value'.ljust(max_value_length)} {'Query'.ljust(max_query_key_length)} {'Value'}")
            print("-" * (max_class_key_length + max_query_key_length + max_value_length * 2 + 3))

            for i in range(num_classes):
                class_key = f"class_{i}"
                query_key = f"query_{i}"
                class_value = class_losses.get(class_key, 0)
                query_value = query_losses.get(query_key, 0)
                
                print(f"{class_key.ljust(max_class_key_length)} {f'{class_value:.4f}'.ljust(max_value_length)} {query_key.ljust(max_query_key_length)} {query_value:.4f}")

            print()  # 빈 줄 추가
            
            if (episode_idx + 1) % train_cfg['EVAL_INTERVAL'] == 0 or (episode_idx + 1) == num_episodes:
                results, active_classes = evaluate_epi(model, episodic_dataset, device, adaptive_threshold, num_episodes=10)
                mf1 = results['avg_f1']
                
                print(f"Accuracy: {results['accuracy']:.2f}%")
                print(f"Average Precision: {results['avg_precision']:.2f}%")
                print(f"Average Recall: {results['avg_recall']:.2f}%")
                print(f"Average F1: {results['avg_f1']:.2f}%")

                print("\nPer-class metrics:")
                for class_idx, metrics in results['class_metrics']['precision'].items():
                    if class_idx in active_classes:
                        print(f"Class {class_idx}:")
                        print(f"  Precision: {results['class_metrics']['precision'][class_idx]:.2f}%")
                        print(f"  Recall: {results['class_metrics']['recall'][class_idx]:.2f}%")
                        print(f"  F1: {results['class_metrics']['f1'][class_idx]:.2f}%")

                if mf1 > best_mf1:
                    best_mf1 = mf1
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{dataset_cfg['NAME']}.pth")
                print(f"Current mf1: {mf1} Best mf1: {best_mf1}")

    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mf1', f"{best_mf1:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    print(cfg)
    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()