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

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mf1 = 0.0
    device = torch.device(cfg['DEVICE'])
    train_cfg, sched_cfg = cfg['TRAIN'], cfg['SCHEDULER']
    num_episodes = train_cfg['NUM_EPISODES']
    dataset_cfg =  cfg['DATASET']
    
    image_dir = Path(dataset_cfg['ROOT']) / 'train_images'
    transformations = get_train_augmentation(train_cfg['IMAGE_SIZE'])

    episodic_dataset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT']+'/train_images', dataset_cfg['N_WAY'], dataset_cfg['K_SHOT'], dataset_cfg['Q_QUERY'], transformations)
    
    print("Episodic dataset is generated")
    
    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], episodic_dataset.n_classes)
    model.init_pretrained(cfg['MODEL']['PRETRAINED'])
    model.unfreezing_layer(cfg['MODEL']['UNFREEZE']) 
    model = model.to(device)
    
    print("Model is initialized")
    
    if train_cfg['DDP']:
        model = DDP(model, device_ids=[gpu])
        sampler = DistributedSampler(episodic_dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        sampler = None

    optimizer = get_optimizer(model, cfg['OPTIMIZER']['NAME'], cfg['OPTIMIZER']['LR'], cfg['OPTIMIZER']['WEIGHT_DECAY'])
    criterion_cls = get_loss(cfg['LOSS_CLS']['NAME'])
    criterion = get_loss(cfg['LOSS']['NAME'])
    
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, num_episodes, sched_cfg['POWER'], num_episodes * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    
    def dot_similarity(embeddings):
        #embeddings = [batch,n_class,embedding]
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        batch_size, n_class, embedding_dim = embeddings.shape
        similarity = torch.zeros(n_class, batch_size, batch_size, device=embeddings.device)
        
        for c in range(n_class):
            class_embeddings = embeddings[:, c, :]  # [batch, embedding_dim]
            similarity[c] = torch.mm(class_embeddings, class_embeddings.t())
        
        return similarity
        
    model.train()
    pbar = tqdm(total=num_episodes, desc=f"Episode: [{0}/{num_episodes}] Loss: {0:.8f}")
    
    print("Start Training ...")
    
    for episode_idx in range(num_episodes):
        # print(f"Episode index: {episode_idx}")
        if train_cfg['DDP']:
            sampler.set_epoch(episode_idx)

        support_x, support_y, query_x, query_y = episodic_dataset.create_episode()

        optimizer.zero_grad(set_to_none=True)
        support_x, support_y = support_x.to(device), support_y.to(device)
        # support_y = [batch,n_class]
        query_x, query_y = query_x.to(device), query_y.to(device)
        
        with autocast(enabled=train_cfg['AMP']):
            support_pred = model(support_x)
            #support_pred = [batch,n_class,embedding]
                
            similarity_matrix = dot_similarity(support_pred)
            #dot_similarity = [n_class,batch,batch]
            support_y_t = support_y.t()
            
            for c in range(similarity_matrix.size(0)):
                class_similarities = similarity_matrix[c]
                class_labels = support_y_t[c]
                
                if class_labels.sum() < 2:  # skip if less than 2 samples for this class
                    continue
                class_loss = criterion_cls(class_similarities, class_labels) #contrastive loss

                # 역전파
                # 계산에 참여한 head만 자동으로 계산됨(디버깅함)
                # retain_traph를 통해 batch단위 loss 역전파동안 계산그래프 유지
                scaler.scale(class_loss).backward(retain_graph=True)
                
                
        # opt step은 한번만
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.cuda.synchronize()
        
        # START ---------------Query 유사도 계산을 위한 함수 ---------------------------
        def compute_prototypes_multi_label(embeddings, labels, num_classes):
            batch_size, n_classes, embedding_dim = embeddings.shape
            prototypes = []
            for c in range(num_classes):
                class_mask = labels[:, c] > 0  # 특정 클래스 c에 속하는 샘플을 선택
                if class_mask.sum() == 0:
                    prototypes.append(torch.zeros(embedding_dim, device=embeddings.device)) # 없으면 0 vector
                else:
                    class_embeddings = embeddings[class_mask]
                    prototype = class_embeddings[:,c,:].mean(dim=0) # class 별 Embedding의 평균 추출 
                    prototypes.append(prototype)
            return torch.stack(prototypes) # ( n_class , embedding )

        def dot_product_similarity(query_embeddings, prototypes):
            """
            query_embeddings: (batch_size, num_classes, embedding_dim)
            prototypes: (1, num_classes, embedding_dim)
            """
            similarities = torch.matmul(query_embeddings, prototypes.transpose(1, 2))  # (15, 11, 11)
            similarities = similarities.diagonal(dim1=-2, dim2=-1)  # (15, 11) 대각 행렬만 추출하여 각 class 별 embedding의 유사도를 추출 
            return similarities # ( batch size , n_class )
        # END ---------------Query 유사도 계산을 위한 함수 ---------------------------
            
        optimizer.zero_grad(set_to_none=True)         
        with autocast(enabled=train_cfg['AMP']):
            support_pred = model(support_x)
            query_pred = model(query_x)
            num_classes = support_pred.size(1)  # 클래스의 수 (라벨의 차원)
            prototypes = compute_prototypes_multi_label(support_pred, support_y, num_classes)
            # prototypes shape : n_class , embedding_dim 
            prototypes = prototypes.unsqueeze(0)  # (1, num_classes, embedding_dim)
            similarities = dot_product_similarity(query_pred, prototypes)  # (batch_size, num_classes)
            thresholded_similarities = torch.where(similarities >= 0.5, torch.tensor(1.0), torch.tensor(0.0)) # << 혹시 라벨화가 필요할까봐 남겨놓음
            query_loss = criterion(similarities, query_y) # BCE loss 계산 
        scaler.scale(query_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.cuda.synchronize()

        train_loss = class_loss.item()
        query_loss_item = query_loss.item()
        pbar.set_description(f"Episode: [{episode_idx+1}/{num_episodes}] Support Loss: {train_loss:.8f} Query Loss: {query_loss_item:.8f}")
        pbar.update(1)

        if (episode_idx + 1) % train_cfg['EVAL_INTERVAL'] == 0 or (episode_idx + 1) == num_episodes:
            results = evaluate_epi(model, episodic_dataset, device, num_episodes=10)
            mf1 = results['avg_f1']
            
            print(f"Accuracy: {results['accuracy']:.2f}%")
            print(f"Average Precision: {results['avg_precision']:.2f}%")
            print(f"Average Recall: {results['avg_recall']:.2f}%")
            print(f"Average F1: {results['avg_f1']:.2f}%")

            print("\nPer-class metrics:")
            for class_idx, metrics in results['class_metrics']['precision'].items():
                print(f"Class {class_idx}:")
                print(f"  Precision: {results['class_metrics']['precision'][class_idx]:.2f}%")
                print(f"  Recall: {results['class_metrics']['recall'][class_idx]:.2f}%")
                print(f"  F1: {results['class_metrics']['f1'][class_idx]:.2f}%")

            if mf1 > best_mf1:
                best_mf1 = mf1
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
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