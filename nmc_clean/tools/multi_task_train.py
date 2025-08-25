import torch 
import argparse
import yaml
import time
import multiprocessing as mp
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
from val import evaluate_multi
from itertools import cycle



def main(cfg, gpu, save_dir):
    start = time.time()
    best_mf1 = 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg_1 , dataset_cfg_2 , model_cfg = cfg['DATASET1'],cfg['DATASET2'], cfg['MODEL']
    dataset_cfg = [dataset_cfg_1, dataset_cfg_2]
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    # multi_task = ['ODIRDataset','APTOSDataset']
    multi_task_dataset_train = []
    multi_task_dataset_vali = []
    for i in range(len(dataset_cfg)):
        trainset = eval(dataset_cfg[i]['NAME'])(dataset_cfg[i]['ROOT']+'/train_images', traintransform)
        valset = eval(dataset_cfg[i]['NAME'])(dataset_cfg[i]['ROOT']+'/val_images', valtransform)
        multi_task_dataset_train.append(trainset)
        multi_task_dataset_vali.append(valset)
        

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], [8,5])
    model.init_pretrained(model_cfg['PRETRAINED'])
    model.unfreezing_layer(model_cfg['UNFREEZE']) 
    model = model.to(device)
    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    else:
        sampler = RandomSampler(trainset)
    
    trainloaders = []
    valloaders = []
    
    for trainset, valset in zip(multi_task_dataset_train,multi_task_dataset_vali):
        trainloaders.append(DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler))
        valloaders.append(DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True))
    
    iters_per_epoch = max(len(loader) for loader in trainloaders)
    # class_weights = trainset.class_weights.to(device)
    loss_fn = get_loss(loss_cfg['NAME'])
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    #writer = SummaryWriter(str(save_dir / 'logs'))

    # for name, param in model.named_parameters():
    #     if str(param.device) == 'cuda:0':
    #         continue
    #     print(f"Parameter {name} is on device: {param.device}")
    # exit()
    for epoch in range(epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0
        
        
        pbar = tqdm(range(iters_per_epoch), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        # for iter, (img, lbl) in pbar:
        
        # pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        # pbar = tqdm(total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}]")
        cycled_loaders = [cycle(loader) for loader in trainloaders]
        for batch_idx in pbar:
            
            # for batch_idx in range(iters_per_epoch):
            # print(batch_idx)
            optimizer.zero_grad()
            total_loss = 0
            multi_loss = 0 
            # 각 작업에 대해 forward pass 및 손실 계산
            for task_idx, loader in enumerate(cycled_loaders):
                inputs, targets = next(loader)
                inputs, targets = inputs.to(device), targets.to(device)
                # print(inputs[0].device)
                # print(f"Model is on device: {next(model.parameters()).device}")
                # exit()
                with autocast(enabled=train_cfg['AMP']):
                    outputs = model(inputs,task_idx)
                    # print(task_idx)
                    # print(targets)
                    # print(outputs)
                    loss = loss_fn(outputs, targets)
                    multi_loss += loss
            train_loss += multi_loss.item()
            scaler.scale(multi_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()
            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{batch_idx+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (batch_idx+1):.8f}")
            
        train_loss /= batch_idx+1
        #writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()
        # val_cycled_loaders = [cycle(loader) for loader in valloaders  ]
        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            # val_cycled_loaders = [cycle(loader) for loader in valloaders  ]
            results_0 = evaluate_multi(model, valloaders[0],0, device)
            results_1 = evaluate_multi(model, valloaders[1],1, device)
            results = {
            'accuracy': (results_0['accuracy'] + results_1['accuracy']) / 2,
            'avg_precision': (results_0['avg_precision'] + results_1['avg_precision']) / 2,
            'avg_recall': (results_0['avg_recall'] + results_1['avg_recall']) / 2,
            'avg_f1': (results_0['avg_f1'] + results_1['avg_f1']) / 2,
            'class_metrics': {
            'precision': {**results_0['class_metrics']['precision'], **results_1['class_metrics']['precision']},
            'recall': {**results_0['class_metrics']['recall'], **results_1['class_metrics']['recall']},
            'f1': {**results_0['class_metrics']['f1'], **results_1['class_metrics']['f1']}
            }
            }
            
            
            mf1 = results['avg_f1']
            #writer.add_scalar('val/mf1', mf1, epoch)
            
            print(f"Accuracy: {results['accuracy']:.2f}%")
            print(f"Average Precision: {results['avg_precision']:.2f}%")
            print(f"Average Recall: {results['avg_recall']:.2f}%")
            print(f"Average F1: {results['avg_f1']:.2f}%")

            print("\nPer-class metrics:")
            print('Task 1 : ')
            for class_idx, metrics in results_0['class_metrics']['precision'].items():
                print(f"Class {class_idx}:")
                print(f"  Precision: {results['class_metrics']['precision'][class_idx]:.2f}%")
                print(f"  Recall: {results['class_metrics']['recall'][class_idx]:.2f}%")
                print(f"  F1: {results['class_metrics']['f1'][class_idx]:.2f}%")
                
            
            print('Task 2 : ')
            for class_idx, metrics in results_0['class_metrics']['precision'].items():
                print(f"Class {class_idx}:")
                print(f"  Precision: {results['class_metrics']['precision'][class_idx]:.2f}%")
                print(f"  Recall: {results['class_metrics']['recall'][class_idx]:.2f}%")
                print(f"  F1: {results['class_metrics']['f1'][class_idx]:.2f}%")

            if mf1 > best_mf1:
                best_mf1 = mf1
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), f"{save_dir}/{model_cfg['NAME']}_{model_cfg['BACKBONE']}_Multi_task.pth")
            print(f"Current mf1: {mf1} Best mf1: {best_mf1}")
        print(save_dir)
        torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), f"{save_dir}/{model_cfg['NAME']}_{model_cfg['BACKBONE']}_Multi_task.pth")
    #writer.close()
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