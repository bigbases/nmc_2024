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
    testloaders = []
    for i in range(len(dataset_cfg)):
        # trainset = eval(dataset_cfg[i]['NAME'])(dataset_cfg[i]['ROOT']+'/train_images', traintransform)
        # valset = eval(dataset_cfg[i]['NAME'])(dataset_cfg[i]['ROOT']+'/val_images', valtransform)
        testset = eval(dataset_cfg[i]['NAME'])(dataset_cfg[i]['ROOT']+'/test_images', valtransform)
        # multi_task_dataset_train.append(trainset)
        # multi_task_dataset_vali.append(valset)
        test_loader = DataLoader(testset, batch_size=8, num_workers=num_workers, pin_memory=True)
        testloaders.append(test_loader)
        

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], [8,5])
    model_path = '/workspace/proj03_code/wrjeong/nmc_2024/output/FGMaxxVit_Multi_FGMaxxVit_Multi_task_100_epoch.pth'
    model.load_state_dict(torch.load(model_path))
    # model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)
    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    else:
        # sampler = RandomSampler(trainset)
        #writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()
        # val_cycled_loaders = [cycle(loader) for loader in valloaders  ]
    
        # val_cycled_loaders = [cycle(loader) for loader in valloaders  ]
        results_0 = evaluate_multi(model, testloaders[0],0, device)
        results_1 = evaluate_multi(model, testloaders[1],1, device)
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

        print(f"Current mf1: {mf1} Best mf1: {best_mf1}")
        
    pbar.close()
    
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