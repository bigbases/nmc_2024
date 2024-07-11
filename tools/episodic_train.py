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
from val import evaluate_epi

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mf1 = 0.0
    device = torch.device(cfg['DEVICE'])
    train_cfg, sched_cfg = cfg['TRAIN'], cfg['SCHEDULER']
    num_episodes = train_cfg['NUM_EPISODES']
    dataset_cfg =  cfg['DATASET']
    
    image_dir = Path(cfg['DATASET']['ROOT']) / 'train_images'
    transformations = get_train_augmentation(train_cfg['IMAGE_SIZE'])

    episodic_dataset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT']+'/train_images', dataset_cfg['N_WAY'], dataset_cfg['K_SHOT'], dataset_cfg['Q_QUERY'], transformations)
    
    print("Episodic dataset is generated")
    
    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], episodic_dataset.n_classes)
    model.init_pretrained(cfg['MODEL']['PRETRAINED'])
    model = model.to(device)
    
    print("Model is initialized")
    
    if train_cfg['DDP']:
        model = DDP(model, device_ids=[gpu])
        sampler = DistributedSampler(episodic_dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        sampler = None

    optimizer = get_optimizer(model, cfg['OPTIMIZER']['NAME'], cfg['OPTIMIZER']['LR'], cfg['OPTIMIZER']['WEIGHT_DECAY'])
    criterion = get_loss(cfg['LOSS']['NAME'], None, None)
    
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, num_episodes, sched_cfg['POWER'], num_episodes * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    
    model.train()
    pbar = tqdm(total=num_episodes, desc=f"Episode: [{0}/{num_episodes}] Loss: {0:.8f}")
    
    print("Start Training ...")

    for episode_idx in range(num_episodes):
        print(f"Episode index: {episode_idx}")
        if train_cfg['DDP']:
            sampler.set_epoch(episode_idx)

        support_x, support_y, query_x, query_y = episodic_dataset.create_episode()

        optimizer.zero_grad(set_to_none=True)
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)
        
        with autocast(enabled=train_cfg['AMP']):
            support_pred = model(support_x)
            support_loss = criterion(support_pred, support_y)

        scaler.scale(support_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.cuda.synchronize()

        # Evaluate on the query set within the same episode
        with torch.no_grad():
            query_pred = model(query_x).softmax(dim=1)
            query_loss = criterion(query_pred, query_y)

        train_loss = support_loss.item()
        query_loss_item = query_loss.item()
        pbar.set_description(f"Episode: [{episode_idx+1}/{num_episodes}] Support Loss: {train_loss:.8f} Query Loss: {query_loss_item:.8f}")
        pbar.update(1)

        if (episode_idx + 1) % train_cfg['EVAL_INTERVAL'] == 0 or (episode_idx + 1) == num_episodes:
            acc, macc, f1, mf1 = evaluate_epi(model, episodic_dataset, device, num_episodes=10)

            if mf1 > best_mf1:
                best_mf1 = mf1
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth")
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