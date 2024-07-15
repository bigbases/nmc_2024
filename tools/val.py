import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from nmc.models import *
from nmc.datasets import *
from nmc.augmentations import get_val_augmentation
from nmc.metrics import Metrics, MultiLabelMetrics
from nmc.utils.utils import setup_cudnn
from typing import Tuple, Dict


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> Dict[str, float]:
    print('Evaluating...')
    model.eval()
    metrics = Metrics(num_classes=dataloader.dataset.n_classes, device=device)

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        metrics.update(outputs, labels)
    
    results = metrics.compute_metrics()
    
    return results

@torch.no_grad()
def evaluate_multilabel(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> Dict[str, float]:
    print('Evaluating...')
    model.eval()
    metrics = MultiLabelMetrics(num_classes=dataloader.dataset.n_classes, device=device)
    
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        metrics.update(outputs, labels)
        
    results = metrics.compute_metrics()
    
    return results

@torch.no_grad()
def evaluate_epi(model, dataset, device, num_episodes=10):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(dataset.n_classes, device=device)

    for _ in tqdm(range(num_episodes)):
        support_x, support_y, query_x, query_y = dataset.create_episode()

        query_x = query_x.to(device)
        query_y = query_y.to(device).argmax(dim=1)

        preds = model(query_x).softmax(dim=1).argmax(dim=1).to(torch.int64).flatten()
        metrics.update_epi(preds, query_y)

    results = metrics.compute_metrics()
    
    return results

def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, 1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)

    acc, macc, f1, mf1 = evaluate(model, dataloader, device)

    table = {
        'Class': list(dataset.CLASSES) + ['Mean'],
        #'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Acc': acc + [macc]
    }

    print(tabulate(table, headers='keys'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)