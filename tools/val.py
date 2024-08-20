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
from nmc.utils.episodic_utils import * 

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
    metrics = MultiLabelMetrics(dataset.n_classes, device=device)

    for _ in tqdm(range(num_episodes)):
        support_x, support_y, query_x, query_y = dataset.create_episode()
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        support_pred = model(support_x)
        query_pred = model(query_x)
        num_classes = support_pred.size(1)  # 클래스의 수 (라벨의 차원)
        prototypes = compute_prototypes_multi_label(support_pred, support_y)
        # prototypes shape : n_class , embedding_dim 
        prototypes = prototypes.unsqueeze(0)  # (1, num_classes, embedding_dim)
        similarities = dot_product_similarity(query_pred, prototypes)  # (batch_size, num_classes)
        # thresholded_similarities = torch.where(similarities >= 0.5, torch.tensor(1.0), torch.tensor(0.0)) # << 혹시 라벨화가 필요할까봐 남겨놓음
        
        metrics.update(similarities, query_y)

    results = metrics.compute_metrics()
    
    return results


@torch.no_grad()
def evaluate_multi(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,task_id : int , device: str) -> Dict[str, float]:
    print('Evaluating...')
    model.eval()
    metrics = Metrics(num_classes=dataloader.dataset.n_classes, device=device)
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images,task_id)
        # print(labels)
        # print(outputs)
        # print(labels.shape , outputs.shape)
        # print(outputs[task_id])
        # print(labels)
        # task_outputs = outputs[task_id]
        metrics.update(outputs, labels)
        
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