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
from nmc.losses import get_loss
from nmc.utils.episodic_utils import * 
import copy

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
        
    results = metrics.compute_metrics_prev()
    return results

def test_support_train(model, support_x, support_y, device):
    temp_model = copy.deepcopy(model).to(device)
    temp_model.train()
    
    optimizer = torch.optim.AdamW(temp_model.parameters(), lr=0.001, weight_decay=0.01)
    criterion_cls = get_loss('Contrastive')
    criterion_bce_cls = get_loss('BCEWithLogitsLoss')
    
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.cuda.amp.autocast():
        support_pred = temp_model(support_x)
        similarity_matrix = dot_similarity(support_pred)
        support_y_t = support_y.t()
        
        optimizer.zero_grad()
        for c in range(similarity_matrix.size(0)):
            total_loss = 0
            class_loss=0
            class_similarities = similarity_matrix[c]
            class_labels = support_y_t[c]
            
            if class_labels.sum() >= 2:
                class_loss = criterion_cls(class_similarities, class_labels)
                total_loss += class_loss
            
            if class_loss is not 0:
                scaler.scale(total_loss).backward(retain_graph=True)
    
    scaler.step(optimizer)
    scaler.update()
    
    ###################
    optimizer.zero_grad(set_to_none=True)
    support_pred = temp_model(support_x,True)
    
    head_loss = criterion_bce_cls(support_pred.squeeze(),support_y)
    #scaler.scale(head_loss).backward()
    for i in range(head_loss.size(1)):
        if support_y[:,i].sum() !=0:
            hl = head_loss[:, i].mean()  # i번째 헤드의 평균 손실
            scaler.scale(hl).backward(retain_graph=True)
    
    scaler.step(optimizer)
    scaler.update()
    
    return temp_model
@torch.no_grad()
def evaluate_epi(model, dataset, device, num_episodes=10):
    print('Evaluating...')
    #global_prototypes : n_class, pn, embeddings
    model.eval()
    metrics = MultiLabelMetrics(dataset.n_classes, device=device)

    support_x, support_y, query_x, query_y = dataset.create_episode(is_train=False)
    support_x, support_y = support_x.to(device), support_y.to(device)
    with torch.enable_grad():
        temp_model = test_support_train(model,support_x, support_y, device)
    temp_model.eval()
    query_x = query_x.to(device)
    query_y = query_y.to(device)
    
    query_pred = temp_model(query_x,True)

    # Update metrics
    metrics.update(query_pred, query_y)
    active_support = torch.where(support_y.sum(dim=0) > 0)[0]
    active_classes = torch.where(query_y.sum(dim=0) > 0)[0]
    print('support:',active_support)
    print('query:',active_classes)
    results = metrics.compute_metrics(active_classes)
    return results, active_classes


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