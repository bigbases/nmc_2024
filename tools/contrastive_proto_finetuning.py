import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from nmc.models.heads import MultiHeadEmbedding, MultiHeadEmbeddingWithClassifier
from nmc.models import *
from nmc.datasets import * 
from nmc.augmentations import get_train_augmentation, get_val_augmentation
from nmc.losses import get_loss
from nmc.schedulers import get_scheduler
from nmc.optimizers import get_optimizer
from nmc.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
import os
from collections import defaultdict
import numpy as np
from torch import optim, nn
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def load_pretrained_model_and_add_classifier(pretrained_path, device):
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 모델 구조 생성
    efficientnet = models.efficientnet_v2_m(pretrained=False)
    num_ftrs = efficientnet.classifier[1].in_features
    num_classes = 7
    embedding_dim = 256
    
    # MultiHeadEmbeddingWithClassifier를 classifier로 사용
    efficientnet.classifier = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        MultiHeadEmbeddingWithClassifier(num_ftrs, num_classes, embedding_dim)
    )
    
    # 가중치 로드
    efficientnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
    efficientnet = efficientnet.to(device)
    
    return efficientnet

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels
            if isinstance(self.labels, np.ndarray):
                self.labels = torch.from_numpy(self.labels)
        elif hasattr(dataset, 'targets'):
            self.labels = dataset.targets
            if isinstance(self.labels, np.ndarray):
                self.labels = torch.from_numpy(self.labels)
        else:
            try:
                self.labels = [sample[1] for sample in dataset]
                if isinstance(self.labels[0], np.ndarray):
                    self.labels = torch.from_numpy(np.array(self.labels))
                else:
                    self.labels = torch.tensor(self.labels)
            except:
                raise ValueError("Cannot access labels from dataset")
        
        self.n_classes = self.labels.shape[1] if len(self.labels.shape) > 1 else len(torch.unique(self.labels))
        self.samples_per_class = batch_size // self.n_classes
        
        self.class_indices = []
        for i in range(self.n_classes):
            if len(self.labels.shape) > 1:
                idx = torch.where(self.labels[:, i] == 1)[0]
            else:
                idx = torch.where(self.labels == i)[0]
            self.class_indices.append(idx)
        
        self.n_batches = len(self.dataset) // batch_size
        if len(self.dataset) % batch_size != 0:
            self.n_batches += 1
    
    def __iter__(self):
        for _ in range(self.n_batches):
            batch_indices = []
            for class_idx in range(self.n_classes):
                class_samples = self.class_indices[class_idx]
                if len(class_samples) == 0:
                    continue
                
                selected = class_samples[torch.randint(len(class_samples), 
                                                     (self.samples_per_class,))]
                batch_indices.extend(selected.tolist())
            
            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[:self.batch_size]
            
            yield batch_indices
    
    def __len__(self):
        return self.n_batches

class GradientManager:
    def __init__(self, strategy='pcgrad', alpha=0.5):
        self.strategy = strategy
        self.alpha = alpha
        self.task_weights = None
        self.initial_losses = None
        
    def compute_weighted_gradients(self, backbone_grads, losses):
        if self.initial_losses is None:
            self.initial_losses = [loss.item() for loss in losses]
            
        current_losses = [loss.item() for loss in losses]
        relative_losses = [curr/init for curr, init in zip(current_losses, self.initial_losses)]
        weights = F.softmax(torch.tensor(relative_losses), dim=0)
        
        weighted_grads = {}
        for param_name in backbone_grads:
            param_grads = torch.stack(backbone_grads[param_name])
            # weights의 shape을 param_grads의 shape에 맞게 조정
            weight_view = weights.view(-1, *([1] * (len(param_grads.shape) - 1))).to(param_grads.device)
            weighted_grad = torch.sum(param_grads * weight_view, dim=0)
            weighted_grads[param_name] = weighted_grad
            
        return weighted_grads
    
    def normalize_gradients(self, backbone_grads):
        normalized_grads = {}
        for param_name in backbone_grads:
            param_grads = torch.stack(backbone_grads[param_name])
            grad_norms = torch.norm(param_grads.view(param_grads.size(0), -1), p=2, dim=1)
            normalized_grads[param_name] = (param_grads / (grad_norms.view(-1, 1, 1, 1) + 1e-8)).mean(0)
        return normalized_grads
    
    def pcgrad(self, backbone_grads):
        processed_grads = {}
        for param_name in backbone_grads:
            param_grads = torch.stack(backbone_grads[param_name])
            n_tasks = param_grads.size(0)
            processed_grad = torch.zeros_like(param_grads[0])
            
            task_indices = np.random.permutation(n_tasks)
            
            for i in task_indices:
                task_grad = param_grads[i]
                other_grads = param_grads[task_indices != i]
                
                for other_grad in other_grads:
                    conflict = torch.sum(task_grad * other_grad)
                    if conflict < 0:
                        task_grad = task_grad - (conflict / (torch.sum(other_grad * other_grad) + 1e-8)) * other_grad
                        
                processed_grad += task_grad
                
            processed_grads[param_name] = processed_grad / n_tasks
        return processed_grads
    
    def combine_gradients(self, backbone_grads, losses):
        if self.strategy == 'average':
            return {name: torch.stack(grads).mean(0) for name, grads in backbone_grads.items()}
        elif self.strategy == 'weighted':
            return self.compute_weighted_gradients(backbone_grads, losses)
        elif self.strategy == 'normalized':
            return self.normalize_gradients(backbone_grads)
        elif self.strategy == 'pcgrad':
            return self.pcgrad(backbone_grads)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

def get_train_augmentation(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Lambda(lambda x: x.float() if x.dtype == torch.uint8 else x),
        transforms.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.Lambda(lambda x: x.float() if x.dtype == torch.uint8 else x),
        transforms.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def calculate_class_contra_loss(pos_embeddings, neg_embeddings, temperature=0.1):
    pos_embeddings = F.normalize(pos_embeddings, p=2, dim=1)
    neg_embeddings = F.normalize(neg_embeddings, p=2, dim=1)
    
    pos_similarities = torch.mm(pos_embeddings, pos_embeddings.t()) / temperature
    neg_similarities = torch.mm(pos_embeddings, neg_embeddings.t()) / temperature
    
    labels = torch.arange(pos_embeddings.size(0)).to(pos_embeddings.device)
    logits = torch.cat([pos_similarities, neg_similarities], dim=1)
    
    return F.cross_entropy(logits, labels)

def train_epoch(model, dataloader, optimizer, scaler, criterion, device, gradient_strategy='pcgrad'):
    model.train()
    total_loss = 0
    
    grad_manager = GradientManager(strategy=gradient_strategy)
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        num_classes = labels.shape[1]
        
        head_grads = defaultdict(list)
        backbone_grads = defaultdict(list)
        task_losses = []
        
        optimizer.zero_grad()
        
        # Forward pass for all classes
        with autocast(enabled=scaler is not None):
            logits = model(images)
        
        # 각 클래스별로 독립적으로 학습
        for class_idx in range(num_classes):
            with autocast(enabled=scaler is not None):
                # 현재 클래스의 logits과 labels
                class_logits = logits[:, class_idx:class_idx+1]
                class_labels = labels[:, class_idx:class_idx+1]
                
                # Loss 계산
                class_loss = criterion(class_logits, class_labels)
                task_losses.append(class_loss)
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(class_loss).backward(retain_graph=True)
                else:
                    class_loss.backward(retain_graph=True)
                
                # Gradient 저장
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if 'classifier_heads' in name:  # classifier heads의 gradient
                            if f'{class_idx}' in name:  # 현재 클래스의 classifier
                                head_grads[name] = param.grad.clone()
                        else:  # backbone의 gradient
                            backbone_grads[name].append(param.grad.clone())
                
                optimizer.zero_grad()
        
        # PCGrad를 통한 gradient 결합
        with torch.no_grad():
            combined_backbone_grads = grad_manager.combine_gradients(backbone_grads, task_losses)
            
            # 최종 gradient 적용
            for name, param in model.named_parameters():
                if name in combined_backbone_grads:  # backbone
                    param.grad = combined_backbone_grads[name]
                elif name in head_grads:  # classifier heads
                    param.grad = head_grads[name]
        
        # Optimization step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        total_loss += sum(loss.item() for loss in task_losses)
    
    return {
        'total_loss': total_loss / (len(dataloader) * num_classes)
    }


def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 수정: logits만 받도록 변경
            logits = model(images)
            predictions = torch.sigmoid(logits) > 0.5
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate metrics
    results = {
        'macro': {
            'f1': f1_score(all_labels, all_predictions, average='macro'),
            'precision': precision_score(all_labels, all_predictions, average='macro'),
            'recall': recall_score(all_labels, all_predictions, average='macro')
        },
        'per_class': {
            'f1': f1_score(all_labels, all_predictions, average=None),
            'precision': precision_score(all_labels, all_predictions, average=None),
            'recall': recall_score(all_labels, all_predictions, average=None)
        }
    }
    
    # Print detailed results
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))
    
    print("\nPer-class F1 Scores:")
    for i, f1 in enumerate(results['per_class']['f1']):
        print(f"Class {i}: {f1:.4f}")
    
    print(f"\nMacro F1: {results['macro']['f1']:.4f}")
    
    return results


# train_and_evaluate 함수에서 호출 시
def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, scaler, 
                      criterion, device, epochs, save_dir):
    print("Start Training...")
    best_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            gradient_strategy='weighted'  # PCGrad 전략 사용
        )
        
        print(f"Training Loss: {train_metrics['total_loss']:.4f}")
        
        # Validation
        model.eval()
        val_metrics = evaluate(model, val_loader, device)
        current_f1 = val_metrics['macro']['f1']
        
        print(f"Validation Macro F1: {current_f1:.4f}")
        
        # Save best model
        if current_f1 > best_f1:
            best_f1 = current_f1
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': current_f1,
                'per_class_f1': val_metrics['per_class']['f1'].tolist()
            }
            torch.save(save_dict, os.path.join(save_dir, 'best_model_finetuned.pth'))
            print("New best model saved!")
        
        scheduler.step(current_f1)
        
    print("\nTraining completed!")
    print(f"Best Validation F1: {best_f1:.4f}")
    
    # Load best model and perform final evaluation
    best_model = load_pretrained_model_and_add_classifier(
        os.path.join(save_dir, 'best_model_finetuned.pth'), device)
    final_metrics = evaluate(best_model, val_loader, device)
    
    return best_f1, final_metrics

def main(cfg, gpu, save_dir):
    device = torch.device(cfg['DEVICE'])
    train_cfg = cfg['TRAIN']
    dataset_cfg = cfg['DATASET']
    num_workers = mp.cpu_count()
    
    # Model setup...
    model = load_pretrained_model_and_add_classifier('output/best_model.pth', device)
    model = model.to(device)
    
    # Dataset and DataLoader setup
    image_size = [256,256]
    train_transform = get_train_augmentation(image_size)
    val_transform = get_val_transform(image_size)
    batch_size = 32

    dataset = eval(dataset_cfg['NAME'])(
        dataset_cfg['ROOT'] + '/cropped_images_1424x1648',
        dataset_cfg['TRAIN_RATIO'],
        dataset_cfg['VALID_RATIO'],
        dataset_cfg['TEST_RATIO'],
        transform=None
    )
    trainset, valset, _ = dataset.get_splits()
    trainset.transform = train_transform
    valset.transform = val_transform

    trainloader = DataLoader(
        trainset, 
        batch_sampler=BalancedBatchSampler(trainset, batch_size=batch_size),
        num_workers=num_workers,
        pin_memory=True
    )
    
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False  # Important for consistent evaluation
    )
    
    # Training setup
    weight_decay = 1e-4
    # 학습 설정
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # 전체 모델 학습
    criterion = nn.BCEWithLogitsLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    scaler = GradScaler(enabled=train_cfg['AMP'])
    
    best_f1, final_metrics = train_and_evaluate(
        model=model,
        train_loader=trainloader,
        val_loader=valloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        criterion=criterion,  # criterion 전달
        device=device,
        epochs=100,
        save_dir=save_dir
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()