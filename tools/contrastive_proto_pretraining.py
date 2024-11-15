import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from nmc.models.heads import MultiHeadEmbedding
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

def train_epoch(model, dataloader, optimizer, scaler, device, 
                temperature=0.1, gradient_strategy='pcgrad'):
    model.train()
    total_loss = 0
    total_contra_loss = 0
    
    grad_manager = GradientManager(strategy=gradient_strategy)
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        num_classes = labels.shape[1]
        
        head_grads = defaultdict(list)
        backbone_grads = defaultdict(list)
        task_losses = []
        
        optimizer.zero_grad()
        
        with autocast(enabled=scaler is not None):
            embeddings = model(images)  # [batch_size, num_classes, embedding_dim]
            embeddings = F.normalize(embeddings, p=2, dim=2)
        
        for class_idx in range(num_classes):
            with autocast(enabled=scaler is not None):
                class_embeddings = embeddings[:, class_idx, :]
                class_labels_single = labels[:, class_idx]
                
                positive_mask = class_labels_single == 1
                negative_mask = class_labels_single == 0
                
                task_loss = torch.tensor(0.0).to(device)
                if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                    pos_embeddings = class_embeddings[positive_mask]
                    neg_embeddings = class_embeddings[negative_mask]
                    task_loss = calculate_class_contra_loss(
                        pos_embeddings=pos_embeddings,
                        neg_embeddings=neg_embeddings,
                        temperature=temperature
                    )
                
                if task_loss > 0:
                    task_losses.append(task_loss)
                    
                    if scaler is not None:
                        scaler.scale(task_loss).backward(retain_graph=True)
                    else:
                        task_loss.backward(retain_graph=True)
                    
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if 'classifier' in name:
                                if f'classifier.{class_idx}' in name:
                                    head_grads[name] = param.grad.clone()
                            else:
                                backbone_grads[name].append(param.grad.clone())
                    
                    optimizer.zero_grad()
                    total_contra_loss += task_loss.item()
        
        if task_losses:  # Only proceed if we have valid losses
            with torch.no_grad():
                combined_backbone_grads = grad_manager.combine_gradients(backbone_grads, task_losses)
                
                for name, param in model.named_parameters():
                    if name in combined_backbone_grads:
                        param.grad = combined_backbone_grads[name]
                    elif name in head_grads:
                        param.grad = head_grads[name]
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            total_loss += sum(loss.item() for loss in task_losses)
    
    n_batches = len(dataloader)
    return {
        'total_loss': total_loss / (n_batches * num_classes),
        'contra_loss': total_contra_loss / (n_batches * num_classes)
    }
    
def train(model, train_loader, optimizer, scheduler, scaler,
          device, epochs, save_dir, gradient_strategy='pcgrad'):
    print("Start Training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            gradient_strategy=gradient_strategy
        )
        
        print(f"Training Loss: {metrics['total_loss']:.4f}")
        print(f"Contrastive Loss: {metrics['contra_loss']:.4f}")
        
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['total_loss'],
                'contra_loss': metrics['contra_loss']
            }
            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))
            print("New best model saved!")
        
        scheduler.step(metrics['total_loss'])
        print()
    
    print("\nTraining completed!")
    print(f"Best Training Loss: {best_loss:.4f}")
    return best_loss

def main(cfg, gpu, save_dir):
    device = torch.device(cfg['DEVICE'])
    train_cfg = cfg['TRAIN']
    dataset_cfg = cfg['DATASET']
    num_workers = mp.cpu_count()
    
    # Model
    efficientnet = models.efficientnet_v2_m(pretrained=True)
    num_ftrs = efficientnet.classifier[1].in_features
    num_classes = 7
    embedding_dim = 256
    efficientnet.classifier = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        MultiHeadEmbedding(num_ftrs, num_classes, embedding_dim)
    )
    efficientnet = efficientnet.to(device)
    print("Model initialized")
    
    # Dataset and DataLoader
    image_size = [256,256]
    train_transform = get_train_augmentation(image_size)
    batch_size = 32

    dataset = eval(dataset_cfg['NAME'])(
        dataset_cfg['ROOT'] + '/cropped_images_1424x1648',
        dataset_cfg['TRAIN_RATIO'],
        dataset_cfg['VALID_RATIO'],
        dataset_cfg['TEST_RATIO'],
        transform=None
    )
    trainset, _, _ = dataset.get_splits()
    trainset.transform = train_transform

    trainloader = DataLoader(
        trainset, 
        batch_sampler=BalancedBatchSampler(trainset, batch_size=batch_size),
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Training setup
    weight_decay = 1e-4
    optimizer = optim.AdamW(efficientnet.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    scaler = GradScaler(enabled=train_cfg['AMP'])

    # Train
    best_loss = train(
        model=efficientnet,
        train_loader=trainloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        epochs=100,
        save_dir=save_dir,
        gradient_strategy='weighted'  # 'average', 'weighted', 'normalized', 'pcgrad'
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