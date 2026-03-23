"""
Training script for Parkinson's disease scale prediction using GraphMamba.

Supports YAML config, 5-fold cross-validation, checkpointing, TensorBoard,
contrastive learning, LR warmup, per-class metrics, weighted loss, acceptable
accuracy, and DataParallel multi-GPU training.
"""
import os
import sys
import argparse
import time
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.graphmamba import GraphMamba
from dataloader.loader import ParkinsonDataModule

class TeeLogger:
    """Mirror stdout to console and log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # flush to disk
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_logger(fold_dir):
    """Configure file logging."""
    log_file = os.path.join(fold_dir, 'log.txt')
    
    # Log start time
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    # Redirect stdout
    tee = TeeLogger(log_file)
    sys.stdout = tee
    
    print(f"[OK] Logging to: {log_file}")
    
    return tee

class AverageMeter:
    """Running average meter."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save config to disk."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Train GraphMamba for Parkinson\'s Disease Scale Prediction')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to train (overrides config)')
    parser.add_argument('--gpu', type=str, default=None,
                       help='GPU ids to use (e.g., "0,1" for multi-GPU)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (overrides config)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate model (overrides config)')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(gpu_ids: str):
    """
    Configure CUDA device(s).

    Args:
        gpu_ids: Comma-separated GPU ids, e.g. "0" or "0,1".

    Returns:
        device: primary torch.device
        device_ids: list of GPU ids
    """
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        return torch.device('cpu'), []
    
    # Parse GPU ids
    if gpu_ids is None or gpu_ids == '':
        device_ids = [0]
    else:
        device_ids = [int(x) for x in gpu_ids.split(',')]
    
    # Primary device
    device = torch.device(f'cuda:{device_ids[0]}')
    
    # Print GPU info
    print(f"\n{'='*60}")
    print("GPU Configuration:")
    print(f"{'='*60}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPUs: {device_ids}")
    for gpu_id in device_ids:
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    print(f"Primary device: {device}")
    print(f"{'='*60}\n")
    
    return device, device_ids


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Build optimizer from config."""
    opt_name = config['training']['optimizer']
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    
    if opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]):
    """Build LR scheduler (optional warmup)."""
    sched_name = config['training']['scheduler']
    sched_args = config['training'].get('scheduler_args', {}).get(sched_name, {})
    
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    
    if sched_name == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_args.get('step_size', 30),
            gamma=sched_args.get('gamma', 0.1)
        )
    elif sched_name == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_args.get('T_max', config['training']['epochs']),
            eta_min=sched_args.get('eta_min', 0)
        )
    elif sched_name == 'plateau':
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_args.get('mode', 'max'),
            factor=sched_args.get('factor', 0.5),
            patience=sched_args.get('patience', 10),
            verbose=sched_args.get('verbose', True)
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")
    
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        if sched_name == 'plateau':
            scheduler = warmup_scheduler
            scheduler.main_scheduler = main_scheduler
            scheduler.warmup_epochs = warmup_epochs
            print(f"[INFO] Using {warmup_epochs}-epoch warmup + {sched_name} scheduler")
        else:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            print(f"[INFO] Using {warmup_epochs}-epoch warmup + {sched_name} scheduler")
    else:
        scheduler = main_scheduler
        print(f"[INFO] Using {sched_name} scheduler (no warmup)")
    
    return scheduler


def compute_class_weights(train_loader, num_classes, device, method='inverse'):
    """Compute class weights for imbalance."""
    print("\n" + "="*60)
    print("Computing Class Weights for Imbalanced Data")
    print("="*60)
    
    class_counts = torch.zeros(num_classes)
    total_samples = 0
    
    for _, _, labels, _ in train_loader:
        for label in labels:
            class_counts[label.item()] += 1
            total_samples += 1
    
    print(f"Total samples: {int(total_samples)}")
    print(f"Class distribution:")
    for class_id in range(num_classes):
        count = int(class_counts[class_id])
        percentage = count / total_samples * 100
        print(f"  Class {class_id}: {count:4d} samples ({percentage:5.2f}%)")
    
    if method == 'inverse':
        class_weights = 1.0 / class_counts
        print(f"\nUsing inverse weighting: w_i = 1 / count_i")
    elif method == 'sqrt_inverse':
        class_weights = 1.0 / torch.sqrt(class_counts)
        print(f"\nUsing sqrt inverse weighting: w_i = 1 / sqrt(count_i)")
    elif method == 'effective_num':
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        class_weights = (1.0 - beta) / effective_num
        print(f"\nUsing Effective Number weighting (beta={beta})")
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"\nNormalized class weights:")
    for class_id in range(num_classes):
        print(f"  Class {class_id}: {class_weights[class_id]:.4f}")
    
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 2:
        print("[WARN] Class distribution is relatively balanced")
    elif imbalance_ratio < 5:
        print("[WARN] Moderate class imbalance detected")
    else:
        print("[WARN] Severe class imbalance detected!")
    
    print("="*60)
    
    return class_weights.to(device)


def compute_acceptable_accuracy(predictions, labels, tolerance=1, num_classes=4):
    """
    Acceptable accuracy: fraction of predictions within tolerance of label.

    Args:
        predictions: predicted labels [N]
        labels: ground-truth labels [N]
        tolerance: allowed label distance (default +/-1)
        num_classes: number of classes

    Returns:
        acceptable_acc: overall acceptable accuracy
        acceptable_acc_per_class: per-class acceptable accuracy
    """
    errors = np.abs(predictions - labels)
    acceptable = errors <= tolerance
    acceptable_acc = acceptable.sum() / len(labels) if len(labels) > 0 else 0.0
    
    acceptable_acc_per_class = []
    for class_id in range(num_classes):
        class_mask = labels == class_id
        if class_mask.sum() > 0:
            class_acceptable = acceptable[class_mask].sum() / class_mask.sum()
            acceptable_acc_per_class.append(class_acceptable)
        else:
            acceptable_acc_per_class.append(0.0)
    
    return acceptable_acc, acceptable_acc_per_class


def graphmamba_forward(
    model: nn.Module,
    data: torch.Tensor,
    *,
    use_hht_injection: bool,
    hht_feat: Optional[torch.Tensor],
    y: Optional[torch.Tensor] = None,
    get_cl_loss: bool = False,
    return_rep: bool = False,
):
    """
    Single entry for single-GPU vs DataParallel; pass hht_feat only if use_hht_injection.
    """
    kwargs: Dict[str, Any] = {}
    if use_hht_injection:
        kwargs['hht_feat'] = hht_feat
    if get_cl_loss:
        return model(data, y=y, get_cl_loss=True, **kwargs)
    return model(data, return_rep=return_rep, **kwargs)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, writer=None, scaler=None):
    """Train one epoch (DataParallel-safe)."""
    model.train()
    
    losses = AverageMeter()
    ce_losses = AverageMeter()
    cl_losses = AverageMeter()
    accs = AverageMeter()
    
    use_amp = config['training'].get('use_amp', False)
    
    use_cl = config['model'].get('cl_mode') is not None
    use_hht = config['model'].get('use_hht_injection', False)
    w_cl_loss = config['training'].get('w_cl_loss', 0.1)
    start_cl_epoch = config['training'].get('start_cl_epoch', 0)
    enable_cl = use_cl and epoch >= start_cl_epoch
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", 
                ncols=120, ascii=True, leave=True)
    
    for batch_idx, (data, hht_feat, labels, patient_ids) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)
        if use_hht:
            hht_feat = hht_feat.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                if enable_cl:
                    logits, cl_loss = graphmamba_forward(
                        model, data,
                        use_hht_injection=use_hht,
                        hht_feat=hht_feat if use_hht else None,
                        y=labels,
                        get_cl_loss=True,
                    )
                    
                    # DataParallel may return per-GPU cl_loss; reduce to scalar
                    if isinstance(cl_loss, torch.Tensor) and cl_loss.dim() > 0:
                        cl_loss = cl_loss.mean()
                    
                    ce_loss = criterion(logits, labels)
                    loss = ce_loss + w_cl_loss * cl_loss
                else:
                    logits = graphmamba_forward(
                        model, data,
                        use_hht_injection=use_hht,
                        hht_feat=hht_feat if use_hht else None,
                        return_rep=False,
                    )
                    ce_loss = criterion(logits, labels)
                    loss = ce_loss
                    cl_loss = torch.tensor(0.0)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['grad_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            if enable_cl:
                logits, cl_loss = graphmamba_forward(
                    model, data,
                    use_hht_injection=use_hht,
                    hht_feat=hht_feat if use_hht else None,
                    y=labels,
                    get_cl_loss=True,
                )
                
                # DataParallel may return per-GPU cl_loss; reduce to scalar
                if isinstance(cl_loss, torch.Tensor) and cl_loss.dim() > 0:
                    cl_loss = cl_loss.mean()
                
                ce_loss = criterion(logits, labels)
                loss = ce_loss + w_cl_loss * cl_loss
            else:
                logits = graphmamba_forward(
                    model, data,
                    use_hht_injection=use_hht,
                    hht_feat=hht_feat if use_hht else None,
                    return_rep=False,
                )
                ce_loss = criterion(logits, labels)
                loss = ce_loss
                cl_loss = torch.tensor(0.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['grad_clip_norm'])
            optimizer.step()
        
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
        
        losses.update(loss.item(), data.size(0))
        ce_losses.update(ce_loss.item(), data.size(0))
        if enable_cl:
            cl_losses.update(cl_loss.item() if isinstance(cl_loss, torch.Tensor) else cl_loss, data.size(0))
        accs.update(acc.item(), data.size(0))
        
        if enable_cl:
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'CE': f'{ce_losses.avg:.4f}',
                'CL': f'{cl_losses.avg:.4f}',
                'Acc': f'{accs.avg:.4f}'
            })
        else:
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accs.avg:.4f}'
            })
    
    pbar.close()
    
    if writer is not None and config['logging']['use_tensorboard']:
        writer.add_scalar('Train/Total_Loss', losses.avg, epoch)
        writer.add_scalar('Train/CE_Loss', ce_losses.avg, epoch)
        if enable_cl:
            writer.add_scalar('Train/CL_Loss', cl_losses.avg, epoch)
        writer.add_scalar('Train/Accuracy', accs.avg, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
    
    return losses.avg, accs.avg


def validate(model, val_loader, criterion, device, epoch, config, writer=None):
    """Validation loop."""
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    all_preds = []
    all_labels = []
    use_hht = config['model'].get('use_hht_injection', False)
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]", 
                ncols=120, ascii=True, leave=True)
    
    with torch.no_grad():
        for data, hht_feat, labels, patient_ids in pbar:
            data = data.to(device)
            labels = labels.to(device)
            if use_hht:
                hht_feat = hht_feat.to(device)
            
            logits = graphmamba_forward(
                model, data,
                use_hht_injection=use_hht,
                hht_feat=hht_feat if use_hht else None,
                return_rep=False,
            )
            loss = criterion(logits, labels)
            
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
            
            losses.update(loss.item(), data.size(0))
            accs.update(acc.item(), data.size(0))
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accs.avg:.4f}'
            })
    
    pbar.close()
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # num_classes from config
    num_classes = config['model']['num_class']
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Include all class labels for per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    
    # Pass num_classes for acceptable accuracy
    acceptable_acc, acceptable_acc_per_class = compute_acceptable_accuracy(
        all_preds, all_labels, tolerance=1, num_classes=num_classes
    )
    
    acc_per_class = []
    for class_id in range(num_classes):
        class_mask = all_labels == class_id
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == class_id).sum() / class_mask.sum()
            acc_per_class.append(class_acc)
        else:
            acc_per_class.append(0.0)
    
    print("\n" + "="*90)
    print("Per-Class Metrics:")
    print("="*90)
    print(f"{'Class':<8} {'Support':<10} {'Accuracy':<12} {'Accept.Acc':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*90)
    
    for class_id in range(num_classes):
        # Bounds-safe indexing
        sup = int(support[class_id]) if class_id < len(support) else 0
        acc = acc_per_class[class_id] if class_id < len(acc_per_class) else 0.0
        acc_acc = acceptable_acc_per_class[class_id] if class_id < len(acceptable_acc_per_class) else 0.0
        prec = precision_per_class[class_id] if class_id < len(precision_per_class) else 0.0
        rec = recall_per_class[class_id] if class_id < len(recall_per_class) else 0.0
        f1 = f1_per_class[class_id] if class_id < len(f1_per_class) else 0.0
        
        print(f"Class {class_id:<3} "
              f"{sup:<10} "
              f"{acc:.4f}       "
              f"{acc_acc:.4f}       "
              f"{prec:.4f}       "
              f"{rec:.4f}       "
              f"{f1:.4f}")
    
    print("-"*90)
    print(f"{'Overall':<8} "
          f"{int(support.sum()):<10} "
          f"{accs.avg:.4f}       "
          f"{acceptable_acc:.4f}       "
          f"{precision_weighted:.4f}       "
          f"{recall_weighted:.4f}       "
          f"{f1_weighted:.4f}")
    print("="*90)
    
    if writer is not None and config['logging']['use_tensorboard']:
        writer.add_scalar('Val/Loss', losses.avg, epoch)
        writer.add_scalar('Val/Accuracy', accs.avg, epoch)
        writer.add_scalar('Val/Acceptable_Accuracy', acceptable_acc, epoch)
        writer.add_scalar('Val/Precision_Weighted', precision_weighted, epoch)
        writer.add_scalar('Val/Recall_Weighted', recall_weighted, epoch)
        writer.add_scalar('Val/F1_Weighted', f1_weighted, epoch)
        
        for class_id in range(num_classes):
            writer.add_scalar(f'Val_PerClass/Accuracy_Class{class_id}', acc_per_class[class_id], epoch)
            writer.add_scalar(f'Val_PerClass/Acceptable_Acc_Class{class_id}', acceptable_acc_per_class[class_id], epoch)
            writer.add_scalar(f'Val_PerClass/Precision_Class{class_id}', precision_per_class[class_id], epoch)
            writer.add_scalar(f'Val_PerClass/Recall_Class{class_id}', recall_per_class[class_id], epoch)
            writer.add_scalar(f'Val_PerClass/F1_Class{class_id}', f1_per_class[class_id], epoch)
    
    return (losses.avg, accs.avg, precision_weighted, recall_weighted, f1_weighted, 
            all_preds, all_labels, acc_per_class, precision_per_class, recall_per_class, f1_per_class, support,
            acceptable_acc, acceptable_acc_per_class)


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth.tar', is_best=False, keep_last_n=5):
    """Save training checkpoint."""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        print(f"[OK] Saved best model to {best_filepath}")
    
    if keep_last_n > 0:
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')])
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n]:
                os.remove(os.path.join(checkpoint_dir, old_ckpt))


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross-entropy.
    """
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        
        log_prob = F.log_softmax(pred, dim=1)
        
        if self.weight is not None:
            # Apply class weights
            weighted_loss = -(one_hot * log_prob * self.weight.unsqueeze(0)).sum(dim=1)
            return weighted_loss.mean()
        else:
            return -(one_hot * log_prob).sum(dim=1).mean()
        
def train_single_fold(fold: int, config: Dict[str, Any], device: torch.device, device_ids: list):
    """Train one fold or one train/test split (DataParallel-safe)."""
    print("\n" + "="*80)
    if config['data'].get('use_split', False):
        print(f"Training with Pre-Split Data (Train/Test)")
    else:
        print(f"Training Fold {fold}/{config['cross_validation']['n_folds']-1} (K-Fold)")
    
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{config['experiment']['name']}_{timestamp}"
    if config['data'].get('use_split', False):
        fold_dir = os.path.join(config['checkpoint']['dir'], exp_name, 'train_test_split')
    else:
        fold_dir = os.path.join(config['checkpoint']['dir'], exp_name, f'fold_{fold}')
    
    os.makedirs(fold_dir, exist_ok=True)
    
    tee_logger = setup_logger(fold_dir)
    
    try:        
        writer = None
        if config['logging']['use_tensorboard']:
            writer = SummaryWriter(log_dir=os.path.join(fold_dir, 'logs'))
        
        save_config(config, os.path.join(fold_dir, 'config.yaml'))
        
        use_split = config['data'].get('use_split', False)
        
        data_module = ParkinsonDataModule(
            data_dir=config['data'].get('data_dir', ''),  # K-fold: data root
            action_type=config['data'].get('action_type', 'LA'),
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            seed=config['experiment']['seed'],
            use_split=use_split,  # pre-split train/test
            split_dir=config['data'].get('split_dir', None),  # split manifest dir
            use_hht_features=config['data'].get('use_hht_features', False),
            hht_subdir=config['data'].get('hht_subdir', 'hht'),
            hht_stem_suffix=config['data'].get('hht_stem_suffix', '_hht'),
            hht_feature_dim=config['data'].get('hht_feature_dim', 10),
        )
        
        model_cfg_pre = config['model']
        if model_cfg_pre.get('use_hht_injection', False):
            fd = config['data'].get('hht_feature_dim', 10)
            hc = model_cfg_pre.get('hht_in_channels', 10)
            if fd != hc:
                raise ValueError(
                    f"data.hht_feature_dim ({fd}) must match model.hht_in_channels ({hc}) when use_hht_injection is True"
                )

        if use_split:
            train_loader, val_loader = data_module.get_train_test_dataloaders()
        else:
            train_loader, val_loader = data_module.get_fold_dataloaders(
                fold=fold,
                n_splits=config['cross_validation']['n_folds'],
                use_stratified=config['cross_validation'].get('use_stratified', True),
            )
        
        class_weights = None
        if config['training'].get('use_class_weights', False):
            weight_method = config['training'].get('class_weight_method', 'inverse')
            class_weights = compute_class_weights(
                train_loader, 
                config['model']['num_class'], 
                device,
                method=weight_method
            )
        
        # --- Model ---
        model_config = config['model']
        model = GraphMamba(
            num_class=model_config['num_class'],
            num_point=model_config['num_point'],
            num_person=model_config['num_person'],
            num_frame=model_config['num_frame'],
            in_channels=model_config['in_channels'],
            graph=model_config['graph'],
            graph_args=model_config.get('graph_args', {}),
            drop_out=model_config['drop_out'],
            adaptive=model_config['adaptive'],
            alpha=model_config['alpha'],
            
            cl_mode=model_config.get('cl_mode', None),
            multi_cl_weights=model_config.get('multi_cl_weights', [0.1, 0.2, 0.5, 1.0]),
            cl_version=model_config.get('cl_version', 'V0'),
            use_hht_injection=model_config.get('use_hht_injection', False),
            hht_in_channels=model_config.get('hht_in_channels', 10),
        )
        
        # DataParallel wrap
        if len(device_ids) > 1:
            print(f"\n[OK] Using DataParallel with {len(device_ids)} GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        
        model = model.to(device)
        
        # Parameter count (use .module under DataParallel)
        if isinstance(model, nn.DataParallel):
            param_count = sum(p.numel() for p in model.module.parameters()) / 1e6
        else:
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
        
        print(f"Model parameters: {param_count:.2f}M")

        print("\n" + "="*60)
        if model_config.get('cl_mode') is not None:
            print(f"[OK] Contrastive Learning ENABLED")
            print(f"   Mode: {model_config['cl_mode']}")
            print(f"   Weights: {model_config.get('multi_cl_weights', [0.1, 0.2, 0.5, 1.0])}")
            print(f"   Start Epoch: {config['training'].get('start_cl_epoch', 0)}")
            print(f"   CL Loss Weight: {config['training'].get('w_cl_loss', 0.1)}")
        else:
            print(f"[WARN] Contrastive Learning DISABLED")
        print("="*60)
        
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        
        if class_weights is not None:
            if label_smoothing > 0:
                criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights)
                print(f"\n[OK] Using Label Smoothing ({label_smoothing}) + Weighted CE")
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                print(f"\n[OK] Using Weighted CrossEntropyLoss")
        else:
            if label_smoothing > 0:
                criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
                print(f"\n[OK] Using Label Smoothing ({label_smoothing})")
            else:
                criterion = nn.CrossEntropyLoss()
                print(f"\n[WARN] Using Standard CrossEntropyLoss")
        
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        
        scaler = torch.cuda.amp.GradScaler() if config['training'].get('use_amp', False) else None
        
        warmup_epochs = config['training'].get('warmup_epochs', 0)
        has_warmup = warmup_epochs > 0
        
        # --- Resume ---
        start_epoch = 0
        best_acc = 0.0
        
        if config['resume']['enabled'] and config['resume']['checkpoint_path']:
            resume_path = config['resume']['checkpoint_path']
            if os.path.isfile(resume_path):
                print(f"Loading checkpoint from {resume_path}")
                checkpoint = torch.load(resume_path, map_location=device)
                start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best_acc']
                
                # Load weights (strip DataParallel if needed)
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint['state_dict'])
                
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")
        
        # --- Training loop ---
        print("\nStarting training...")
        
        for epoch in range(start_epoch, config['training']['epochs']):
            print(f"\nEpoch [{epoch+1}/{config['training']['epochs']}]")
            print("-" * 60)
            
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, config, writer, scaler
            )
            
            (val_loss, val_acc, precision, recall, f1, 
            val_preds, val_labels, acc_per_class, precision_per_class, recall_per_class, f1_per_class, support,
            acceptable_acc, acceptable_acc_per_class) = validate(
                model, val_loader, criterion, device, epoch, config, writer
            )
            
            if config['training']['scheduler'] == 'plateau':
                if has_warmup and epoch < warmup_epochs:
                    scheduler.step()
                elif has_warmup and epoch == warmup_epochs:
                    scheduler = scheduler.main_scheduler
                    print(f"[INFO] Warmup completed, switching to main scheduler")
                else:
                    scheduler.step(val_acc)
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Val Acceptable Acc: {acceptable_acc:.4f}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"  Current LR: {current_lr:.6f}")
            
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            
            if (epoch + 1) % config['checkpoint']['save_freq'] == 0 or is_best:
                # Save state_dict from .module if DataParallel
                if isinstance(model, nn.DataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': state_dict,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'config': config
                }, fold_dir, 
                filename=f'checkpoint_epoch_{epoch+1}.pth.tar', 
                is_best=is_best,
                keep_last_n=config['checkpoint']['keep_last_n'])
            
            if (epoch + 1) % config['logging']['plot_confusion_matrix_freq'] == 0:
                cm = confusion_matrix(val_labels, val_preds)
                cm_path = os.path.join(fold_dir, f'confusion_matrix_epoch_{epoch+1}.png')
                plot_confusion_matrix(cm, [str(i) for i in range(config['model']['num_class'])], cm_path)
        
        # --- Final eval ---
        print("\n" + "="*80)
        print("Final Evaluation on Best Model")
        print("="*80)
        
        best_model_path = os.path.join(fold_dir, 'model_best.pth.tar')
        checkpoint = torch.load(best_model_path)
        
        # Load best weights for final eval
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        
        (val_loss, val_acc, precision, recall, f1, 
        val_preds, val_labels, acc_per_class, precision_per_class, recall_per_class, f1_per_class, support,
        acceptable_acc, acceptable_acc_per_class) = validate(
            model, val_loader, criterion, device, config['training']['epochs'], config, None
        )
        
        cm = confusion_matrix(val_labels, val_preds)
        cm_path = os.path.join(fold_dir, 'confusion_matrix_final.png')
        plot_confusion_matrix(cm, [str(i) for i in range(config['model']['num_class'])], cm_path)
        
        num_classes = config['model']['num_class']
        results = {
            'fold': fold,
            'best_epoch': checkpoint['epoch'],
            'best_acc': float(best_acc),
            'final_acc': float(val_acc),
            'acceptable_acc': float(acceptable_acc),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            
            'per_class_metrics': {
                f'class_{i}': {
                    'accuracy': float(acc_per_class[i]),
                    'acceptable_acc': float(acceptable_acc_per_class[i]),
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support[i])
                }
                for i in range(num_classes)
            },
            
            'confusion_matrix': cm.tolist()
        }
        
        with open(os.path.join(fold_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        if writer is not None:
            writer.close()
        
        print(f"\n[OK] Fold {fold} completed!")
        print(f"   Best Val Acc: {best_acc:.4f}")
        print(f"   Final Val Acc: {val_acc:.4f}")
        print(f"   Final Acceptable Acc: {acceptable_acc:.4f}")
        
        return results
    
    finally:
        # Tear down tee logger
        with open(os.path.join(fold_dir, 'log.txt'), 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
# Restore stdout
        sys.stdout = tee_logger.terminal
        tee_logger.close()

def main():
    """CLI entry."""
    args = get_args()
    
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    
    use_split = config['data'].get('use_split', False)
    
    if use_split:
        # Pre-split: ignore fold CLI
        print("\n[WARN] Using pre-split data, fold parameter ignored")
        config['cross_validation']['fold'] = 0  # placeholder fold index
    else:
        # K-fold: honor fold CLI
        if args.fold is not None:
            config['cross_validation']['fold'] = args.fold
    
    # GPU ids from CLI or config
    if args.gpu is not None:
        gpu_ids_str = args.gpu
    else:
        gpu_ids_str = str(config['experiment'].get('gpu', '0'))
    
    if args.resume is not None:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = args.resume
    if args.eval_only:
        config['evaluation']['eval_only'] = True
    
    set_seed(config['experiment']['seed'])
    
    # Device setup (multi-GPU supported)
    device, device_ids = setup_device(gpu_ids_str)
    
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("="*80)
    
    if use_split:
        # Pre-split: single train run
        results = train_single_fold(0, config, device, device_ids)
    else:    
        fold = config['cross_validation']['fold']
        
        if fold is not None:
            results = train_single_fold(fold, config, device, device_ids)
        else:
            all_results = []
            n_folds = config['cross_validation']['n_folds']
            
            for fold in range(n_folds):
                results = train_single_fold(fold, config, device, device_ids)
                all_results.append(results)
            
            # --- K-fold summary ---
            print("\n" + "="*80)
            print("5-FOLD CROSS-VALIDATION RESULTS")
            print("="*80)
            
            avg_acc = np.mean([r['final_acc'] for r in all_results])
            std_acc = np.std([r['final_acc'] for r in all_results])
            avg_acceptable_acc = np.mean([r['acceptable_acc'] for r in all_results])
            std_acceptable_acc = np.std([r['acceptable_acc'] for r in all_results])
            avg_precision = np.mean([r['precision_weighted'] for r in all_results])
            avg_recall = np.mean([r['recall_weighted'] for r in all_results])
            avg_f1 = np.mean([r['f1_weighted'] for r in all_results])
            
            print(f"\n=== Overall Metrics (Weighted Average) ===")
            print(f"Average Accuracy:           {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"Average Acceptable Accuracy: {avg_acceptable_acc:.4f} ± {std_acceptable_acc:.4f}")
            print(f"Average Precision:          {avg_precision:.4f}")
            print(f"Average Recall:             {avg_recall:.4f}")
            print(f"Average F1-Score:           {avg_f1:.4f}")
            
            print(f"\n=== Per-Class Metrics (Averaged over {n_folds} folds) ===")
            num_classes = config['model']['num_class']
            
            for class_id in range(num_classes):
                class_accs = [r['per_class_metrics'][f'class_{class_id}']['accuracy'] for r in all_results]
                class_acceptable_accs = [r['per_class_metrics'][f'class_{class_id}']['acceptable_acc'] for r in all_results]
                class_precs = [r['per_class_metrics'][f'class_{class_id}']['precision'] for r in all_results]
                class_recalls = [r['per_class_metrics'][f'class_{class_id}']['recall'] for r in all_results]
                class_f1s = [r['per_class_metrics'][f'class_{class_id}']['f1_score'] for r in all_results]
                
                print(f"\nClass {class_id}:")
                print(f"  Accuracy:           {np.mean(class_accs):.4f} ± {np.std(class_accs):.4f}")
                print(f"  Acceptable Acc:     {np.mean(class_acceptable_accs):.4f} ± {np.std(class_acceptable_accs):.4f}")
                print(f"  Precision:          {np.mean(class_precs):.4f} ± {np.std(class_precs):.4f}")
                print(f"  Recall:             {np.mean(class_recalls):.4f} ± {np.std(class_recalls):.4f}")
                print(f"  F1-Score:           {np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}")
            
            print("\n=== Per-fold Overall Results ===")
            for i, r in enumerate(all_results):
                print(f"  Fold {i}: Acc={r['final_acc']:.4f}, Accept.Acc={r['acceptable_acc']:.4f}, F1={r['f1_weighted']:.4f}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            exp_name = f"{config['experiment']['name']}_{timestamp}"
            summary_dir = os.path.join(config['checkpoint']['dir'], exp_name)
            summary_path = os.path.join(summary_dir, 'summary.json')
            
            summary = {
                'experiment': config['experiment'],
                'data': config['data'],
                'model': config['model'],
                'n_folds': n_folds,
                
                'overall_metrics': {
                    'avg_accuracy': float(avg_acc),
                    'std_accuracy': float(std_acc),
                    'avg_acceptable_acc': float(avg_acceptable_acc),
                    'std_acceptable_acc': float(std_acceptable_acc),
                    'avg_precision': float(avg_precision),
                    'avg_recall': float(avg_recall),
                    'avg_f1': float(avg_f1),
                },
                
                'per_class_summary': {
                    f'class_{class_id}': {
                        'avg_accuracy': float(np.mean([r['per_class_metrics'][f'class_{class_id}']['accuracy'] for r in all_results])),
                        'std_accuracy': float(np.std([r['per_class_metrics'][f'class_{class_id}']['accuracy'] for r in all_results])),
                        'avg_acceptable_acc': float(np.mean([r['per_class_metrics'][f'class_{class_id}']['acceptable_acc'] for r in all_results])),
                        'std_acceptable_acc': float(np.std([r['per_class_metrics'][f'class_{class_id}']['acceptable_acc'] for r in all_results])),
                        'avg_precision': float(np.mean([r['per_class_metrics'][f'class_{class_id}']['precision'] for r in all_results])),
                        'avg_recall': float(np.mean([r['per_class_metrics'][f'class_{class_id}']['recall'] for r in all_results])),
                        'avg_f1': float(np.mean([r['per_class_metrics'][f'class_{class_id}']['f1_score'] for r in all_results])),
                    }
                    for class_id in range(num_classes)
                },
                
                'fold_results': all_results
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n[OK] Summary saved to {summary_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()