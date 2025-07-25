import torch
import numpy as np
import yaml
import torch.nn as nn
import torch.optim as optim
import fire
import os
from pathlib import Path

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets
from torchmetrics.retrieval import RetrievalRecall, RetrievalPrecision

from custom_datasets import LoadFlickr30K
from encoders import ResNetEncoder, BertEncoder, ViTEncoder, SentenceBertEncoder
from loss_func import DistanceBasedContrastiveLoss, get_distance

def evaluate(dataloader, vision_encoder, text_encoder, metric, device, top_k=1):
    vision_encoder.eval()
    text_encoder.eval()
    
    recall = RetrievalRecall(top_k=top_k)
    precision = RetrievalPrecision(top_k=top_k)
    
    v_mean_all = []
    t_mean_all = []
    v_var_all = []
    t_var_all = []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['images'].to(device)
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_masks'].squeeze(1).to(device)

            v_mean, v_var = vision_encoder(imgs)
            t_mean, t_var = text_encoder(input_ids, attention_mask)
            
            v_mean_all.append(v_mean)
            t_mean_all.append(t_mean)
            v_var_all.append(v_var)
            t_var_all.append(t_var)
        
        v_mean = torch.cat(v_mean_all)
        t_mean = torch.cat(t_mean_all)
        v_var = torch.cat(v_var_all)
        t_var = torch.cat(t_var_all)

        preds = get_distance(v_mean, v_var, t_mean, t_var, metric)

        targets = torch.eye(preds.size(0), dtype=torch.int).to(device)
        indexes = torch.arange(preds.size(0), dtype=torch.long).unsqueeze(1).expand(*preds.size()).to(device)
       
        i2t_r = recall(preds, targets, indexes)
        t2i_r = recall(preds.T, targets, indexes)
        r_val = (i2t_r + t2i_r) / 2
        print(f'Recall: image-to-text: {i2t_r}, text-to-image: {t2i_r}')

        i2t_p = precision(preds, targets, indexes)
        t2i_p = precision(preds.T, targets, indexes)
        p_val = (i2t_p + t2i_p) / 2
        print(f'Precision: image-to-text: {i2t_p}, text-to-image: {t2i_p}')

    return r_val, p_val

def get_default_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(state, output_dir, epoch, vision_name, text_name, metric, batch_size):
    """Saves training checkpoint"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"checkpoint-{vision_name}-{text_name}-{metric}-e{epoch}-b{batch_size}.pth"
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    return filepath

def load_checkpoint(checkpoint_path, vision_encoder, text_encoder, contrastive_loss, optimizer, scheduler, scaler):
    """Loads training checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    vision_encoder.load_state_dict(checkpoint['vision_model'])
    text_encoder.load_state_dict(checkpoint['language_model'])
    contrastive_loss.load_state_dict(checkpoint['temperature'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch,vision_encoder,text_encoder,contrastive_loss,optimizer,scheduler,scaler

def train(
    config: str = "config.yaml",
    tokenizer_name: str = None,
    batch_size: int = None,
    lr: float = None,
    embed_dim: int = None,
    init_temp: float = None,
    epochs: int = None,
    metric: str = None,
    reg_weight: float = None,
    output_dir: str = "./checkpoints",
    resume_ckpt_path: str = None,
    checkpoint_interval: int = 10
):
    """Train vision-text retrieval model with checkpoint support
    
    Args:
        config: Path to YAML configuration file
        tokenizer_name: Override tokenizer name in config
        batch_size: Override batch size in config
        lr: Override learning rate in config
        embed_dim: Override embedding dimension in config
        init_temp: Override initial temperature in config
        epochs: Override number of epochs in config
        metric: Override distance metric in config
        reg_weight: Override regularization weight in config
        output_dir: Directory to save checkpoints
        resume: Path to checkpoint to resume training
        checkpoint_interval: Save checkpoint every N epochs
    """
    cfg = get_default_config(config)
    
    # Apply command-line overrides
    if tokenizer_name is None:
        tokenizer_name=cfg['TOKENIZER_NAME']
    if batch_size is None:
        batch_size=cfg['BATCH_SIZE']
    if lr is None:
        lr=cfg['LR'] = lr
    if embed_dim is None:
        embed_dim=cfg['EMBED_DIM'] 
    if init_temp is None:
        init_temp=cfg['INIT_TEMP'] 
    if epochs is None:
        epochs=cfg['EPOCHS'] 
    if metric is None:
        metric=cfg['METRIC'] 
    if reg_weight is None:
        reg_weight=cfg['REG_WEIGHT'] 

    # Data loading
    train_loader, val_loader, test_loader = LoadFlickr30K(
        TOKENIZER_NAME=tokenizer_name,
        BATCH_SIZE=batch_size
    ).get_loaders()
    
    print(f'Batch size: train={train_loader.batch_size}, val={val_loader.batch_size}, test={test_loader.batch_size}')
    print(f"Using metric: {metric}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model initialization
    vision_encoder = ViTEncoder(embed_dim).to(device)
    text_encoder = SentenceBertEncoder(embed_dim).to(device)
    contrastive_loss = DistanceBasedContrastiveLoss(
        init_temp=init_temp,
        metric=metric
    ).to(device)

    # Optimizer setup
    optimizer = optim.AdamW(
        list(vision_encoder.parameters()) + 
        list(text_encoder.parameters()) + 
        list(contrastive_loss.parameters()),
        lr=lr
    )

    # Training utilities
    scaler = torch.amp.GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs* len(train_loader))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_ckpt_path:
        start_epoch,vision_encoder,text_encoder,contrastive_loss,optimizer,scheduler,scaler = load_checkpoint(
            resume_ckpt_path,
            vision_encoder,
            text_encoder,
            contrastive_loss,
            optimizer,
            scheduler,
            scaler
        )

    # Training loop
    for epoch in tqdm(range(start_epoch, epochs), desc="Training"):
        vision_encoder.train()
        text_encoder.train()
        total_loss = 0
        
        for batch in train_loader:
            imgs = batch['images'].to(device)
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_masks'].squeeze(1).to(device)
            
            with torch.amp.autocast('cuda'):
                v_mean, v_var = vision_encoder(imgs)
                t_mean, t_var = text_encoder(input_ids, attention_mask)
                loss = contrastive_loss(
                    v_mean, v_var, t_mean, t_var,
                    reg_weight=reg_weight
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            #print(f"Current LR = {current_lr}")

            total_loss += loss.item()
        
        # Training diagnostics
        print(f'[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}')
        print(f'Mean norms: Vision={v_mean.norm(dim=-1).mean().item():.2f} | Text={t_mean.norm(dim=-1).mean().item():.2f}')
        print(f'Var norms: Vision={v_var.norm(dim=-1).mean().item():.2f} | Text={t_var.norm(dim=-1).mean().item():.2f}')
        print(f'Temperature: {1/contrastive_loss.log_temp.exp().item():.4f}')
        print(f'Weights: Mean={contrastive_loss.mean_weight.item():.4f} | Var={contrastive_loss.var_weight.item():.4f}')
        
        # Periodic evaluation and checkpointing
        if (epoch + 1) % checkpoint_interval == 0 or epoch == cfg['epochs'] - 1:
            recall, precision = evaluate(
                test_loader, vision_encoder, text_encoder,
                cfg['metric'], device, top_k=1
            )
            print(f"Top-1 Metrics: Recall={recall:.4f} | Precision={precision:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'vision_model': vision_encoder.state_dict(),
                'language_model': text_encoder.state_dict(),
                'temperature': contrastive_loss.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'config': cfg
            }
            
            save_checkpoint(
                checkpoint,
                output_dir,
                epoch + 1,
                vision_encoder.__class__.__name__,
                text_encoder.__class__.__name__,
                metric,
                batch_size
            )

    print('Training DONE.')

if __name__ == '__main__':
    fire.Fire(train)
