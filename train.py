import torch
import numpy as np
import argparse, yaml
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms,datasets
from torchmetrics.retrieval import RetrievalRecall, RetrievalPrecision

from custom_datasets import LoadFlickr30K
from encoders import ResNetEncoder,BertEncoder,ViTEncoder,SentenceBertEncoder
from loss_func import W2ContrastiveLoss,wasserstein2_distance

def evaluate(dataloader,vision_encoder,text_encoder,device, top_k=1):
    vision_encoder.eval()
    text_encoder.eval()
    correct = 0
    total = 0
    
    recall = RetrievalRecall(top_k=top_k)
    precision = RetrievalPrecision(top_k=top_k)
    
    v_mean_all=[]
    t_mean_all=[]
    v_var_all=[]
    t_var_all=[]
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['images'].to(device)
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_masks'].squeeze(1).to(device)

            v_mean, v_var= vision_encoder(imgs)
            t_mean, t_var= text_encoder(input_ids, attention_mask)
            
            v_mean_all.append(v_mean)
            t_mean_all.append(t_mean)
            v_var_all.append(v_var)
            t_var_all.append(t_var)
        
        v_mean=torch.cat(v_mean_all)
        t_mean=torch.cat(t_mean_all)
        v_var=torch.cat(v_var_all)
        t_var=torch.cat(t_var_all)

        preds = -wasserstein2_distance(
            v_mean.unsqueeze(1), v_var.unsqueeze(1).sqrt(),
            t_mean.unsqueeze(0), t_var.unsqueeze(0).sqrt()
        )
        targets=torch.eye(preds.size(0), dtype=torch.int).to(device)
        indexes=torch.arange(preds.size(0), dtype=torch.long).unsqueeze(1).expand(*preds.size()).to(device)

        r_val=recall(preds,targets,indexes)
        p_val=precision(preds,targets,indexes)

    return r_val,p_val

def get_default_config(config_path):
    with open(config_path,'r') as f:
        return yaml.safe_load(f)

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--tokenizer_name",type=str)
    parser.add_argument("--batch_size", type=int, help="override batch size")
    parser.add_argument("--lr",type=float, help="override learning rate")
    parser.add_argument('--embed_dim',type=int,help='new embed dim')
    parser.add_argument('--init_temp',type=float,help='new temperature')
    parser.add_argument('--epochs',type=int,help='new epochs')
    args = parser.parse_args()

    cfg = get_default_config(args.config)

    if args.tokenizer_name:
        cfg['tokenizer_name']=args.tokenizer_name
    if args.batch_size:  
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["lr"] = args.lr
    if args.embed_dim:
        cfg['embed_dim']=agrs.embed_dim
    if args.init_temp:
        cfg['init_temp']=args.init_temp
    if args.epochs:
        cfg['epochs']=args.epochs
    
    return cfg

cfg=get_config()

train_loader,val_loader,test_loader=LoadFlickr30K(tokenizer_name=cfg['tokenizer_name'],BATCH_SIZE=cfg['batch_size']).get_loaders()
print(f'Batch size: train={train_loader.batch_size}, val={val_loader.batch_size}, test={test_loader.batch_size}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vision_encoder = ViTEncoder(cfg['embed_dim']).to(device)
text_encoder = SentenceBertEncoder(cfg['embed_dim']).to(device)
contrastive_loss=W2ContrastiveLoss(cfg['init_temp']).to(device)

'''
optimizer = optim.AdamW(
    [
        {'params': vision_encoder.mean.parameters()},
        {'params': vision_encoder.logvar.parameters()},
        {'params': text_encoder.mean.parameters()},
        {'params': text_encoder.logvar.parameters()}
    ],
    lr=cfg['lr'],
    #weight_decay=1e-4)
'''
optimizer=optim.AdamW(
list(vision_encoder.parameters())+list(text_encoder.parameters())+list(contrastive_loss.parameters()),
lr=cfg['lr']
)


scaler = torch.amp.GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs']*len(train_loader))

for epoch in tqdm(range(cfg['epochs'])):
    vision_encoder.train()
    text_encoder.train()
    total_loss = 0
    
    for batch in train_loader:
        imgs = batch['images'].to(device)
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_masks'].squeeze(1).to(device)
        with torch.amp.autocast('cuda'):
            # Forward pass
            v_mean, v_var= vision_encoder(imgs)
            t_mean, t_var= text_encoder(input_ids, attention_mask)

            # Compute loss
            loss = contrastive_loss(v_mean, v_var, t_mean, t_var,reg_weight=cfg['reg_weight'])
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        scheduler.step()

        total_loss += loss.item()
    
    print(f'Last batch mean norm: {v_mean.norm(dim=-1).mean(dim=-1).item():.2f} {t_mean.norm(dim=-1).mean(dim=-1).item():.2f} \n variance norm: {v_var.norm(dim=-1).mean(dim=-1).item():.2f} {t_var.norm(dim=-1).mean(dim=-1).item():.2f}')
    print(f'Learned temperature: {1/contrastive_loss.log_temp.exp().item()}')
    print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
    if (epoch+1)%10==0:
        recall,precision=evaluate(test_loader,vision_encoder,text_encoder,device, top_k=1)
        print(f"Top-1 Recall: {recall:.4f}; Precision: {precision:.4f}")

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict()
}, f"./checkpoints/checkpoint-{vision_encoder.__class__.__name__}-{text_encoder.__class__.__name__}-{cfg['epochs']}-b{cfg['batch_size']}.pth")
