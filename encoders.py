import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34,ResNet34_Weights,vit_b_16, ViT_B_16_Weights
from transformers import AutoModel, AutoTokenizer

class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.backbone.fc=nn.Identity()
        
        self.mean = nn.Linear(512, embed_dim)
        self.logvar = nn.Linear(512, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        #features=F.normalize(features,dim=1,p=2.0)
        variance = self.logvar(features).exp() + 1e-6
        return self.mean(features), variance

class ViTEncoder(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.vit=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads=nn.Identity()
        
        self.mean = nn.Linear(768, embed_dim)
        self.logvar = nn.Linear(768, embed_dim)

    def forward(self, x):
        features = self.vit(x)
        #features=F.normalize(features,dim=1,p=2.0)

        variance = self.logvar(features).exp()
        return self.mean(features), variance


class BertEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        self.mean = nn.Linear(768, embed_dim)
        self.logvar = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        #pooled=F.normalize(pooled,dim=1,p=2.0)
        
        variance = self.logvar(pooled).exp()
        return self.mean(pooled), variance

class SentenceBertEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

        self.mean = nn.Linear(self.model.config.hidden_size, embed_dim)
        self.logvar = nn.Linear(self.model.config.hidden_size, embed_dim)
    
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                               attention_mask=attention_mask)
        features = self.mean_pooling(outputs,attention_mask)

        variance = self.logvar(features).exp()
        return self.mean(features), variance

if __name__=='__main__':
    embed_dim=32
    vision_encoder=ViTEncoder(embed_dim)
    text_encoder=BertEncoder(embed_dim)
    mean,var=vision_encoder(torch.randn(1,3,224,224))
    print(mean.size())
