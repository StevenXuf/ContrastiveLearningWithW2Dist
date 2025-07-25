import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_expand(v_mean, v_var, t_mean, t_var):
    # Expand dimensions to (B1, 1, D) and (1, B2, D)
    v_mean = v_mean.unsqueeze(1)     # (B1, 1, D)
    v_var = v_var.unsqueeze(1)   # (B1, 1, D)
    t_mean = t_mean.unsqueeze(0)     # (1, B2, D)
    t_var = t_var.unsqueeze(0)   # (1, B2, D)
    return v_mean, v_var, t_mean, t_var

def wasserstein2_distance(v_mean, sigma1, t_mean, sigma2,mean_weight,var_weight):
    term1 = torch.sum((v_mean - t_mean)**2, dim=-1)
    term2 = torch.sum((sigma1-sigma2)**2, dim=-1)
    term3 = torch.sum(sigma1**2+sigma2**2,dim=-1)
    return mean_weight*term1+var_weight*term2

def covariance_regularizer(var):
    return torch.mean(torch.log(var)**2)

def get_distance(v_mean,v_var,t_mean,t_var,metric):
    v_mean,v_var,t_mean,t_var=pairwise_expand(v_mean,v_var,t_mean,t_var)
    if metric=='w2':
       distance=-wasserstein2_distance(
        v_mean, v_var.sqrt(),
        t_mean, t_var.sqrt())
    elif metric=='jeff':
        distance=-kl_divergence(v_mean,v_var,t_mean,t_var)-kl_divergence(t_mean, t_var, v_mean, v_var)
    elif metric=='bhat':
        distance=bhattacharyyadistance(v_mean,v_var,t_mean,t_var)
    elif metric=='hell':
        distance=-hellingerdistance(v_mean,v_var,t_mean,t_var)
    else:
        raise ValueError(f"Invalid metric: '{self.metric}'. Expected one of: 'w2', 'jeff', 'bhat', 'hell'")
    return distance

class DistanceBasedContrastiveLoss(nn.Module):
    def __init__(self,init_temp=0.07,mean_weight=1.0,var_weight=1.0,metric='w2'):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1/init_temp)))
        self.mean_weight=nn.Parameter(torch.tensor(mean_weight))
        self.var_weight=nn.Parameter(torch.tensor(var_weight))
    
        self.metric=metric

    def forward(self,v_mean, v_var, t_mean, t_var, reg_weight=0.001):
        batch_size = v_mean.size(0)
        temperature=self.log_temp.exp().clamp(max=100.0) 
        
        distance=get_distance(v_mean,v_var,t_mean,t_var,self.metric)
        logits = distance*temperature
        
        # Cross entropy losses
        labels = torch.arange(batch_size).to(logits.device)
        loss_i = nn.CrossEntropyLoss()(logits, labels)
        loss_t = nn.CrossEntropyLoss()(logits.T, labels)
        cl_loss = (loss_i + loss_t) / 2

        # Covariance regularization
        #reg_loss=.0
        reg_loss = reg_weight * (covariance_regularizer(v_var) + covariance_regularizer(t_var))

        return cl_loss + reg_loss


class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable temperature parameter (initialized at log(1/0.07))
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def forward(self, img_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_embed: Tensor of shape (batch_size, embed_dim)
            text_embed: Tensor of shape (batch_size, embed_dim)
        Returns:
            loss: Symmetric InfoNCE loss
        """
        # Normalize embeddings
        img_embed = F.normalize(img_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)

        # Compute cosine similarity matrix
        logits_per_image = img_embed @ text_embed.t()  # shape: (B, B)
        logits_per_text = logits_per_image.t()         # shape: (B, B)

        # Apply temperature scaling
        temperature = self.logit_scale.exp()
        logits_per_image = logits_per_image * temperature
        logits_per_text = logits_per_text * temperature

        # Ground-truth similarity (diagonal)
        batch_size = img_embed.size(0)
        labels = torch.arange(batch_size, device=img_embed.device)

        # Cross-entropy loss (image-to-text and text-to-image)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        # Final loss is the average
        return (loss_i2t + loss_t2i) / 2

def kl_divergence(v_mean, v_var, t_mean, t_var):
    v_var = v_var
    t_var = t_var
    term1 = torch.sum(torch.log(t_var / v_var),dim=-1)
    term2 = torch.sum((v_var + (v_mean - t_mean) ** 2) / t_var,dim=-1)
    return 0.5 * (term1 + term2 - v_mean.size(-1))
        
class JeffreysDivergenceLoss(nn.Module):
    def __init__(self,init_temp=.5):
        super(JeffreysDivergenceLoss, self).__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1/init_temp)))

    def forward(self, v_mean, v_var, t_mean, t_var):
        temperature=self.log_temp.exp().clamp(max=100.0)
        kl1 = kl_divergence(v_mean, v_var, t_mean, t_var)
        kl2 = kl_divergence(t_mean, t_var, v_mean, v_var)
        return (kl1 + kl2)*temperature


def bhattacharyyadistance(v_mean, v_var, t_mean, t_var):
    v_var = v_var+1.0e-8
    t_var = t_var+1.0e-8
    avg_var = 0.5 * (v_var + t_var)
    mean_diff = v_mean - t_mean

    term1 = 0.125 * torch.sum((mean_diff ** 2) / avg_var,dim=-1)
    term2 = 0.5 * torch.sum(torch.log(avg_var / torch.sqrt(v_var * t_var)),dim=-1)
    return term1 + term2

class BhattacharyyaDistanceLoss(nn.Module):
    def __init__(self,init_temp=.5):
        super(BhattacharyyaDistanceLoss, self).__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1/init_temp)))

    def forward(self, v_mean, v_var, t_mean, t_var):
        temperature=self.log_temp.exp().clamp(max=100.0)
        return bhattacharyyadistance(v_mean, v_var, t_mean, t_var)

def hellingerdistance(v_mean, v_var, t_mean, t_var):
    v_var = v_var
    t_var = t_var
    avg_var = 0.5 * (v_var + t_var)
    mean_diff = v_mean - t_mean

    sqrt_prod_var = torch.sqrt(v_var * t_var)
    sqrt_avg_var = torch.sqrt(avg_var)
    exp_term = torch.exp(-0.25 * torch.sum(mean_diff ** 2 / avg_var,dim=-1))

    bc = torch.prod(sqrt_prod_var,dim=-1) / torch.prod(sqrt_avg_var,dim=-1) * exp_term
    bc = torch.clamp(bc, min=1e-10, max=1.0)  # numerical stability

    return torch.sqrt(1.0 - bc)

class HellingerDistanceLoss(nn.Module):
    def __init__(self,init_temp=.5):
        super(HellingerDistanceLoss, self).__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1/init_temp)))

    def forward(self, v_mean, v_var, t_mean, t_var):
        temperature=self.log_temp.exp().clamp(max=100.0)
        return hellingerdistance(v_mean, v_var, t_mean, t_var)*temperature

if __name__=='__main__':
    v_mean=torch.randn(10,768).unsqueeze(1)
    v_var=torch.randn(10,768).exp().unsqueeze(1)
    t_mean=torch.randn(10,768).unsqueeze(0)
    t_var=torch.randn(10,768).exp().unsqueeze(0)

    jeff=JeffreysDivergenceLoss()
    bhat=BhattacharyyaDistanceLoss()
    hell=HellingerDistanceLoss()

    print(jeff(v_mean,v_var,t_mean,t_var).size())
    print(bhat(v_mean,v_var,t_mean,t_var).size())
    print(hell(v_mean,v_var,t_mean,t_var).size())

