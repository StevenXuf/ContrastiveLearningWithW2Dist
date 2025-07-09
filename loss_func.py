import torch
import torch.nn as nn
import torch.nn.functional as F

def wasserstein2_distance(mu1, sigma1, mu2, sigma2):
    term1 = torch.sum((mu1 - mu2)**2, dim=-1)
    term2 = torch.sum((sigma1-sigma2)**2, dim=-1)
    term3 = torch.sum(sigma1**2+sigma2**2,dim=-1)
    return term1+term2

def covariance_regularizer(var):
    return torch.mean(1/var) + torch.mean(var)

class W2ContrastiveLoss(nn.Module):
    def __init__(self,init_temp=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1/init_temp)))

    def forward(self,v_mean, v_var, t_mean, t_var, reg_weight=0.01):
        batch_size = v_mean.size(0)
        temperature=self.log_temp.exp().clamp(max=100.0) 
        # Compute pairwise distances
        logits = -wasserstein2_distance(
            v_mean.unsqueeze(1), v_var.unsqueeze(1).sqrt(),
            t_mean.unsqueeze(0), t_var.unsqueeze(0).sqrt()
        )*temperature
        
        # Cross entropy losses
        labels = torch.arange(batch_size).to(logits.device)
        loss_i = nn.CrossEntropyLoss()(logits, labels)
        loss_t = nn.CrossEntropyLoss()(logits.T, labels)
        cl_loss = (loss_i + loss_t) / 2

        # Covariance regularization
        reg_loss=.0
        #reg_loss = reg_weight * (covariance_regularizer(v_var) + covariance_regularizer(t_var))

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

if __name__=='__main__':
    v_mean=torch.randn(10,768)
    v_var=torch.randn(10,768).exp()
    t_mean=torch.randn(10,768)
    t_var=torch.randn(10,768).exp()

    loss=contrastive_loss(F.normalize(v_mean), F.normalize(v_var), F.normalize(t_mean), F.normalize(t_var))
    print(loss)
