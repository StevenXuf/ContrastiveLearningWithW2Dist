import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Hyperparameters
BATCH_SIZE = 32
EMBED_DIM = 512
LEARNING_RATE = 2e-5
TEMPERATURE = 0.07
REG_WEIGHT = 0.01
EPOCHS = 10

# 1. Load Dataset
dataset = load_dataset('nlphuji/flickr30k', cache_dir='./data')

# 2. Define Transforms
transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225])
])

# 3. Preprocess Function
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def process_examples(examples):
    # Process images
    examples['image'] = [transform(img.convert('RGB')) for img in examples['image']]
    
    # Process text (tokenize all 5 captions)
    captions = [c for sublist in examples['caption'] for c in sublist]
    tokenized = tokenizer(
        captions,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    
    # Reshape to (batch_size, 5, seq_len)
    batch_size = len(examples['caption'])
    examples['input_ids'] = tokenized['input_ids'].view(batch_size, 5, -1)
    examples['attention_mask'] = tokenized['attention_mask'].view(batch_size, 5, -1)
    
    return examples

# 4. Prepare Dataset
dataset = dataset.map(
    process_examples,
    batched=True,
    batch_size=32,
    remove_columns=[ 'sentids', 'img_id', 'filename']
)

# 5. Collate Function with Random Caption Selection
def collate_fn(batch):
    images = []
    input_ids = []
    attention_masks = []
    
    for item in batch:
        images.append(item['image'])
        # Randomly select one caption from five
        idx = torch.randint(0, 5, (1,)).item()
        input_ids.append(item['input_ids'][idx])
        attention_masks.append(item['attention_mask'][idx])
    
    return {
        'images': torch.stack(images),
        'input_ids': torch.stack(input_ids),
        'attention_masks': torch.stack(attention_masks)
    }

# 6. Create DataLoaders
train_loader = DataLoader(
    dataset['train'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    dataset['validation'],
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# 7. Model Architecture (Same as Previous)
class VisionEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.backbone = models.resnet34(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.mean = nn.Linear(512, embed_dim)
        self.logvar = nn.Linear(512, embed_dim)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        variance = self.logvar(features).exp() + 1e-6
        return self.mean(features), variance

class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.mean = nn.Linear(768, embed_dim)
        self.logvar = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, 
                               attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        variance = self.logvar(pooled).exp() + 1e-6
        return self.mean(pooled), variance

# 8. Training Loop (Same as Previous)
vision_encoder = VisionEncoder(EMBED_DIM).cuda()
text_encoder = TextEncoder(EMBED_DIM).cuda()

optimizer = optim.AdamW(
    list(vision_encoder.parameters()) + list(text_encoder.parameters()),
    lr=LEARNING_RATE
)

def evaluate_recall(dataloader, top_k=1):
    vision_encoder.eval()
    text_encoder.eval()
    correct = 0
    total = 0

    # Build caption cache
    caption_cache = {}
    for img_path, captions in dataloader.dataset.captions.items():
        caption_cache[img_path] = [
            text_encoder(
                dataloader.dataset.tokenizer(
                    c, padding='max_length', truncation=True,
                    max_length=64, return_tensors='pt'
                )['input_ids'].squeeze(1).to(device),
                dataloader.dataset.tokenizer(
                    c, padding='max_length', truncation=True,
                    max_length=64, return_tensors='pt'
                )['attention_mask'].squeeze(1).to(device)
            ) for c in captions
        ]

    with torch.no_grad():
        for images, _, img_paths in dataloader:
            images = images.to(device)

            # Get image embeddings
            v_mean, v_var = vision_encoder(images)

            # Compare with all captions
            for i, path in enumerate(img_paths):
                # Get all text embeddings for this image
                text_means = []
                text_vars = []
                for c in caption_cache[path]:
                    t_mean, t_var = text_encoder(c[0], c[1])
                    text_means.append(t_mean)
                    text_vars.append(t_var)

                # Calculate similarities
                sims = []
                for t_m, t_v in zip(text_means, text_vars):
                    sim = -wasserstein2_distance(
                        v_mean[i].unsqueeze(0), v_var[i].unsqueeze(0),
                        t_m, t_v
                    )
                    sims.append(sim.item())

                # Check if any correct caption in top-k
                sorted_indices = np.argsort(sims)[::-1]
                if 0 in sorted_indices[:top_k]:  # First caption is original
                    correct += 1
                total += 1

    return correct / total

# 4. Training Loop Adjustments
# [Same training loop as previous code but with Flickr30K data]

# 5. Run Evaluation
print("Validation Metrics:")
print(f"Top-1 Recall: {evaluate_recall(val_loader):.4f}")
print(f"Top-5 Recall: {evaluate_recall(val_loader, top_k=5):.4f}")
