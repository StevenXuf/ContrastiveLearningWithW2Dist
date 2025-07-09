import torch
import os
import torch.nn as nn
from datasets import load_dataset
from torchvision import models,datasets
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class CIFAR10Text(datasets.CIFAR10):
    def __init__(self, root, tokenizer_name='bert-base-uncased',train=True,transform=None):
        super().__init__(root, train=train, download=True,transform=transform)
        self.classes = ['airplane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        text = f"A photo of a {self.classes[label]}"
        tokens = self.tokenizer(text, padding='max_length', truncation=True,
                               max_length=32, return_tensors='pt')
        return img, tokens, label

class LoadFlickr30K():
    def __init__(self,tokenizer_name='bert-base-uncased',BATCH_SIZE=64):
        self.n_workers=os.cpu_count()
        self.BATCH_SIZE=BATCH_SIZE
        flickr30k_dataset= load_dataset('nlphuji/flickr30k', cache_dir='./data',split='test')

        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
        ])

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.dataset = flickr30k_dataset.map(
            self.process_examples,
            batched=True,
            batch_size=64,
            #num_proc=self.n_workers,
            remove_columns=['img_id', 'filename', 'sentids']
        )
    def process_examples(self,examples):
        # Process images
        examples['image'] = [self.transform(img.convert('RGB')) for img in examples['image']]

        # Process text (tokenize all 5 captions)
        captions = [c for sublist in examples['caption'] for c in sublist]
        tokenized = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=76,
            return_tensors='pt'
        )

        # Reshape to (batch_size, 5, seq_len)
        batch_size = len(examples['caption'])
        examples['input_ids'] = tokenized['input_ids'].view(batch_size, 5, -1)
        examples['attention_masks'] = tokenized['attention_mask'].view(batch_size, 5, -1)

        return examples

    # 5. Collate Function with Random Caption Selection
    def collate_fn(self,batch):
        images = []
        input_ids = []
        attention_masks = []

        for item in batch:
            images.append(item['image'])
            # Randomly select one caption from five
            idx = torch.randint(0, 5, (1,)).item()
            input_ids.append(item['input_ids'][idx])
            attention_masks.append(item['attention_masks'][idx])
        
        return {
            'images': torch.stack(list(map(torch.tensor,images))),
            'input_ids': torch.stack(list(map(torch.tensor,input_ids))),
            'attention_masks': torch.stack(list(map(torch.tensor,attention_masks)))
        }

    
    def get_loaders(self):
        loaders=[]
        for split in ['train','val','test']:
            if split=='train':
                batch_size=self.BATCH_SIZE
            else:
                batch_size=1000
            loaders.append(
            DataLoader(
            self.dataset.filter(lambda x: x["split"] == split,num_proc=self.n_workers),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.n_workers,
            pin_memory=True)
            )

        return loaders


if __name__=='__main__':
    flickr30k=LoadFlickr30K()
    print(list(map(len,flickr30k.get_loaders())))
