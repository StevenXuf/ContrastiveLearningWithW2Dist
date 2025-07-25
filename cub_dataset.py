from datasets import DatasetDict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset

class LoadCUB():
    def __init__(self,tokenizer_name='bert-base-uncased',batch_size=64,image_size=224,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size=batch_size
        self.image_size= image_size# Standard size for CNNs

        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure RGB format
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),  # Converts to [0,1] range
            transforms.Normalize(mean=mean,std=std)
        ])

    def transform_batch(self,batch):
        """Process a batch of examples into model inputs."""
        # Tokenize all texts in the batch
        text_inputs = self.tokenizer(
            batch["text"],
            padding="max_length",  # Pad to longest in batch
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        # Process all images
        images = [self.image_transform(img) for img in batch["image"]]
        
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": torch.stack(images)  # Stack into batch dimension
        }

    def collate_fn(self,batch):
        """Custom collate to handle dataset structure."""
        # Batch is a list of dicts; convert to dict of lists
        return {
            "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch]),
            "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in batch]),
            "pixel_values": torch.stack([torch.tensor(item["pixel_values"]) for item in batch])
        }

    def get_loaders(self):
        dataset=load_dataset('cassiekang/cub200_dataset')
        # Apply transformations to datasets (lazily)
        transformed_dataset = dataset.map(
            self.transform_batch,
            batched=True,  # Process in batches for efficiency
            batch_size=self.batch_size,  # Adjust based on memory
            remove_columns=["image", "text"]  # Remove original columns
        )

        # Create DataLoaders
        train_loader = DataLoader(
            transformed_dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        test_loader = DataLoader(
            transformed_dataset["test"],
            batch_size=32,
            collate_fn=self.collate_fn
        )
       
        return train_loader,test_loader

if __name__=='__main__':
    cub=LoadCUB()
    train_loader,test_loader=cub.get_loaders()
