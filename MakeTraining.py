import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import os
import pickle
from torchtext.data.utils import get_tokenizer


class TextDataset(Dataset):
    def __init__(self, text_paths, seq_length, tokenizer):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.text_encoded = []

        # Read text from multiple files and aggregate into a single list
        for text_path in text_paths:
            with open(text_path, 'r', encoding='utf-8') as file:
                text = file.read()
            self.text_encoded.extend(tokenizer(text))

        self.vocab = sorted(set(self.text_encoded))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def __len__(self):
        return len(self.text_encoded) - self.seq_length

    def __getitem__(self, idx):
        inputs = [self.word_to_idx[word] for word in self.text_encoded[idx:idx+self.seq_length]]
        targets = [self.word_to_idx[word] for word in self.text_encoded[idx+1:idx+self.seq_length+1]]
        return torch.tensor(inputs), torch.tensor(targets)

def collate_batch(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(inputs), torch.stack(targets)

# Define parameters
seq_length = 5
cwd = os.getcwd()
text_directory = os.path.join(cwd, 'extraTrainingTexts')
output_file = 'training_data.pkl'

# Get paths of all text files in the directory
text_paths = [os.path.join(text_directory, filename) for filename in os.listdir(text_directory) if filename.endswith('.txt')]

print(text_paths)

# Create dataset and dataloader
tokenizer = get_tokenizer('basic_english')
dataset = TextDataset(text_paths, seq_length, tokenizer)

data_to_save = {
    'text_paths': text_paths,
    'seq_length': seq_length,
    'vocab': dataset.vocab,
    'word_to_idx': dataset.word_to_idx,
    'idx_to_word': dataset.idx_to_word,
}

with open(output_file, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"Training data saved to {output_file}")