import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

class TextDataset(Dataset):
    def __init__(self, text_paths=None, seq_length=None, vocab=None, word_to_idx=None, idx_to_word=None, tokenizer=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer or get_tokenizer('basic_english')
        self.text_encoded = []

        if text_paths:
            # Read text from multiple files and aggregate into a single list
            for text_path in text_paths:
                with open(text_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                self.text_encoded.extend(self.tokenizer(text))

            self.vocab = sorted(set(self.text_encoded))
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        else:
            # Use provided vocab, word_to_idx, and idx_to_word
            self.vocab = vocab
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word

    def __len__(self):
        return max(len(self.text_encoded) - self.seq_length, 0)

    def __getitem__(self, idx):
        if len(self.text_encoded) <= self.seq_length:
            # If the text is shorter than the sequence length, return an empty sample
            return torch.tensor([]), torch.tensor([])
        else:
            inputs = [self.word_to_idx.get(word, 0) for word in self.text_encoded[idx:idx+self.seq_length]]
            targets = [self.word_to_idx.get(word, 0) for word in self.text_encoded[idx+1:idx+self.seq_length+1]]
            return torch.tensor(inputs), torch.tensor(targets)