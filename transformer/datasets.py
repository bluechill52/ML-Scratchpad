import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import SimpleTokenizer


class DatasetV1(Dataset):
    def __init__(self, text, tokenizer, context_size, stride):
        super(DatasetV1, self).__init__()
        
        token_ids = tokenizer.encode(text)
        
        self.input_ids = []
        self.target_ids = []
        
        # Create input ids and target ids using sliding window of context size and stride
        start_idx = 0
        for start_idx in range(0, len(token_ids) - context_size, stride):
            self.input_ids.append(torch.tensor(token_ids[start_idx:start_idx+context_size]))
            self.target_ids.append(torch.tensor(token_ids[start_idx + 1:start_idx+context_size + 1]))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    def print(self):
        for input_id, target_id in zip(self.input_ids, self.target_ids):
            print(input_id, end=' - ')
            print(target_id)


def createDataloaderV1(text, tokenizer, batch_size=4, context_size=256, stride=128, shuffle=True, drop_last=True, num_workers=8):
    # Create dataset
    dataset = DatasetV1(text, tokenizer, context_size, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':
    fname = 'the-verdict.txt'
    context_size = 5
    stride = 1
    batch_size = 1

    tokenizer = SimpleTokenizer(fname)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), 'r', encoding='utf-8') as f:
        raw_text = f.read()

    dataset = DatasetV1(raw_text, tokenizer, context_size, stride)
    dataloader = createDataloaderV1(raw_text, tokenizer, batch_size, context_size, stride)
    dataset.print()