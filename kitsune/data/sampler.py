import math
import numpy as np
import torch
from torch.utils.data import Sampler

class TextBucketSampler(Sampler):
    """
    Groups data into batches of similar lengths based on phoneme sequence length.
    Dramatically reduces padding overhead and eliminates GPU utilization oscillation.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # We group by the length of the phoneme sequence (highly correlated with audio length)
        # This is instantaneous because phonemized_texts is already in memory.
        self.lengths = [len(seq) for seq in dataset.phonemized_texts]
        
        # Sort indices by length
        self.sorted_indices = np.argsort(self.lengths).tolist()
        
        # Group into batches
        self.batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i:i + self.batch_size]
            self.batches.append(batch)
            
    def __iter__(self):
        batches = self.batches.copy()
        if self.shuffle:
            # Shuffle the order of the batches (so the model doesn't always see short -> long)
            np.random.shuffle(batches)
            
            # Slightly shuffle within each batch to prevent exact same pairs every epoch
            for batch in batches:
                np.random.shuffle(batch)
                
        # Yield lists of indices directly (batch-by-batch) to act as a BatchSampler
        return iter(batches)
        
    def __len__(self):
        return len(self.batches)
