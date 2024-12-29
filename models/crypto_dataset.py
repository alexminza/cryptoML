import torch
from torch.utils.data import Dataset

class CryptoDataset(Dataset):
    """Dataset class for cryptocurrency data"""
    
    def __init__(self, features, labels):
        """
        Initialize the dataset.
        
        Args:
            features: numpy array of feature values
            labels: numpy array of target labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.features)
        
    def __getitem__(self, idx):
        """Return a single sample and label pair"""
        return self.features[idx], self.labels[idx]