import torch
import torch.nn as nn

class PricePredictor(nn.Module):
    VERSION = 3
    
    def __init__(self, input_size, is_stablecoin=False):
        super().__init__()
        
        self.is_stablecoin = is_stablecoin
        self.input_size = input_size
        self.build_network()
        
    def build_network(self):
        """Build the neural network architecture"""
        if self.is_stablecoin:
            # Simpler architecture for stablecoins
            layers = [
                nn.BatchNorm1d(self.input_size),
                nn.Linear(self.input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ]
        else:
            # Enhanced architecture for regular cryptocurrencies
            layers = [
                nn.BatchNorm1d(self.input_size),
                nn.Linear(self.input_size, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.BatchNorm1d(256),
                
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.BatchNorm1d(128),
                
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                nn.BatchNorm1d(64),
                
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(32),
                
                nn.Linear(32, 1),
                nn.Sigmoid()
            ]
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
        
    def load_state_dict(self, state_dict, strict=True):
        """Custom state dict loading to handle different key formats"""
        # Create a new state dict with corrected keys
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Handle the case where the loaded model has 'network.network.' prefix
            if key.startswith('network.network.'):
                new_key = key.replace('network.network.', 'network.')
                new_state_dict[new_key] = value
            # Handle the case where the loaded model has just 'network.' prefix
            elif key.startswith('network.'):
                new_state_dict[key] = value
            else:
                new_state_dict[f'network.{key}'] = value
                
        # Call the parent class's load_state_dict with the corrected state dict
        return super().load_state_dict(new_state_dict, strict=False)