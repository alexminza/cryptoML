import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import logging

class ModelTrainingService:
    """Service class responsible for training individual models"""
    
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def train_model(
        self,
        model: nn.Module,
        train_data,
        val_data,
        epochs: int,
        batch_size: int,
        min_epochs: int,
        patience: int,
        is_stablecoin: bool = False
    ) -> Tuple[nn.Module, dict]:
        """
        Train a single model with the given parameters
        
        Returns:
            Tuple of (trained model, training metrics)
        """
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        # Calculate class weights for imbalanced dataset
        pos_weight = (train_data.labels == 0).sum() / (train_data.labels == 1).sum()
        class_weights = torch.FloatTensor([1.0, pos_weight])
        
        # Initialize training components
        criterion = nn.BCELoss(reduction='none')
        optimizer = self._setup_optimizer(model, is_stablecoin)
        scheduler = self._setup_scheduler(optimizer, is_stablecoin)
        
        # Training loop
        best_val_loss = float('inf')
        best_val_accuracy = 0
        patience_counter = 0
        metrics_history = []
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion, class_weights
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion
            )
            
            # Learning rate scheduling
            if not is_stablecoin:
                scheduler.step()
            else:
                scheduler.step(val_loss)
                
            # Record metrics
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self._log_progress(epoch, epochs, train_loss, val_loss, train_acc, val_acc)
                
            # Early stopping check
            if epoch >= min_epochs:
                if val_loss < best_val_loss or val_acc > best_val_accuracy:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    if val_acc > best_val_accuracy:
                        best_val_accuracy = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f'Early stopping triggered at epoch {epoch+1}')
                        break
                        
        return model, metrics_history
        
    def _setup_optimizer(self, model: nn.Module, is_stablecoin: bool) -> torch.optim.Optimizer:
        """Setup optimizer based on coin type"""
        if is_stablecoin:
            return torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            return torch.optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            
    def _setup_scheduler(self, optimizer, is_stablecoin: bool):
        """Setup learning rate scheduler"""
        if is_stablecoin:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        else:
            def lr_lambda(epoch):
                warmup_epochs = 10
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                return 0.5 ** (epoch / 50)
                
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
    def _train_epoch(self, model, train_loader, optimizer, criterion, class_weights):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Apply class weights to loss
            loss = criterion(outputs, batch_y.unsqueeze(1))
            weights = class_weights[batch_y.long()]
            loss = (loss * weights.unsqueeze(1)).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            predicted = (outputs.data > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted.squeeze() == batch_y).sum().item()
            
        return total_loss / len(train_loader), 100 * correct / total
        
    def _validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1)).mean()
                total_loss += loss.item()
                
                predicted = (outputs.data > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted.squeeze() == batch_y).sum().item()
                
        return total_loss / len(val_loader), 100 * correct / total
        
    def _log_progress(self, epoch, epochs, train_loss, val_loss, train_acc, val_acc):
        """Log training progress"""
        self.logger.info(
            f'Epoch [{epoch+1}/{epochs}], '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%'
        )