import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class CustomEarlyStopping(Callback):
    """
    Custom early stopping callback that monitors train accuracy instead of validation loss.
    Based on the user's early stopping implementation.
    """
    def __init__(self, patience=7, min_delta=0, monitor='train_acc'):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_metric = float('-inf')  # For accuracy, higher is better
        self.counter = 0
        self.early_stop = False

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the current train accuracy
        if hasattr(pl_module, 'train_metric'):
            current_metric = pl_module.train_metric.compute().item()
        else:
            # Fallback to logged metrics
            current_metric = trainer.logged_metrics.get(self.monitor, float('-inf'))
        
        # Check if we should stop early
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                trainer.should_stop = True
                print(f"Early stopping triggered after {self.patience} epochs without improvement in {self.monitor}")

    def on_train_start(self, trainer, pl_module):
        self.best_metric = float('-inf')
        self.counter = 0
        self.early_stop = False
