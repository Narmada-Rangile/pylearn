import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir='logs', model_name='model'):
        self.log_dir = log_dir
        self.model_name = model_name
        self.history = {'epoch': []}
        self.start_time = None
        self.epoch_start_time = None
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
    def on_train_begin(self, n_epochs):
        self.n_epochs = n_epochs
        self.start_time = time.time()
        print(f"Starting training for {self.n_epochs} epochs...")
        print(f"Logs will be saved to: {self.log_dir}")
        print("-" * 60)

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        # Ensure epoch entry exists
        if len(self.history['epoch']) < epoch + 1:
            self.history['epoch'].append(epoch)

    def log_metric(self, name, value, epoch=None):
        if epoch is None:
            epoch = self.current_epoch
        
        if name not in self.history:
            self.history[name] = []
            
        # Fill missing values if any (simple forward fill or None)
        while len(self.history[name]) < epoch:
             self.history[name].append(None)
             
        if len(self.history[name]) == epoch:
            self.history[name].append(value)
        else:
            # If we are updating an existing epoch
            self.history[name][epoch] = value

    def on_batch_end(self, batch, n_batches, metrics=None):
        # Simple progress bar
        # [=====     ] 50% - Loss: 0.5
        
        percent = (batch + 1) / n_batches
        bar_len = 30
        filled_len = int(bar_len * percent)
        bar = '=' * filled_len + ' ' * (bar_len - filled_len)
        
        metric_str = ""
        if metrics:
            metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            
        print(f"\rEpoch {self.current_epoch+1}/{self.n_epochs} [{bar}] {int(percent*100)}% - {metric_str}", end='')

    def on_epoch_end(self, epoch, metrics=None):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        avg_time_per_epoch = total_time / (epoch + 1)
        remaining_epochs = self.n_epochs - (epoch + 1)
        eta_seconds = avg_time_per_epoch * remaining_epochs
        
        # Format ETA
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        # Update history with provided metrics
        if metrics:
            for k, v in metrics.items():
                self.log_metric(k, v, epoch)
        
        # Newline after batch progress bar
        print() 
        
        metric_str = f"Time: {epoch_time:.2f}s - ETA: {eta_str}"
        if metrics:
            metric_str += " - " + " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            
        print(f"Epoch {epoch+1}/{self.n_epochs} Finished. {metric_str}")
        print("-" * 60)
        
    def save_checkpoint(self, model, filename=None, is_best=False):
        if filename is None:
             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
             filename = f"{self.model_name}_ep{self.current_epoch+1}_{timestamp}.pkl"
        
        path = os.path.join(self.log_dir, filename)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Checkpoint saved: {path}")
            
            if is_best:
                best_path = os.path.join(self.log_dir, f"{self.model_name}_best.pkl")
                with open(best_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Best model saved: {best_path}")
                
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def plot_metrics(self, save_path=None):
        # Convert history to DataFrame for easier plotting
        # Ensure all lists have same length
        max_len = len(self.history['epoch'])
        cleaned_history = {}
        for k, v in self.history.items():
            if len(v) < max_len:
                 v = v + [None] * (max_len - len(v))
            cleaned_history[k] = v[:max_len]
            
        df = pd.DataFrame(cleaned_history)
        
        # Plot Loss if available
        loss_cols = [c for c in df.columns if 'loss' in c.lower()]
        acc_cols = [c for c in df.columns if 'acc' in c.lower() or 'score' in c.lower()]
        
        n_plots = 0
        if loss_cols: n_plots += 1
        if acc_cols: n_plots += 1
        
        if n_plots == 0:
            print("No metrics to plot.")
            return

        plt.figure(figsize=(10, 5 * n_plots))
        
        plot_idx = 1
        if loss_cols:
            plt.subplot(n_plots, 1, plot_idx)
            for c in loss_cols:
                plt.plot(df['epoch'], df[c], label=c)
            plt.title("Loss vs. Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plot_idx += 1
            
        if acc_cols:
            plt.subplot(n_plots, 1, plot_idx)
            for c in acc_cols:
                plt.plot(df['epoch'], df[c], label=c)
            plt.title("Metrics vs. Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training plot saved to {save_path}")
        else:
            plt.show()
