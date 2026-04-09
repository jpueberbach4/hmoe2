import os
import json
import torch
from typing import Dict, List, Iterable
from hmoe2.nodes import HmoeNode
from hmoe2.loss import HmoeLossEngine, HmoeLossResult
from hmoe2.tensor import HmoeTensor, HmoeInput

class HmoeTrainer:
    """
    Abstracts the PyTorch training loop, metric tracking, and checkpointing 
    for the Hierarchical Mixture of Experts architecture.
    """
    def __init__(
        self, 
        model: HmoeNode, 
        loss_engine: HmoeLossEngine, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        scheduler_patience: int = 25 # THE FIX: Lowered default to fire before early stopping
    ):
        self.model = model.to(device)
        self.loss_engine = loss_engine.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=scheduler_patience ,
            threshold=1e-4,    # Respect improvements as small as 0.0001
            min_lr=1e-5        # NEVER drop the LR below this hard floor
        )

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "task_metrics": []
        }

    def _process_batch(self, clean_master: HmoeTensor, is_training: bool) -> HmoeLossResult:
        """Internal helper to guarantee identical data handling for Train and Val."""
        
        clean_master = clean_master.to(self.device)
        
        input_payload = HmoeInput(tensor=clean_master)
        predictions = self.model(input_payload)
        
        loss_result = self.loss_engine(predictions, clean_master)
        
        if is_training:
            self.optimizer.zero_grad()
            loss_result.total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
        return loss_result

    def fit(
        self, 
        train_dataloader: Iterable[HmoeTensor], 
        val_dataloader: Iterable[HmoeTensor], 
        epochs: int, 
        patience: int = 15 # THE FIX: Bumped default so the scheduler can fire multiple times
    ):
        print(f"[*] Starting Training Loop | Device: {self.device} | Epochs: {epochs}")
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # --- TRAIN EPOCH ---
            self.model.train()
            train_epoch_loss = 0.0
            
            for batch in train_dataloader:
                result = self._process_batch(batch, is_training=True)
                train_epoch_loss += result.total_loss.item()
                
            avg_train_loss = train_epoch_loss / len(train_dataloader)

            # --- VALIDATION EPOCH ---
            self.model.eval()
            val_epoch_loss = 0.0
            epoch_task_metrics = {}
            
            with torch.no_grad():
                for batch in val_dataloader:
                    result = self._process_batch(batch, is_training=False)
                    val_epoch_loss += result.total_loss.item()
                    
                    for task, val in result.task_metrics.items():
                        epoch_task_metrics[task] = epoch_task_metrics.get(task, 0.0) + val
            
            avg_val_loss = val_epoch_loss / len(val_dataloader)
            
            for task in epoch_task_metrics:
                epoch_task_metrics[task] /= len(val_dataloader)

            # --- METRICS & CHECKPOINTING ---
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["task_metrics"].append(epoch_task_metrics)

            print(f"Epoch {epoch:03d}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Metrics: {epoch_task_metrics}")

            self.save_checkpoint(filename="latest_checkpoint.pt")
            self.save_metrics()

            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]['lr']
                
                self.scheduler.step(avg_val_loss)
                
                new_lr = self.optimizer.param_groups[0]['lr']
                
                if new_lr < old_lr:
                    print(f"\n[!!!] PLATEAU REACHED: Validation loss stalled. Slashing Learning Rate to {new_lr:.2e}\n")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_checkpoint(filename="best_checkpoint.pt")
                print(f"  -> [New Best Model Saved]")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[*] Early stopping triggered at epoch {epoch}. Reverting to best model.")
                    break

    def save_checkpoint(self, filename: str):
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)

    def save_metrics(self):
        filepath = os.path.join(self.checkpoint_dir, "metrics_history.json")
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
            
    def load_checkpoint(self, filename: str = "best_checkpoint.pt"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")
            
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[*] Successfully loaded checkpoint: {filename}")