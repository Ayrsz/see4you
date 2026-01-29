import torch
from torch import nn
from tqdm import tqdm

# TODO: salvar modelos em /checkpoints

def train_one_epoch(model, dataloader, loss_function, device):
    model.train()
    for images, captions, captions_lengths in dataloader:
        images = images.to(device)
        captions = captions.to(device)
        captions_lengths = captions_lengths.to(device)

        logits = model(images, captions, captions_lengths)
        loss = loss_function(logits, captions, reduction="mean")
       
        total_loss += loss.item() * captions.size(0)

def eval_one_epoch(model, dataloader, loss_function, device):
    model.eval()
    with torch.no_grad():
        for images, captions, captions_lengths in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            captions_lengths = captions_lengths.to(device)

            logits = model(images, captions, captions_lengths)
            loss = loss_function(logits, captions, reduction="mean")
        
            total_loss += loss.item() * captions.size(0)

def fit():
    loss_function = nn.CrossEntropyLoss()
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

# Integrando tudo em uma classe
    
class ImageCaptionTrainer:
    def __init__(self, model, optimizer, loss_function, device=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = loss_function
        
        self.model.to(self.device)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="  Training Batches", leave=False)
        
        for images, captions, captions_lengths in pbar:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # Flattening for CrossEntropyLoss:
            # We want to treat every word in every sequence as an individual sample.
            self.optimizer.zero_grad()

            # Logits shape: [Batch, Seq_Len + 1, Vocab]
            logits = self.model(images, captions, captions_lengths)

            logits_flattened = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            captions_flattened = captions.contiguous().view(-1)

            loss = self.loss_function(logits_flattened, captions_flattened)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())
            
        return total_loss / len(dataloader.dataset)

    def eval_one_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="  Evaluating Batches", leave=False)
        
        with torch.no_grad():
            for images, captions, captions_lengths in pbar:
                images = images.to(self.device)
                captions = captions.to(self.device)

                logits = self.model(images, captions, captions_lengths)
                loss = self.loss_function(logits, captions)
            
                total_loss += loss.item() * images.size(0)
                
        return total_loss / len(dataloader.dataset)

    def fit(self, train_loader, val_loader, epochs, patience, epsilon=1e-4):
            """
            Training loop with Early Stopping logic.
            """
            print(f"Training is beginning with device: {self.device}")
            
            min_val_loss = float('inf')
            epochs_no_improve = 0
            
            # Use tqdm for a visual progress bar over epochs
            epoch_pbar = tqdm(range(epochs), desc="Epochs")
            
            for epoch in epoch_pbar:
                # Training phase
                train_loss = self.train_one_epoch(train_loader)
                
                # Validation phase
                val_loss = self.eval_one_epoch(val_loader)
                
                # Check if the improvement is greater than epsilon
                if val_loss < min_val_loss - epsilon:
                    min_val_loss = val_loss
                    epochs_no_improve = 0
                    # torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1
                
                # Update progress bar description
                epoch_pbar.set_postfix({
                    'Train Loss': f"{train_loss:.4f}",
                    'Val Loss': f"{val_loss:.4f}",
                    'Patience': f"{epochs_no_improve}/{patience}"
                })
                
                # Check if patience limit is reached
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break