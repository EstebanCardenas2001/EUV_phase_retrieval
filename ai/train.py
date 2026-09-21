import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

# Force Python to add the root folder to its radar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.dataset import PhaseRetrievalDataset
from ai.unet import UNet

def train_model():
    # 1. Hardware Optimization & Hyperparameters
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Initializing overnight training on: {device}")
    
    batch_size = 16
    initial_lr = 1e-3
    epochs = 200              # Increased for overnight convergence
    samples_per_epoch = 2000  
    
    # Create directories for saving outputs
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('training_progress', exist_ok=True)
    
    # 2. Setup the Static Validation Anchor (For visual tracking)
    print("Generating static validation anchor...")
    val_dataset = PhaseRetrievalDataset(num_samples=1, device=torch.device('cpu'))
    static_intensity, static_truth = val_dataset[0]
    static_intensity_gpu = static_intensity.unsqueeze(0).to(device)
    static_mask = val_dataset.mask.squeeze().cpu().numpy()
    
    # Save the input/truth arrays for the plotting function later
    np_intensity = static_intensity.squeeze().cpu().numpy()
    np_truth = static_truth.squeeze().cpu().numpy()

    # 3. Initialize Dataset, DataLoader, Model, and Optimizer
    train_dataset = PhaseRetrievalDataset(num_samples=samples_per_epoch, device=torch.device('cpu'))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,        
        pin_memory=False,     
        prefetch_factor=4     
    )
    
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Initialize the Learning Rate Scheduler
    # If the loss doesn't drop for 4 epochs (patience), reduce the LR by half (factor=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    
    # 4. The Overnight Training Loop
    print(f"Starting U-Net Training for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0.0
        
        for batch_idx, (intensities, true_phases) in enumerate(train_loader):
            intensities = intensities.to(device, non_blocking=True)
            true_phases = true_phases.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            predicted_phases = model(intensities)
            loss = criterion(predicted_phases, true_phases)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
                
        avg_epoch_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"==> Epoch {epoch+1:03d}/{epochs} | Avg Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.2e}")
        
        # Step the scheduler based on the epoch's performance
        scheduler.step(avg_epoch_loss)
        
        # Save the absolute best model strictly based on mathematical loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'saved_models/unet_phase_retrieval_best.pth')
            
        # 5. Visual Epoch Tracking (Every 10 Epochs)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Predict and apply physical mask to erase square artifacts
                pred = model(static_intensity_gpu).squeeze().cpu().numpy()
                pred = pred * static_mask
                
            # Generate the comparative time-lapse plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            c1 = axes[0].imshow(np_intensity, cmap='inferno')
            axes[0].set_title("Input: Sensor Intensity")
            axes[0].axis('off')
            
            c2 = axes[1].imshow(np_truth, cmap='RdBu', extent=[-0.005, 0.005, -0.005, 0.005], vmin=-2.0, vmax=2.0)
            axes[1].set_title("Target: True Phase Map")
            fig.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)
            
            c3 = axes[2].imshow(pred, cmap='RdBu', extent=[-0.005, 0.005, -0.005, 0.005], vmin=-2.0, vmax=2.0)
            axes[2].set_title(f"Network Output (Epoch {epoch+1})")
            fig.colorbar(c3, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(f'training_progress/epoch_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig) # Close the figure to free up RAM over the night
            
    # Save the final checkpoint when the loop finishes
    torch.save(model.state_dict(), 'saved_models/unet_phase_retrieval_final.pth')
    print(f"\nTraining complete. Best model saved with loss: {best_loss:.6f}")

if __name__ == "__main__":
    train_model()