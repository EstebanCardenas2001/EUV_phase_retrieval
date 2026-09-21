import torch
import matplotlib.pyplot as plt
import sys
import os

# Force Python to add the root folder to its radar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from ai.dataset import PhaseRetrievalDataset
from ai.unet import UNet

def run_monte_carlo_inference(model_path: str, mc_passes: int = 50):
    # 1. Hardware Optimization: Utilize Apple Silicon MPS
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running Monte Carlo Inference on: {device}")

    # 2. Initialize the dataset to generate a single unseen test sample
    test_dataset = PhaseRetrievalDataset(num_samples=1, device=device)
    intensity_input, true_phase = test_dataset[0]
    
    # Add the batch dimension [1, 1, 256, 256] expected by the network
    intensity_input = intensity_input.unsqueeze(0).to(device)
    true_phase = true_phase.to(device)

    # 3. Load the trained network
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded trained model weights.")
    else:
        print(f"Warning: {model_path} not found. Running with untrained random weights.")

    # 4. Setup Monte Carlo Dropout
    model.eval()               
    model.enable_mc_dropout()  

    print(f"Executing {mc_passes} forward passes for Uncertainty Quantification...")
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(mc_passes):
            pred = model(intensity_input)
            predictions.append(pred)
            
    # 5. Statistical Aggregation
    predictions_tensor = torch.stack(predictions)
    mean_prediction = torch.mean(predictions_tensor, dim=0).squeeze()
    variance_map = torch.var(predictions_tensor, dim=0).squeeze()
    
    # FIX: We now explicitly return the physical mask alongside the other tensors
    return intensity_input.squeeze(), true_phase.squeeze(), mean_prediction, variance_map, test_dataset.mask.squeeze()


if __name__ == "__main__":
    model_path = 'saved_models/unet_phase_retrieval.pth'
    
    # FIX: Unpack the 5 variables, including the mask
    intensity, truth, mean_pred, uncertainty, mask = run_monte_carlo_inference(model_path, mc_passes=50)
    
    # Move tensors to CPU and convert to NumPy for Matplotlib
    intensity = intensity.cpu().numpy()
    truth = truth.cpu().numpy()
    mean_pred = mean_pred.cpu().numpy()
    uncertainty = uncertainty.cpu().numpy()
    mask = mask.cpu().numpy()

    # Apply the physical lens mask to the AI predictions
    # This erases the U-Net's "square boundary" artifact, enforcing physical reality
    mean_pred = mean_pred * mask
    uncertainty = uncertainty * mask

    # 6. Visualize the Portfolio Deliverable
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    c1 = axes[0].imshow(intensity, cmap='inferno')
    axes[0].set_title("Input: Sensor Intensity")
    axes[0].axis('off')
    
    c2 = axes[1].imshow(truth, cmap='RdBu', extent=[-0.005, 0.005, -0.005, 0.005])
    axes[1].set_title("Target: True Phase Map")
    fig.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)
    
    c3 = axes[2].imshow(mean_pred, cmap='RdBu', extent=[-0.005, 0.005, -0.005, 0.005])
    axes[2].set_title("Output: Predicted Phase (Mean)")
    fig.colorbar(c3, ax=axes[2], fraction=0.046, pad=0.04)
    
    c4 = axes[3].imshow(uncertainty, cmap='magma', extent=[-0.005, 0.005, -0.005, 0.005])
    axes[3].set_title("UQ: Predictive Variance")
    fig.colorbar(c4, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()