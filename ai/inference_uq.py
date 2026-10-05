import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Force Python to add the root folder to its radar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from ai.dataset import PhaseRetrievalDataset
from ai.checkpoint import load_model

def run_monte_carlo_inference(model_path: str, mc_passes: int = 50):
    # 1. Hardware Optimization: Utilize CUDA if available, fallback to MPS/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Running Monte Carlo Inference on: {device}")

    # 2. Rebuild the trained network from its checkpoint (geometry and diversity planes included).
    # Raises if the checkpoint is missing or does not match; never runs with random weights
    model, cfg, _ = load_model(model_path, device)
    print(f"Loaded trained model weights from {model_path} (K={cfg.K} planes {cfg.diversity_defocus}).")

    # 3. Initialize the dataset to generate a single unseen test sample
    test_dataset = PhaseRetrievalDataset(cfg, num_samples=1, device=device, return_coeffs=True)
    sample = test_dataset[0]
    intensity_input, true_phase = sample[0], sample[1]
    
    # Add the batch dimension [1, K, crop, crop] expected by the network
    intensity_input = intensity_input.unsqueeze(0).to(device)
    true_phase = true_phase.to(device)

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
    
    return intensity_input[0], true_phase.squeeze(), mean_prediction, variance_map, test_dataset.mask.squeeze(), cfg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monte Carlo dropout inference on one fresh synthetic sample")
    parser.add_argument('--checkpoint', type=str, default='saved_models/latest/best.pth')
    parser.add_argument('--mc-passes', type=int, default=50)
    args = parser.parse_args()

    # Unpack the outputs, including the mask and the checkpoint's config
    intensity, truth, mean_pred, uncertainty, mask, cfg = run_monte_carlo_inference(args.checkpoint, mc_passes=args.mc_passes)
    
    # Move tensors to CPU and convert to NumPy for Matplotlib
    intensity = np.concatenate(list(intensity.cpu().numpy()), axis=1)  # diversity planes tiled left to right
    truth = truth.cpu().numpy()
    mean_pred = mean_pred.cpu().numpy()
    uncertainty = uncertainty.cpu().numpy()
    mask = mask.cpu().numpy()

    # Apply the physical lens mask to the AI predictions
    mean_pred = mean_pred * mask
    uncertainty = uncertainty * mask

    # 6. Visualize the Portfolio Deliverable
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    c1 = axes[0].imshow(intensity, cmap='inferno')
    axes[0].set_title("Input: Sensor Intensity")
    axes[0].axis('off')
    
    c2 = axes[1].imshow(truth, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
    axes[1].set_title("Target: True Phase Map")
    fig.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)
    
    c3 = axes[2].imshow(mean_pred, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
    axes[2].set_title("Output: Predicted Phase (Mean)")
    fig.colorbar(c3, ax=axes[2], fraction=0.046, pad=0.04)
    
    c4 = axes[3].imshow(uncertainty, cmap='magma', extent=cfg.extent)
    axes[3].set_title("UQ: Predictive Variance")
    fig.colorbar(c4, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('uq_monte_carlo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Inference UQ visual saved to uq_monte_carlo.png")
