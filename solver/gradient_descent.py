import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import our Week 1 physics engine
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.simulator import OpticalSystem
from physics.grid import create_spatial_grid
from physics.zernike import get_polar_coordinates, zernike_polynomial

def total_variation_loss(img: torch.Tensor):
    """
    Calculates the Total Variation (TV) of a 2D tensor to penalize high-frequency noise.
    """
    # Difference between adjacent rows (vertical edges)
    tv_h = torch.mean(torch.abs(img[1:, :] - img[:-1, :]))
    # Difference between adjacent columns (horizontal edges)
    tv_w = torch.mean(torch.abs(img[:, 1:] - img[:, :-1]))
    return tv_h + tv_w

def run_inverse_solver():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # 1. Initialize the Physics Engine
    simulator = OpticalSystem(N=256, L=0.01, pupil_radius=0.004, device=device)

    # 2. Generate the "Ground Truth" (The hidden reality we want to discover)
    X, Y = create_spatial_grid(256, 0.01, device=device)
    rho, theta = get_polar_coordinates(X, Y, 0.004)
    
    # Let's hide a complex combination of aberrations
    true_phase = (zernike_polynomial(rho, theta, simulator.mask, 'astigmatism_vertical') * 2.0 + 
                  zernike_polynomial(rho, theta, simulator.mask, 'coma_horizontal') * -1.5)
    
    # Run it through the simulator to get the sensor measurement. We detach it from 
    # the computation graph because it is our fixed target, not a variable.
    target_intensity = simulator(true_phase, noise_std=0.0).detach()

    # 3. Initialize the Optimizer's Guess
    # We start with a completely flat, un-aberrated wavefront (all zeros).
    # requires_grad=True is the magic that tells PyTorch to calculate derivatives for this tensor.
    predicted_phase = torch.zeros((256, 256), requires_grad=True, device=device)

    # We use the Adam optimizer. A learning rate of 0.1 is aggressive but works well for phase retrieval.
    optimizer = optim.Adam([predicted_phase], lr=0.1)

    # 4. The Optimization Loop
    iterations = 300
    loss_history = []

    print("Starting optimization...")
    for i in range(iterations):
        optimizer.zero_grad() # Clear old gradients

# Forward pass
        simulated_intensity = simulator(predicted_phase)

        # 1. Data Loss (MSE on Amplitude)
        loss_data = F.mse_loss(torch.sqrt(simulated_intensity), torch.sqrt(target_intensity))
        
        # 2. Regularization Loss (Physical Smoothness)
        loss_tv = total_variation_loss(predicted_phase)
        
        # 3. Total Loss
        # lambda_tv is the weight. Too high, and the phase becomes a flat plane. 
        # Too low, and the noise remains. 0.05 is a solid baseline for phase retrieval.
        lambda_tv = 0.05 
        loss = loss_data + lambda_tv * loss_tv

        # Backward pass
        loss.backward()

        # Update the phase tensor
        optimizer.step()

        # Apply the mask to the predicted phase to keep it clean outside the lens
        with torch.no_grad():
            predicted_phase.data *= simulator.mask

        loss_history.append(loss.item())
        if (i + 1) % 50 == 0:
            print(f"Iteration {i+1}/{iterations} | Loss: {loss.item():.6f}")

    return true_phase, target_intensity, predicted_phase.detach(), loss_history

if __name__ == "__main__":
    true_phase, target_intensity, recovered_phase, loss_history = run_inverse_solver()

    # 5. Visualize the Results
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    c1 = axes[0].imshow(true_phase.cpu().numpy(), cmap='RdBu', extent=[-0.005, 0.005, -0.005, 0.005])
    axes[0].set_title("Ground Truth Phase (Hidden)")
    fig.colorbar(c1, ax=axes[0])

    c2 = axes[1].imshow(target_intensity.cpu().numpy()**0.5, cmap='inferno')
    axes[1].set_title("Sensor Target (Measured)")
    axes[1].axis('off')

    c3 = axes[2].imshow(recovered_phase.cpu().numpy(), cmap='RdBu', extent=[-0.005, 0.005, -0.005, 0.005])
    axes[2].set_title("Recovered Phase (Solver Output)")
    fig.colorbar(c3, ax=axes[2])

    axes[3].plot(loss_history, color='blue', linewidth=2)
    axes[3].set_title("Optimization Loss Curve")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("MSE (Amplitude)")
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()