import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.config import OpticsConfig

class DifferentiableZernikeGenerator(nn.Module):
    """
    Differentiable GPU-accelerated Zernike Phase Generator.
    Synthesizes a 2D phase map from a batch of Zernike coefficients:
        Phase(x, y) = Sum_j [ c_j * Z_j(x, y) ]
    """
    def __init__(self, basis_tensor: torch.Tensor):
        super().__init__()
        # Register basis tensor: [M, H, W]
        self.register_buffer('basis', basis_tensor, persistent=False)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeffs: Tensor of shape [Batch, M]
        Returns:
            phase_map: Tensor of shape [Batch, 1, H, W]
        """
        # Batched tensordot / einsum: [B, M] x [M, H, W] -> [B, H, W]
        phase = torch.einsum('bm,mhw->bhw', coeffs, self.basis)
        return phase.unsqueeze(1)

class ResidualBlock(nn.Module):
    """
    ResNet-style Convolutional Block with identity/projection shortcut.
    Allows high-frequency spatial features and faint diffraction fringes
    to bypass destructive pooling layers via identity mapping.
    """
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out

# Backward-compatibility alias
DoubleConv = ResidualBlock

class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018).
    Dynamically filters skip connection feature maps (x) using the coarser gating
    signal (g) from the decoder. Forces the network to spatially attend to the faint
    outer edges of the diffraction tensor where high-frequency physical data lives,
    rather than being blinded by the central saturated intensity peak.
    """
    def __init__(self, f_g: int, f_l: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionResUNet(nn.Module):
    def __init__(
        self,
        cfg: OpticsConfig,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = None,
        mode: str = 'hybrid',
        negative_slope: float = 0.1
    ):
        """
        Attention Res-UNet for EUV Phase Retrieval:
        1. Residual Convolutional Encoder: Preserves high-frequency spatial features
           via identity shortcuts, preventing data loss across pooling layers.
        2. Attention-Gated Skip Connections: Suppresses the blinding central DC peak
           and forces spatial weighting onto faint outer diffraction rings.
        3. Global Average Pooling -> Linear Modal Head predicting low-order Zernike basis (Noll 4-22).
        4. Differentiable Zernike Generator synthesizing 256x256 modal baseline.
        5. Spatial Decoder with Native 256x256 High-Frequency Refinement Stage.
        6. Monte Carlo Dropout support for Uncertainty Quantification.
        
        Args:
            cfg: Shared optics config. Output size is cfg.N; the input is cfg.crop_size.
            mode: 'hybrid' (modal + residual U-Net), 'modal' (pure modal), or 'unet' (pure spatial).
        """
        super().__init__()
        if features is None:
            features = [64, 128, 256]

        self.features = features
        self.mode = mode
        N = cfg.N
        self.N = N
        self.noll_indices = cfg.noll_indices
        self.num_zernike = len(self.noll_indices)

        # 1. Precompute circular aperture mask and Zernike basis (N x N) from the config.
        # Non-persistent: geometry always comes from the config, never from a checkpoint.
        geo = cfg.build_geometry()
        self.register_buffer('mask', geo.mask.unsqueeze(0).unsqueeze(0), persistent=False) # [1, 1, N, N]
        self.zernike_generator = DifferentiableZernikeGenerator(geo.basis)

        # 2. Residual Encoder Backbone (receives cropped intensity, e.g. 128x128)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        curr_in = in_channels
        for feature in features:
            self.encoder.append(ResidualBlock(curr_in, feature, negative_slope=negative_slope))
            curr_in = feature

        # 3. Bottleneck (with Spatial Dropout for UQ)
        bottleneck_channels = features[-1] * 2
        self.bottleneck = nn.Sequential(
            ResidualBlock(features[-1], bottleneck_channels, negative_slope=negative_slope),
            nn.Dropout2d(p=0.3)
        )

        # 4. Modal Zernike Head (GAP + Linear Head)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.zernike_head = nn.Sequential(
            nn.Linear(bottleneck_channels, 256),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(128, self.num_zernike)
        )

        # 5. Spatial Decoder with Attention Gates on Skip Connections
        self.upconvs = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        in_ch = bottleneck_channels
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2)
            )
            self.att_gates.append(
                AttentionGate(f_g=feature, f_l=feature, inter_channels=feature // 2)
            )
            self.decoder_blocks.append(
                ResidualBlock(feature * 2, feature, negative_slope=negative_slope)
            )
            in_ch = feature

        # 6. Native 256x256 High-Frequency Refinement Stage
        refine_channels = features[0] // 2  # 32 channels at 256x256
        self.up_to_256 = nn.Sequential(
            nn.Upsample(size=(N, N), mode='bilinear', align_corners=False),
            nn.Conv2d(features[0], refine_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.refinement_256 = ResidualBlock(refine_channels, refine_channels, negative_slope=negative_slope)
        self.final_conv = nn.Conv2d(refine_channels, out_channels, kernel_size=1)

    def freeze_modal_head(self):
        """Freezes the Zernike head and generator weights."""
        for param in self.zernike_head.parameters():
            param.requires_grad = False
        for param in self.zernike_generator.parameters():
            param.requires_grad = False

    def unfreeze_modal_head(self):
        """Unfreezes the Zernike head and generator weights."""
        for param in self.zernike_head.parameters():
            param.requires_grad = True
        for param in self.zernike_generator.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, return_coeffs: bool = False, return_components: bool = False):
        skip_connections = []

        # Run Residual Encoder (skip taken before pooling to preserve full spatial detail)
        out = x
        for enc in self.encoder:
            out = enc(out)
            skip_connections.append(out)
            out = self.pool(out)

        # Run Bottleneck
        bottleneck_feat = self.bottleneck(out)

        # Modal Head: GAP -> Linear Regression -> 256x256 Basis Synthesis
        gap_feat = self.gap(bottleneck_feat).flatten(1)
        pred_coeffs = self.zernike_head(gap_feat)
        pred_coeffs = torch.clamp(pred_coeffs, -20.0, 20.0)
        modal_phase = self.zernike_generator(pred_coeffs) # [B, 1, 256, 256]

        if self.mode == 'modal':
            total_phase = modal_phase * self.mask
            if return_components:
                return total_phase, pred_coeffs, modal_phase, torch.zeros_like(modal_phase)
            if return_coeffs:
                return total_phase, pred_coeffs
            return total_phase

        # Spatial Decoder with Attention-Gated Skip Connections
        dec_out = bottleneck_feat
        rev_skips = skip_connections[::-1]
        for i in range(len(self.upconvs)):
            dec_out = self.upconvs[i](dec_out)
            skip = rev_skips[i]
            # Attention Gate spatially filters skip features using coarser decoder state
            skip = self.att_gates[i](g=dec_out, x=skip)
            dec_out = torch.cat((skip, dec_out), dim=1)
            dec_out = self.decoder_blocks[i](dec_out)

        # Native 256x256 Refinement (resolves steep gradients and complex multi-lobe structures)
        dec_out = self.up_to_256(dec_out)
        dec_out = self.refinement_256(dec_out)
        residual_phase = self.final_conv(dec_out) # [B, 1, 256, 256]
        residual_phase = torch.clamp(residual_phase, -30.0, 30.0)

        if self.mode == 'unet':
            total_phase = residual_phase * self.mask
        else:
            # Hybrid mode: 256x256 Modal baseline + 256x256 Fine spatial residuals
            total_phase = (modal_phase + residual_phase) * self.mask

        if return_components:
            return total_phase, pred_coeffs, modal_phase, residual_phase
        if return_coeffs:
            return total_phase, pred_coeffs
        return total_phase

    def enable_mc_dropout(self):
        """Forces all dropout layers to remain active during evaluation mode."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

# Primary class alias for seamless compatibility with train.py and inference_uq.py
UNet = AttentionResUNet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = OpticsConfig()
    model = AttentionResUNet(cfg, in_channels=1, out_channels=1).to(device)

    # Cropped intensity input preserving outer diffraction rings
    dummy_intensity = torch.randn(4, 1, cfg.crop_size, cfg.crop_size, device=device)
    pred_phase, pred_coeffs, modal_phase, residual_phase = model(dummy_intensity, return_components=True)
    
    print("Attention Res-UNet initialized successfully!")
    print(f"Input Shape:          {dummy_intensity.shape}")
    print(f"Predicted Total Phase:{pred_phase.shape}")
    print(f"Modal Phase:          {modal_phase.shape}")
    print(f"Residual Phase:       {residual_phase.shape}")
    print(f"Predicted Coeffs:     {pred_coeffs.shape}")
    print(f"Num Zernike Modes:    {model.num_zernike}")
    
    all_trainable = all(p.requires_grad for p in model.parameters())
    print("All parameters trainable (requires_grad = True):", all_trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters:     {total_params:,}")

