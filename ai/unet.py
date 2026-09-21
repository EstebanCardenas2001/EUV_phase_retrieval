import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256]):
        """
        U-Net Architecture equipped with Monte Carlo Dropout for Uncertainty Quantification.
        """
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. Build the Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 2. Build the Bottleneck (with Dropout for UQ)
        # We use Dropout2d to drop entire feature maps rather than individual pixels
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1] * 2),
            nn.Dropout2d(p=0.3) 
        )

        # 3. Build the Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature*2, feature))

        # 4. Final output layer maps back to a single phase channel
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Run the Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Run the Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for the decoder
        skip_connections = skip_connections[::-1]

        # Run the Decoder
        for i in range(0, len(self.decoder), 2):
            # Upsample
            x = self.decoder[i](x)
            
            # Fetch the skip connection
            skip = skip_connections[i//2]
            
            # Concatenate skip connection and upsampled features along the channel dimension
            x = torch.cat((skip, x), dim=1)
            
            # Run the DoubleConv block
            x = self.decoder[i+1](x)

        return self.final_conv(x)

    def enable_mc_dropout(self):
        """
        Forces all dropout layers to remain active during evaluation mode.
        This is the mathematical core of Monte Carlo Uncertainty Quantification.
        """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

if __name__ == "__main__":
    # Test the architecture geometry
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Create a dummy intensity tensor matching our dataloader output: [Batch, Channel, Height, Width]
    dummy_intensity = torch.randn(4, 1, 256, 256).to(device)
    
    # Push it through the network
    predicted_phase = model(dummy_intensity)
    
    print(f"Input Shape:  {dummy_intensity.shape}")
    print(f"Output Shape: {predicted_phase.shape}")