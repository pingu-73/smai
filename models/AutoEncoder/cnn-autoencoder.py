import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),    # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                      # 28x28 -> 14x14
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),   # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                      # 14x14 -> 7x7
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                      # 7x7 -> 3x3
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # First deconvolutional block
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,        
                stride=2,
                padding=1,
                output_padding=1
            ),  # 3x3 -> 7x7
            nn.ReLU(),
            
            # Second deconvolutional block
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # 7x7 -> 14x14
            nn.ReLU(),
            
            # Third deconvolutional block
            nn.ConvTranspose2d(
                32, 1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # 14x14 -> 28x28
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed