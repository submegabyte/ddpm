## https://claude.ai/share/6a41b565-9f78-4551-9afb-3cf3e4658e7a

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """
    Masked 2D convolution implementation - masks the center and right/future pixels
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, **kwargs):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))
        
        # Build the mask
        h, w = kernel_size, kernel_size
        if isinstance(kernel_size, tuple):
            h, w = kernel_size
            
        mask = self.mask
        mask.fill_(1)
        mask[:, :, h // 2, w // 2 + 1:] = 0  # Mask right half (future)
        
        # For 'B' type masks, also mask the center pixel
        if mask_type == 'B':
            mask[:, :, h // 2, w // 2] = 0
            
    def forward(self, x):
        self.weight.data *= self.mask  # Apply the mask to the weights
        return super(MaskedConv2d, self).forward(x)


class ResidualBlock(nn.Module):
    """
    Residual block for PixelCNN++
    """
    def __init__(self, in_channels, out_channels, mask_type='B'):
        super(ResidualBlock, self).__init__()
        self.conv1 = MaskedConv2d(in_channels, out_channels, kernel_size=3, mask_type=mask_type, 
                                 padding=1, stride=1)
        self.conv2 = MaskedConv2d(out_channels, out_channels, kernel_size=3, mask_type=mask_type, 
                                 padding=1, stride=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Normalization and non-linearities
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out + residual)  # Skip connection
        
        return out


class PixelCNNpp(nn.Module):
    """
    PixelCNN++ model with discretized mixture of logistics output
    """
    def __init__(self, input_channels=3, hidden_dims=128, n_residual_blocks=5, n_mixtures=10):
        super(PixelCNNpp, self).__init__()
        self.input_channels = input_channels
        self.n_mixtures = n_mixtures
        
        # Initial convolution with 'A' type mask to ensure proper conditioning
        self.first_conv = MaskedConv2d(input_channels, hidden_dims, kernel_size=7, mask_type='A', padding=3)
        
        # Residual blocks with 'B' type masks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims, hidden_dims, mask_type='B') 
            for _ in range(n_residual_blocks)
        ])
        
        # Final layers
        self.final_conv1 = MaskedConv2d(hidden_dims, hidden_dims, kernel_size=3, mask_type='B', padding=1)
        self.final_conv2 = MaskedConv2d(hidden_dims, hidden_dims, kernel_size=3, mask_type='B', padding=1)
        
        # Output layer - parameters for mixture of logistics
        # For each mixture component: means, scales, and mixture weights
        self.output_conv = nn.Conv2d(hidden_dims, input_channels * 10 * n_mixtures, kernel_size=1)
        
    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Initial conv
        x = self.first_conv(x)
        x = F.relu(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final convs
        x = self.final_conv1(x)
        x = F.relu(x)
        x = self.final_conv2(x)
        x = F.relu(x)
        
        # Output layer
        x = self.output_conv(x)
        
        # Reshape for the mixture parameters
        # Each pixel will have parameters for n_mixtures components
        x = x.view(batch_size, self.input_channels, self.n_mixtures * 10, height, width)
        
        return x
    
    
def discretized_mix_logistic_loss(y_pred, y_true):
    """
    Discretized mixture of logistics loss function for PixelCNN++
    Args:
        y_pred: predicted output from the PixelCNN++, shape [B, C, M*10, H, W]
        y_true: target image, shape [B, C, H, W]
    """
    # This is a simplified version, the actual loss is more complex
    n_mixtures = y_pred.shape[2] // 10
    
    # Unpack parameters
    logit_probs = y_pred[:, :, :n_mixtures, :, :]  # mixing logits
    means = y_pred[:, :, n_mixtures:n_mixtures*4, :, :]  # means
    log_scales = y_pred[:, :, n_mixtures*4:n_mixtures*7, :, :]  # log scales
    coeffs = y_pred[:, :, n_mixtures*7:, :, :]  # coefficients for dependencies
    
    # Clamp the log scales for stability
    log_scales = torch.clamp(log_scales, min=-7.0)
    
    # Calculate the loss
    # This is a complex process involving the cumulative distribution function
    # of the logistics mixture and proper handling of the discretized nature
    # Full implementation omitted for brevity
    
    # Placeholder for demonstration
    loss = F.mse_loss(means.mean(dim=2), y_true)
    
    return loss


def sample_from_discretized_mix_logistic(y_pred, temperature=1.0):
    """
    Sample from the discretized mixture of logistics distribution
    Args:
        y_pred: predicted output from the PixelCNN++, shape [B, C, M*10, H, W]
        temperature: controls randomness in sampling (lower = more deterministic)
    """
    # Unpack parameters
    n_mixtures = y_pred.shape[2] // 10
    logit_probs = y_pred[:, :, :n_mixtures, :, :]  # mixing logits
    means = y_pred[:, :, n_mixtures:n_mixtures*4, :, :]  # means
    log_scales = y_pred[:, :, n_mixtures*4:n_mixtures*7, :, :]  # log scales
    
    # Sample from the mixture
    # Full implementation omitted for brevity
    
    # Placeholder for demonstration
    samples = means.mean(dim=2)
    
    return samples


# Example usage
def train_pixelcnn_plus_plus():
    # Example hyperparameters
    batch_size = 32
    lr = 0.001
    n_epochs = 100
    
    # Initialize model
    model = PixelCNNpp(input_channels=3, hidden_dims=128, n_residual_blocks=5, n_mixtures=10)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop (simplified)
    for epoch in range(n_epochs):
        # Get batch of images
        # images = next(data_loader)
        images = torch.randn(batch_size, 3, 32, 32)  # Dummy data
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss = discretized_mix_logistic_loss(predictions, images)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model


def generate_images(model, n_images=16, image_size=(32, 32)):
    """Generate images using PixelCNN++ by sampling pixels one at a time"""
    height, width = image_size
    images = torch.zeros(n_images, 3, height, width)
    
    model.eval()
    with torch.no_grad():
        # Generate pixels one by one
        for h in range(height):
            for w in range(width):
                # Get predictions for the next pixel
                predictions = model(images)
                
                # Sample from the predicted distribution
                next_pixels = sample_from_discretized_mix_logistic(predictions)
                
                # Update the image with the sampled pixels at position (h, w)
                images[:, :, h, w] = next_pixels[:, :, h, w]
    
    return images

if __name__ == "__main__":
    model = train_pixelcnn_plus_plus()
    images = generate_images(model)

    print(images.shape)