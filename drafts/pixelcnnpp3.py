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
    Residual block for PixelCNNpp
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
    PixelCNNpp model with discretized mixture of logistics output
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
        # 10 parameters per mixture * n_mixtures
        # self.output_layer = nn.Conv2d(hidden_dims, input_channels * 10 * n_mixtures, kernel_size=1)
        self.output_layer = nn.Conv2d(hidden_dims, input_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the PixelCNNpp model
        Args:
            x: Input image [B, C, H, W]
        Returns:
            Distribution parameters with the same spatial dimensions as input
        """
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
        x = self.output_layer(x)  # [B, C*10*n_mixtures, H, W]
        
        # Reshape for the mixture parameters while preserving spatial dimensions
        # Each pixel will have parameters for n_mixtures components
        # x = x.reshape(batch_size, self.input_channels, 10 * self.n_mixtures, height, width)
        x = x.reshape(batch_size, self.input_channels, height, width)
        
        return x
    
    def sample(self, batch_size=1, image_size=(32, 32)):
        """
        Sample new images from the model
        Args:
            batch_size: Number of images to sample
            image_size: Size of the images to generate (height, width)
        Returns:
            Generated images [B, C, H, W]
        """
        height, width = image_size
        device = next(self.parameters()).device
        
        # Start with zeros
        samples = torch.zeros(batch_size, self.input_channels, height, width, device=device)
        
        # Generate each pixel sequentially
        with torch.no_grad():
            for h in range(height):
                for w in range(width):
                    # Get predictions for current pixel
                    out = self.forward(samples)
                    
                    # Extract distribution parameters for this pixel
                    mixture_params = out[:, :, :, h, w]
                    
                    # Sample from the predicted distribution at this pixel
                    new_pixel = sample_from_discretized_mix_logistic(mixture_params)
                    
                    # Update the sample
                    samples[:, :, h, w] = new_pixel
        
        return samples


def discretized_mix_logistic_loss(y_pred, y_true):
    """
    Discretized mixture of logistics loss function for PixelCNNpp
    Args:
        y_pred: predicted output from the PixelCNNpp, shape [B, C, M*10, H, W]
        y_true: target image, shape [B, C, H, W]
    """
    batch_size, n_channels, _, height, width = y_pred.shape
    
    # Reshape y_true to match dimensions for calculations
    y_true = y_true.unsqueeze(2)  # [B, C, 1, H, W]
    
    # Extract parameters (simplified)
    n_mixtures = y_pred.shape[2] // 10
    
    # Mixture weights [B, C, M, H, W]
    logit_probs = y_pred[:, :, :n_mixtures, :, :]
    
    # Means for each mixture and channel [B, C, M, H, W]
    means = y_pred[:, :, n_mixtures:2*n_mixtures, :, :]
    
    # Log scales [B, C, M, H, W]
    log_scales = torch.clamp(y_pred[:, :, 2*n_mixtures:3*n_mixtures, :, :], min=-7.0)
    
    # Convert to scales
    scales = torch.exp(log_scales)
    
    # Calculate log-likelihood for each mixture component
    centered_y = y_true - means
    inv_stdv = 1.0 / scales
    
    plus_in = inv_stdv * (centered_y + 1.0/255.0)
    cdf_plus = torch.sigmoid(plus_in)
    
    min_in = inv_stdv * (centered_y - 1.0/255.0)
    cdf_min = torch.sigmoid(min_in)
    
    # Log probability for edge cases (0 and 255)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    
    # Log probability in the middle of the bin
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_y
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    
    # Log probability for all cases
    log_probs = torch.where(
        y_true < -0.999,
        log_cdf_plus,
        torch.where(
            y_true > 0.999,
            log_one_minus_cdf_min,
            torch.where(
                cdf_delta > 1e-5,
                torch.log(torch.clamp(cdf_delta, min=1e-12)),
                log_pdf_mid - torch.log(torch.tensor(127.5, device=y_pred.device))
            )
        )
    )
    
    # Apply mixture weights
    log_probs = log_probs + F.log_softmax(logit_probs, dim=2)
    
    # Sum over mixtures
    log_probs = torch.logsumexp(log_probs, dim=2)
    
    # Sum over channels and pixels
    return -torch.sum(log_probs) / (batch_size * height * width)


def sample_from_discretized_mix_logistic(mixture_params, temperature=1.0):
    """
    Sample from the discretized mixture of logistics distribution for a single pixel
    Args:
        mixture_params: Parameters for the mixture, shape [B, C, M*10]
        temperature: Controls randomness in sampling (lower = more deterministic)
    Returns:
        Sampled pixel values for each channel [B, C]
    """
    n_mixtures = mixture_params.shape[2] // 10
    batch_size, n_channels = mixture_params.shape[0], mixture_params.shape[1]
    
    # Extract parameters
    logit_probs = mixture_params[:, :, :n_mixtures]  # [B, C, M]
    means = mixture_params[:, :, n_mixtures:2*n_mixtures]  # [B, C, M]
    log_scales = torch.clamp(mixture_params[:, :, 2*n_mixtures:3*n_mixtures], min=-7.0)
    
    # Sample mixture components
    temp_adjusted_probs = logit_probs / temperature
    mixture_idxs = torch.distributions.Categorical(
        logits=temp_adjusted_probs.permute(0, 2, 1)  # [B, M, C]
    ).sample().unsqueeze(2)  # [B, C, 1]
    
    # Gather means and scales for the sampled mixtures
    batch_indices = torch.arange(batch_size).view(-1, 1, 1).to(means.device)
    channel_indices = torch.arange(n_channels).view(1, -1, 1).to(means.device)
    
    selected_means = means[batch_indices, channel_indices, mixture_idxs]  # [B, C, 1]
    selected_scales = torch.exp(log_scales[batch_indices, channel_indices, mixture_idxs])
    
    # Sample from the selected logistic distributions
    uniform = torch.rand_like(selected_means)
    logistic_samples = selected_means + selected_scales * (
        torch.log(uniform) - torch.log(1 - uniform)
    )
    
    # Clamp to [-1, 1]
    samples = torch.clamp(logistic_samples, -1, 1).squeeze(2)
    
    return samples


# Example training function
def train_pixelcnnpp(model, train_loader, optimizer, epochs=10, device='cuda'):
    """
    Train a PixelCNNpp model
    Args:
        model: PixelCNNpp model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to train on
    """
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Normalize data to [-1, 1]
            data = (data - 0.5) * 2.0
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = discretized_mix_logistic_loss(output, data)
            loss.backward()
            
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.6f}')
        
        # Generate sample images
        if epoch % 5 == 0:
            generate_and_save_samples(model, f'samples_epoch_{epoch}.png', device)
    
    return model


def generate_and_save_samples(model, filename, device='cuda', n_samples=16, image_size=(32, 32)):
    """
    Generate and save sample images from the model
    Args:
        model: Trained PixelCNNpp model
        filename: Filename to save the samples
        device: Device to generate on
        n_samples: Number of samples to generate
        image_size: Size of the samples
    """
    model.eval()
    samples = model.sample(n_samples, image_size).cpu()
    
    # Convert from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2.0
    
    # Save using torchvision or similar
    # torchvision.utils.save_image(samples, filename, nrow=int(n_samples**0.5))
    
    # For now, just print that samples were generated
    print(f"Generated {n_samples} samples with shape {samples.shape}")
    

# Example usage
if __name__ == "__main__":
    # Create model
    model = PixelCNNpp(input_channels=3, hidden_dims=128, n_residual_blocks=5, n_mixtures=10)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy data
    batch_size = 32
    dummy_loader = [(torch.rand(batch_size, 3, 32, 32), torch.zeros(batch_size)) for _ in range(10)]
    
    # Train model
    train_pixelcnnpp(model, dummy_loader, optimizer, epochs=1, device='cpu')
    
    # Generate samples
    samples = model.sample(batch_size=4, image_size=(32, 32))
    print(f"Generated samples shape: {samples.shape}")