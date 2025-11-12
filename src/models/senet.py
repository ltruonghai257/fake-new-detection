import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze and Excitation Block Module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool1d(x, 1)  # Squeeze
        w = self.fc(w)
        w, b = w.split(w.data.size(1) // 2, dim=1)  # Excitation
        w = torch.sigmoid(w)
        return x * w + b  # Scale and add bias


class ResBlock(nn.Module):
    """Residual Block with SEBlock."""
    
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv_lower = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        self.conv_upper = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels)
        )
        self.se_block = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path = self.conv_lower(x)
        path = self.conv_upper(path)
        path = self.se_block(path)
        path = x + path
        return F.relu(path)


class SENetwork(nn.Module):
    """SE-ResNet Network Module for feature extraction and classification."""
    
    def __init__(self, in_channel: int, filters: int, blocks: int, num_classes: int):
        super(SENetwork, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channel, filters, 3, padding=1, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks - 1)])
        self.out_conv = nn.Sequential(
            nn.Conv1d(filters, 128, 1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.data.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class SEAttentionModule(nn.Module):
    """SE-based attention module for multimodal fusion."""
    
    def __init__(self, text_dim: int, image_dim: int, correlation_dim: int, 
                 filters: int = 128, blocks: int = 19):
        super(SEAttentionModule, self).__init__()
        self.senet = SENetwork(
            in_channel=3,  # text, image, correlation
            filters=filters,
            blocks=blocks,
            num_classes=3  # attention weights for 3 modalities
        )
        
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor, 
                correlation_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (batch_size, text_dim)
            image_features: (batch_size, image_dim) 
            correlation_features: (batch_size, correlation_dim)
        Returns:
            attention_weights: (batch_size, 3)
        """
        # Reshape features for 1D convolution
        text_se = text_features.unsqueeze(-1)  # (batch_size, text_dim, 1)
        image_se = image_features.unsqueeze(-1)  # (batch_size, image_dim, 1)
        correlation_se = correlation_features.unsqueeze(-1)  # (batch_size, correlation_dim, 1)
        
        # Pad features to same dimension if needed
        max_dim = max(text_features.size(1), image_features.size(1), correlation_features.size(1))
        
        if text_features.size(1) < max_dim:
            padding = torch.zeros(text_features.size(0), max_dim - text_features.size(1), 1, 
                                device=text_features.device)
            text_se = torch.cat([text_se, padding], dim=1)
            
        if image_features.size(1) < max_dim:
            padding = torch.zeros(image_features.size(0), max_dim - image_features.size(1), 1,
                                device=image_features.device)
            image_se = torch.cat([image_se, padding], dim=1)
            
        if correlation_features.size(1) < max_dim:
            padding = torch.zeros(correlation_features.size(0), max_dim - correlation_features.size(1), 1,
                                device=correlation_features.device)
            correlation_se = torch.cat([correlation_se, padding], dim=1)
        
        # Concatenate along channel dimension
        combined_features = torch.cat([text_se, image_se, correlation_se], dim=-1)  # (batch_size, max_dim, 3)
        combined_features = combined_features.permute(0, 2, 1)  # (batch_size, 3, max_dim)
        
        # Get attention weights
        attention_weights = self.senet(combined_features)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        return attention_weights


class AdaptiveSEBlock(nn.Module):
    """Adaptive SE block that can handle variable input dimensions."""
    
    def __init__(self, reduction: int = 16):
        super(AdaptiveSEBlock, self).__init__()
        self.reduction = reduction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.size(1)
        reduced_channels = max(1, channels // self.reduction)
        
        # Create SE layers dynamically
        fc = nn.Sequential(
            nn.Conv1d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(reduced_channels, channels * 2, 1, bias=False),
        ).to(x.device)
        
        w = F.adaptive_avg_pool1d(x, 1)  # Squeeze
        w = fc(w)
        w, b = w.split(channels, dim=1)  # Excitation
        w = torch.sigmoid(w)
        
        return x * w + b  # Scale and add bias


class MultiScaleSEBlock(nn.Module):
    """Multi-scale SE block for capturing features at different scales."""
    
    def __init__(self, channels: int, scales: tuple = (1, 3, 5), reduction: int = 16):
        super(MultiScaleSEBlock, self).__init__()
        self.scales = scales
        self.se_blocks = nn.ModuleList([
            SEBlock(channels, reduction) for _ in scales
        ])
        self.fusion = nn.Conv1d(channels * len(scales), channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                scale_out = self.se_blocks[i](x)
            else:
                # Apply different kernel sizes for multi-scale processing
                conv = nn.Conv1d(x.size(1), x.size(1), scale, padding=scale//2, groups=x.size(1)).to(x.device)
                scale_x = conv(x)
                scale_out = self.se_blocks[i](scale_x)
            scale_outputs.append(scale_out)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=1)
        output = self.fusion(fused)
        
        return output + x  # Residual connection
