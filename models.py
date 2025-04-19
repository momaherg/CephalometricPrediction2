import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from utils import soft_argmax

class HRNetBackbone(nn.Module):
    """HRNet backbone for feature extraction"""
    
    def __init__(self, pretrained=True, in_channels=3):
        super(HRNetBackbone, self).__init__()
        
        # Load pretrained HRNet model
        self.hrnet = timm.create_model('hrnet_w32', pretrained=pretrained, features_only=True)
        
        # If input is depth map (1 channel), modify first conv layer
        if in_channels != 3:
            old_conv = self.hrnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            # Initialize new conv weights using pretrained weights
            if pretrained:
                with torch.no_grad():
                    # Average the weights across the input channels
                    new_conv.weight[:, 0:1, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                    if old_conv.bias is not None:
                        new_conv.bias = old_conv.bias
            
            self.hrnet.conv1 = new_conv
    
    def forward(self, x):
        # Get multi-scale features from HRNet
        features = self.hrnet(x)
        
        # Return the highest resolution feature map
        return features[-1]

class HeatmapHead(nn.Module):
    """Head module to generate heatmaps from features"""
    
    def __init__(self, in_channels, num_landmarks):
        super(HeatmapHead, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_landmarks, kernel_size=1)
        )
    
    def forward(self, x):
        return self.conv(x)

class RefinementMLP(nn.Module):
    """MLP for fusing and refining coordinates from both streams"""
    
    def __init__(self, num_landmarks):
        super(RefinementMLP, self).__init__()
        
        input_dim = num_landmarks * 2 * 2  # Both RGB and depth coords
        hidden_dim = 512
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_landmarks * 2)  # Output coordinates
        )
        
        self.num_landmarks = num_landmarks
    
    def forward(self, rgb_coords, depth_coords):
        # Flatten coordinates
        batch_size = rgb_coords.size(0)
        rgb_flat = rgb_coords.view(batch_size, -1)
        depth_flat = depth_coords.view(batch_size, -1)
        
        # Concatenate
        fused = torch.cat([rgb_flat, depth_flat], dim=1)
        
        # Process through MLP
        output = self.mlp(fused)
        
        # Reshape back to coordinate format
        return output.view(batch_size, self.num_landmarks, 2)

class RGBStream(nn.Module):
    """RGB stream of the dual-stream model"""
    
    def __init__(self, num_landmarks, pretrained=True):
        super(RGBStream, self).__init__()
        
        self.backbone = HRNetBackbone(pretrained=pretrained, in_channels=3)
        self.heatmap_head = HeatmapHead(in_channels=32, num_landmarks=num_landmarks)
    
    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.heatmap_head(features)
        coords = soft_argmax(heatmaps)
        return heatmaps, coords

class DepthStream(nn.Module):
    """Depth stream of the dual-stream model"""
    
    def __init__(self, num_landmarks, pretrained=True):
        super(DepthStream, self).__init__()
        
        self.backbone = HRNetBackbone(pretrained=pretrained, in_channels=1)
        self.heatmap_head = HeatmapHead(in_channels=32, num_landmarks=num_landmarks)
    
    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.heatmap_head(features)
        coords = soft_argmax(heatmaps)
        return heatmaps, coords

class DualStreamModel(nn.Module):
    """Full dual-stream model with refinement MLP"""
    
    def __init__(self, num_landmarks, pretrained=True):
        super(DualStreamModel, self).__init__()
        
        self.rgb_stream = RGBStream(num_landmarks, pretrained)
        self.depth_stream = DepthStream(num_landmarks, pretrained)
        self.refinement_mlp = RefinementMLP(num_landmarks)
        
        # Flags for freezing streams during training
        self.freeze_streams = False
    
    def forward(self, rgb_image, depth_map):
        rgb_heatmaps, rgb_coords = self.rgb_stream(rgb_image)
        depth_heatmaps, depth_coords = self.depth_stream(depth_map)
        
        # Refinement
        final_coords = self.refinement_mlp(rgb_coords, depth_coords)
        
        return {
            'rgb_heatmaps': rgb_heatmaps,
            'depth_heatmaps': depth_heatmaps,
            'rgb_coords': rgb_coords,
            'depth_coords': depth_coords,
            'final_coords': final_coords
        }
    
    def freeze_stream_parameters(self):
        """Freeze parameters of both streams for MLP training"""
        for param in self.rgb_stream.parameters():
            param.requires_grad = False
        for param in self.depth_stream.parameters():
            param.requires_grad = False
        self.freeze_streams = True
    
    def unfreeze_all_parameters(self):
        """Unfreeze all parameters for final training"""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_streams = False 