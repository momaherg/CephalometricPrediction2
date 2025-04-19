import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """Load the dataset from pickle file"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def normalize_landmarks(landmarks, img_size=224):
    """Normalize landmarks to range [0, 1]"""
    return landmarks / img_size

def denormalize_landmarks(landmarks, img_size=224):
    """Denormalize landmarks from range [0, 1] to original scale"""
    return landmarks * img_size

def calculate_mre(pred, target, img_size=224):
    """
    Calculate Mean Radial Error (MRE)
    
    Args:
        pred: Predicted landmarks (batch_size, num_landmarks, 2)
        target: Target landmarks (batch_size, num_landmarks, 2)
        img_size: Image size for normalization
        
    Returns:
        Mean Radial Error in pixels
    """
    # Denormalize if needed
    if pred.max() <= 1.0 and target.max() <= 1.0:
        pred = denormalize_landmarks(pred, img_size)
        target = denormalize_landmarks(target, img_size)
    
    # Calculate Euclidean distance for each landmark
    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))
    
    # Mean across all landmarks
    mre_per_sample = torch.mean(distances, dim=1)
    
    # Mean across batch
    mre = torch.mean(mre_per_sample)
    
    return mre

def generate_heatmaps(landmarks, img_size=224, sigma=2.0):
    """
    Generate Gaussian heatmaps from landmarks coordinates
    
    Args:
        landmarks: Tensor of shape (batch_size, num_landmarks, 2)
        img_size: Size of the output heatmaps
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Heatmaps tensor of shape (batch_size, num_landmarks, img_size, img_size)
    """
    batch_size, num_landmarks, _ = landmarks.shape
    heatmaps = torch.zeros(batch_size, num_landmarks, img_size, img_size)
    
    # Create coordinate grid
    x = torch.arange(img_size).repeat(img_size, 1)
    y = torch.arange(img_size).repeat(img_size, 1).t()
    
    x = x.unsqueeze(0).repeat(batch_size, num_landmarks, 1, 1)
    y = y.unsqueeze(0).repeat(batch_size, num_landmarks, 1, 1)
    
    # Denormalize landmarks if they are normalized
    if landmarks.max() <= 1.0:
        lm = landmarks * img_size
    else:
        lm = landmarks.clone()
    
    # Calculate Gaussian for each landmark
    for b in range(batch_size):
        for l in range(num_landmarks):
            lm_x, lm_y = lm[b, l]
            
            # Calculate Gaussian
            gaussian = torch.exp(-((x[b, l] - lm_x)**2 + (y[b, l] - lm_y)**2) / (2 * sigma**2))
            
            # Normalize to sum to 1
            gaussian = gaussian / gaussian.sum()
            
            heatmaps[b, l] = gaussian
    
    return heatmaps

def soft_argmax(heatmaps):
    """
    Soft-argmax function to extract coordinates from heatmaps
    
    Args:
        heatmaps: Tensor of shape (batch_size, num_landmarks, height, width)
        
    Returns:
        Coordinates tensor of shape (batch_size, num_landmarks, 2)
    """
    batch_size, num_landmarks, height, width = heatmaps.shape
    
    # Create coordinate grid
    x = torch.arange(width).float()
    y = torch.arange(height).float()
    
    # Normalize coordinates to [0, 1]
    x = x / (width - 1)
    y = y / (height - 1)
    
    # Create meshgrid
    yy, xx = torch.meshgrid(y, x)
    xx = xx.to(heatmaps.device)
    yy = yy.to(heatmaps.device)
    
    # Reshape heatmaps for softmax
    heatmaps_flat = heatmaps.reshape(batch_size, num_landmarks, -1)
    
    # Apply softmax to get probabilities
    heatmaps_prob = torch.nn.functional.softmax(heatmaps_flat * 100, dim=2)
    heatmaps_prob = heatmaps_prob.reshape(batch_size, num_landmarks, height, width)
    
    # Calculate expected coordinates
    expected_x = torch.sum(heatmaps_prob * xx.unsqueeze(0).unsqueeze(0), dim=(2, 3))
    expected_y = torch.sum(heatmaps_prob * yy.unsqueeze(0).unsqueeze(0), dim=(2, 3))
    
    # Stack coordinates
    coords = torch.stack([expected_x, expected_y], dim=2)
    
    return coords

class CephDataset(Dataset):
    """Dataset for cephalometric landmark detection"""
    
    def __init__(self, data, landmark_cols, set_type='train', transform=None):
        """
        Args:
            data: DataFrame containing the dataset
            landmark_cols: List of landmark column names
            set_type: 'train', 'dev', or 'test'
            transform: Optional transforms to apply to images
        """
        self.data = data[data['set'] == set_type].reset_index(drop=True)
        self.landmark_cols = landmark_cols
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get RGB image
        rgb_image = self.data.iloc[idx]['Image'].astype(np.float32) / 255.0
        rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        # Get depth map
        depth_map = self.data.iloc[idx]['depth_feature'].astype(np.float32)
        depth_map = torch.from_numpy(depth_map).unsqueeze(0)  # Add channel dimension
        
        # Extract landmark coordinates
        landmarks = []
        for i in range(0, len(self.landmark_cols), 2):
            x_col = self.landmark_cols[i]
            y_col = self.landmark_cols[i+1]
            x = self.data.iloc[idx][x_col]
            y = self.data.iloc[idx][y_col]
            landmarks.append([x, y])
        
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        landmarks = normalize_landmarks(landmarks)  # Normalize to [0, 1]
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_map = self.transform(depth_map)
        
        return {
            'rgb_image': rgb_image,
            'depth_map': depth_map,
            'landmarks': landmarks,
            'patient_id': self.data.iloc[idx]['patient_id']
        }

def create_dataloaders(data, landmark_cols, batch_size=16):
    """Create DataLoaders for train, validation, and test sets"""
    train_dataset = CephDataset(data, landmark_cols, set_type='train')
    val_dataset = CephDataset(data, landmark_cols, set_type='dev')
    test_dataset = CephDataset(data, landmark_cols, set_type='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader 