import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os
import argparse
from utils import load_data, denormalize_landmarks
from models import DualStreamModel
from utils import CephDataset
from torch.utils.data import DataLoader
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize cephalometric landmark predictions')
    parser.add_argument('--data_path', type=str, default='data/train_data_pure_depth.pkl',
                        help='Path to dataset pickle file')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/full_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--set_type', type=str, default='test',
                        choices=['train', 'dev', 'test'],
                        help='Dataset split to visualize')
    args = parser.parse_args()
    return args

def visualize_landmarks(image, landmarks_gt, landmarks_pred, output_path=None):
    """
    Visualize landmarks on an image
    
    Args:
        image: RGB image (H, W, C)
        landmarks_gt: Ground truth landmarks (num_landmarks, 2)
        landmarks_pred: Predicted landmarks (num_landmarks, 2)
        output_path: Path to save visualization
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Plot ground truth landmarks
    for i, (x, y) in enumerate(landmarks_gt):
        plt.plot(x, y, 'go', markersize=5)
        plt.text(x, y, str(i+1), color='white', fontsize=8, 
                 bbox=dict(facecolor='green', alpha=0.7))
    
    # Plot predicted landmarks
    for i, (x, y) in enumerate(landmarks_pred):
        plt.plot(x, y, 'ro', markersize=5)
        plt.text(x, y, str(i+1), color='white', fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.7))
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Ground Truth'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Prediction')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Cephalometric Landmark Detection')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define landmark columns
    landmark_cols = [
        'sella_x', 'sella_y', 
        'nasion_x', 'nasion_y', 
        'A point_x', 'A point_y',
        'B point_x', 'B point_y', 
        'upper 1 tip_x', 'upper 1 tip_y',
        'upper 1 apex_x', 'upper 1 apex_y', 
        'lower 1 tip_x', 'lower 1 tip_y',
        'lower 1 apex_x', 'lower 1 apex_y', 
        'ANS_x', 'ANS_y', 
        'PNS_x', 'PNS_y',
        'Gonion _x', 'Gonion _y', 
        'Menton_x', 'Menton_y', 
        'ST Nasion_x', 'ST Nasion_y',
        'Tip of the nose_x', 'Tip of the nose_y', 
        'Subnasal_x', 'Subnasal_y',
        'Upper lip_x', 'Upper lip_y', 
        'Lower lip_x', 'Lower lip_y',
        'ST Pogonion_x', 'ST Pogonion_y', 
        'gnathion_x', 'gnathion_y'
    ]
    
    # Number of landmarks
    num_landmarks = len(landmark_cols) // 2
    
    # Load data
    print("Loading dataset...")
    data = load_data(args.data_path)
    
    # Create dataset and dataloader
    dataset = CephDataset(data, landmark_cols, set_type=args.set_type)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("Loading model...")
    model = DualStreamModel(num_landmarks=num_landmarks)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Visualize samples
    print(f"Visualizing {args.num_samples} samples...")
    for i, batch in enumerate(dataloader):
        if i >= args.num_samples:
            break
        
        # Get data
        rgb_image = batch['rgb_image'].to(device)
        depth_map = batch['depth_map'].to(device)
        landmarks_gt = batch['landmarks'].to(device)
        patient_id = batch['patient_id'][0]
        
        # Make prediction
        with torch.no_grad():
            outputs = model(rgb_image, depth_map)
            landmarks_pred = outputs['final_coords']
        
        # Convert tensors to numpy arrays
        rgb_image_np = rgb_image[0].permute(1, 2, 0).cpu().numpy()
        landmarks_gt_np = denormalize_landmarks(landmarks_gt[0].cpu().numpy())
        landmarks_pred_np = denormalize_landmarks(landmarks_pred[0].cpu().numpy())
        
        # Visualize
        output_path = os.path.join(args.output_dir, f"{patient_id}_landmarks.png")
        visualize_landmarks(rgb_image_np, landmarks_gt_np, landmarks_pred_np, output_path)
        
        # Also visualize RGB with depth overlay
        depth_map_np = depth_map[0, 0].cpu().numpy()
        depth_colored = cv2.applyColorMap((depth_map_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay image
        overlay = rgb_image_np.copy()
        alpha = 0.5
        overlay = (1 - alpha) * rgb_image_np + alpha * depth_colored / 255.0
        overlay = np.clip(overlay, 0, 1)
        
        # Visualize overlay with landmarks
        output_path = os.path.join(args.output_dir, f"{patient_id}_depth_overlay.png")
        visualize_landmarks(overlay, landmarks_gt_np, landmarks_pred_np, output_path)
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main() 