# %%
import torch
import os
import pandas as pd
from utils import load_data, create_dataloaders
from trainer import Trainer

# %%
# Configuration settings - adjust these as needed
config = {
    # Dataset paths
    'data_path': 'data/train_data_pure_depth.pkl',
    
    # Training parameters
    'batch_size': 16,
    'num_epochs': 50,
    'num_epochs_mlp': 20,
    'num_epochs_finetune': 30,
    'learning_rate': 1e-4,
    'learning_rate_finetune': 1e-5,
    
    # Output paths
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'experiment_name': 'ceph_landmarks',
}

# %%
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

# Number of landmarks (19 landmarks, each with x and y coordinates)
config['num_landmarks'] = len(landmark_cols) // 2
config['landmark_cols'] = landmark_cols

# %%
# Setup device and directories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Make sure checkpoint and log directories exist
os.makedirs(config['checkpoint_dir'], exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

# %%
# Load data and create dataloaders
print("Loading dataset...")
data = load_data(config['data_path'])

# Create dataloaders
print("Creating dataloaders...")
train_loader, val_loader, test_loader = create_dataloaders(data, landmark_cols, batch_size=config['batch_size'])
print(f"Train set: {len(train_loader.dataset)} samples")
print(f"Validation set: {len(val_loader.dataset)} samples")
print(f"Test set: {len(test_loader.dataset)} samples")

# %%
# Create trainer
trainer = Trainer(config)

# %%
# Train RGB stream
print("Training RGB Stream...")
trainer.train_rgb_stream(train_loader, val_loader)

# %%
# Train Depth stream
print("Training Depth Stream...")
trainer.train_depth_stream(train_loader, val_loader)

# %%
# Train MLP refinement (with frozen streams)
print("Training Refinement MLP...")
trainer.train_mlp(train_loader, val_loader)

# %%
# Fine-tune full model
print("Fine-tuning Full Model...")
trainer.train_full_model(train_loader, val_loader)

# %%
# Evaluate on test set
print("Evaluating on test set...")
test_mre = trainer.test(test_loader)
print(f"Final Test MRE: {test_mre:.2f}") 