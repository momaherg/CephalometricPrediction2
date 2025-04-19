import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
import time
from models import RGBStream, DepthStream, DualStreamModel
from utils import calculate_mre

class Trainer:
    """Trainer class for landmark detection models"""
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], config['experiment_name']))
        
        # Create output directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize variables
        self.rgb_model = None
        self.depth_model = None
        self.dual_model = None
        self.current_stage = None
    
    def train_rgb_stream(self, train_loader, val_loader):
        """Train RGB stream model"""
        print("=== Training RGB Stream ===")
        self.current_stage = "rgb"
        
        # Create model
        self.rgb_model = RGBStream(num_landmarks=self.config['num_landmarks'], pretrained=True)
        self.rgb_model = self.rgb_model.to(self.device)
        
        # Create optimizer and loss
        optimizer = optim.Adam(self.rgb_model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()
        
        best_val_mre = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            # Training
            self.rgb_model.train()
            train_loss = 0
            train_mre = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            for batch in pbar:
                rgb_images = batch['rgb_image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                optimizer.zero_grad()
                
                heatmaps, coords = self.rgb_model(rgb_images)
                
                # Calculate loss
                loss = criterion(coords, landmarks)
                
                loss.backward()
                optimizer.step()
                
                # Calculate MRE
                mre = calculate_mre(coords.detach(), landmarks.detach())
                
                train_loss += loss.item()
                train_mre += mre.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mre': mre.item()
                })
            
            train_loss /= len(train_loader)
            train_mre /= len(train_loader)
            
            # Validation
            val_loss, val_mre = self._validate(self.rgb_model, val_loader, criterion, 'rgb')
            
            # Scheduler step
            scheduler.step(val_mre)
            
            # Log metrics
            self.writer.add_scalar('RGB/Loss/Train', train_loss, epoch)
            self.writer.add_scalar('RGB/Loss/Val', val_loss, epoch)
            self.writer.add_scalar('RGB/MRE/Train', train_mre, epoch)
            self.writer.add_scalar('RGB/MRE/Val', val_mre, epoch)
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train MRE: {train_mre:.2f}, "
                  f"Val Loss: {val_loss:.4f}, Val MRE: {val_mre:.2f}")
            
            # Save best model
            if val_mre < best_val_mre:
                best_val_mre = val_mre
                self._save_checkpoint(self.rgb_model, optimizer, epoch, val_mre, "rgb_best")
            
            # Save latest model
            self._save_checkpoint(self.rgb_model, optimizer, epoch, val_mre, "rgb_latest")
    
    def train_depth_stream(self, train_loader, val_loader):
        """Train Depth stream model"""
        print("=== Training Depth Stream ===")
        self.current_stage = "depth"
        
        # Create model
        self.depth_model = DepthStream(num_landmarks=self.config['num_landmarks'], pretrained=True)
        self.depth_model = self.depth_model.to(self.device)
        
        # Create optimizer and loss
        optimizer = optim.Adam(self.depth_model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()
        
        best_val_mre = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            # Training
            self.depth_model.train()
            train_loss = 0
            train_mre = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            for batch in pbar:
                depth_maps = batch['depth_map'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                optimizer.zero_grad()
                
                heatmaps, coords = self.depth_model(depth_maps)
                
                # Calculate loss
                loss = criterion(coords, landmarks)
                
                loss.backward()
                optimizer.step()
                
                # Calculate MRE
                mre = calculate_mre(coords.detach(), landmarks.detach())
                
                train_loss += loss.item()
                train_mre += mre.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mre': mre.item()
                })
            
            train_loss /= len(train_loader)
            train_mre /= len(train_loader)
            
            # Validation
            val_loss, val_mre = self._validate(self.depth_model, val_loader, criterion, 'depth')
            
            # Scheduler step
            scheduler.step(val_mre)
            
            # Log metrics
            self.writer.add_scalar('Depth/Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Depth/Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Depth/MRE/Train', train_mre, epoch)
            self.writer.add_scalar('Depth/MRE/Val', val_mre, epoch)
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train MRE: {train_mre:.2f}, "
                  f"Val Loss: {val_loss:.4f}, Val MRE: {val_mre:.2f}")
            
            # Save best model
            if val_mre < best_val_mre:
                best_val_mre = val_mre
                self._save_checkpoint(self.depth_model, optimizer, epoch, val_mre, "depth_best")
            
            # Save latest model
            self._save_checkpoint(self.depth_model, optimizer, epoch, val_mre, "depth_latest")
    
    def train_mlp(self, train_loader, val_loader):
        """Train refinement MLP with frozen streams"""
        print("=== Training Refinement MLP ===")
        self.current_stage = "mlp"
        
        # Create dual model
        self.dual_model = DualStreamModel(num_landmarks=self.config['num_landmarks'], pretrained=True)
        self.dual_model = self.dual_model.to(self.device)
        
        # Load pretrained stream weights
        rgb_checkpoint = torch.load(
            os.path.join(self.config['checkpoint_dir'], "rgb_best.pth"),
            map_location=self.device
        )
        depth_checkpoint = torch.load(
            os.path.join(self.config['checkpoint_dir'], "depth_best.pth"),
            map_location=self.device
        )
        
        # Load weights to streams
        self.dual_model.rgb_stream.load_state_dict(rgb_checkpoint['model_state_dict'])
        self.dual_model.depth_stream.load_state_dict(depth_checkpoint['model_state_dict'])
        
        # Freeze stream parameters
        self.dual_model.freeze_stream_parameters()
        
        # Create optimizer and loss for MLP only
        optimizer = optim.Adam(self.dual_model.refinement_mlp.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()
        
        best_val_mre = float('inf')
        
        for epoch in range(self.config['num_epochs_mlp']):
            # Training
            self.dual_model.train()
            train_loss = 0
            train_mre = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs_mlp']}")
            for batch in pbar:
                rgb_images = batch['rgb_image'].to(self.device)
                depth_maps = batch['depth_map'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.dual_model(rgb_images, depth_maps)
                final_coords = outputs['final_coords']
                
                # Calculate loss
                loss = criterion(final_coords, landmarks)
                
                loss.backward()
                optimizer.step()
                
                # Calculate MRE
                mre = calculate_mre(final_coords.detach(), landmarks.detach())
                
                train_loss += loss.item()
                train_mre += mre.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mre': mre.item()
                })
            
            train_loss /= len(train_loader)
            train_mre /= len(train_loader)
            
            # Validation
            val_loss, val_mre = self._validate_dual(val_loader, criterion)
            
            # Scheduler step
            scheduler.step(val_mre)
            
            # Log metrics
            self.writer.add_scalar('MLP/Loss/Train', train_loss, epoch)
            self.writer.add_scalar('MLP/Loss/Val', val_loss, epoch)
            self.writer.add_scalar('MLP/MRE/Train', train_mre, epoch)
            self.writer.add_scalar('MLP/MRE/Val', val_mre, epoch)
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs_mlp']} - "
                  f"Train Loss: {train_loss:.4f}, Train MRE: {train_mre:.2f}, "
                  f"Val Loss: {val_loss:.4f}, Val MRE: {val_mre:.2f}")
            
            # Save best model
            if val_mre < best_val_mre:
                best_val_mre = val_mre
                self._save_checkpoint(self.dual_model, optimizer, epoch, val_mre, "mlp_best")
            
            # Save latest model
            self._save_checkpoint(self.dual_model, optimizer, epoch, val_mre, "mlp_latest")
    
    def train_full_model(self, train_loader, val_loader):
        """End-to-end training of full dual-stream model"""
        print("=== Training Full Dual-Stream Model ===")
        self.current_stage = "full"
        
        # Load model with pretrained weights
        if self.dual_model is None:
            self.dual_model = DualStreamModel(num_landmarks=self.config['num_landmarks'], pretrained=True)
            self.dual_model = self.dual_model.to(self.device)
            
            # Load best MLP checkpoint
            checkpoint = torch.load(
                os.path.join(self.config['checkpoint_dir'], "mlp_best.pth"),
                map_location=self.device
            )
            self.dual_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Unfreeze all parameters
        self.dual_model.unfreeze_all_parameters()
        
        # Create optimizer and loss
        optimizer = optim.Adam(self.dual_model.parameters(), lr=self.config['learning_rate_finetune'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()
        
        best_val_mre = float('inf')
        
        for epoch in range(self.config['num_epochs_finetune']):
            # Training
            self.dual_model.train()
            train_loss = 0
            train_mre = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs_finetune']}")
            for batch in pbar:
                rgb_images = batch['rgb_image'].to(self.device)
                depth_maps = batch['depth_map'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.dual_model(rgb_images, depth_maps)
                final_coords = outputs['final_coords']
                
                # Calculate loss
                loss = criterion(final_coords, landmarks)
                
                loss.backward()
                optimizer.step()
                
                # Calculate MRE
                mre = calculate_mre(final_coords.detach(), landmarks.detach())
                
                train_loss += loss.item()
                train_mre += mre.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mre': mre.item()
                })
            
            train_loss /= len(train_loader)
            train_mre /= len(train_loader)
            
            # Validation
            val_loss, val_mre = self._validate_dual(val_loader, criterion)
            
            # Scheduler step
            scheduler.step(val_mre)
            
            # Log metrics
            self.writer.add_scalar('Full/Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Full/Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Full/MRE/Train', train_mre, epoch)
            self.writer.add_scalar('Full/MRE/Val', val_mre, epoch)
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs_finetune']} - "
                  f"Train Loss: {train_loss:.4f}, Train MRE: {train_mre:.2f}, "
                  f"Val Loss: {val_loss:.4f}, Val MRE: {val_mre:.2f}")
            
            # Save best model
            if val_mre < best_val_mre:
                best_val_mre = val_mre
                self._save_checkpoint(self.dual_model, optimizer, epoch, val_mre, "full_best")
            
            # Save latest model
            self._save_checkpoint(self.dual_model, optimizer, epoch, val_mre, "full_latest")
    
    def _validate(self, model, val_loader, criterion, mode):
        """
        Validate model
        
        Args:
            model: Model to validate
            val_loader: Validation DataLoader
            criterion: Loss function
            mode: 'rgb' or 'depth'
            
        Returns:
            val_loss, val_mre
        """
        model.eval()
        val_loss = 0
        val_mre = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if mode == 'rgb':
                    images = batch['rgb_image'].to(self.device)
                else:  # depth
                    images = batch['depth_map'].to(self.device)
                
                landmarks = batch['landmarks'].to(self.device)
                
                heatmaps, coords = model(images)
                
                # Calculate loss
                loss = criterion(coords, landmarks)
                
                # Calculate MRE
                mre = calculate_mre(coords, landmarks)
                
                val_loss += loss.item()
                val_mre += mre.item()
        
        val_loss /= len(val_loader)
        val_mre /= len(val_loader)
        
        return val_loss, val_mre
    
    def _validate_dual(self, val_loader, criterion):
        """
        Validate dual-stream model
        
        Args:
            val_loader: Validation DataLoader
            criterion: Loss function
            
        Returns:
            val_loss, val_mre
        """
        self.dual_model.eval()
        val_loss = 0
        val_mre = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb_images = batch['rgb_image'].to(self.device)
                depth_maps = batch['depth_map'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                # Forward pass
                outputs = self.dual_model(rgb_images, depth_maps)
                final_coords = outputs['final_coords']
                
                # Calculate loss
                loss = criterion(final_coords, landmarks)
                
                # Calculate MRE
                mre = calculate_mre(final_coords, landmarks)
                
                val_loss += loss.item()
                val_mre += mre.item()
        
        val_loss /= len(val_loader)
        val_mre /= len(val_loader)
        
        return val_loss, val_mre
    
    def _save_checkpoint(self, model, optimizer, epoch, val_mre, name):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f"{name}.pth")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mre': val_mre,
        }, checkpoint_path)
    
    def test(self, test_loader):
        """Test final model on test set"""
        print("=== Testing Model ===")
        
        # Load best model
        self.dual_model = DualStreamModel(num_landmarks=self.config['num_landmarks'], pretrained=False)
        self.dual_model = self.dual_model.to(self.device)
        
        checkpoint = torch.load(
            os.path.join(self.config['checkpoint_dir'], "full_best.pth"),
            map_location=self.device
        )
        self.dual_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.dual_model.eval()
        test_mre = 0
        
        landmark_mres = np.zeros(self.config['num_landmarks'])
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                rgb_images = batch['rgb_image'].to(self.device)
                depth_maps = batch['depth_map'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                # Forward pass
                outputs = self.dual_model(rgb_images, depth_maps)
                final_coords = outputs['final_coords']
                
                # Calculate MRE
                mre = calculate_mre(final_coords, landmarks)
                test_mre += mre.item() * batch['rgb_image'].size(0)
                
                # Calculate per-landmark MRE
                batch_size = rgb_images.size(0)
                total_samples += batch_size
                
                # Denormalize for landmark evaluation
                pred_coords = final_coords * 224
                target_coords = landmarks * 224
                
                # Calculate error for each landmark
                for i in range(self.config['num_landmarks']):
                    distances = torch.sqrt(torch.sum((pred_coords[:, i] - target_coords[:, i]) ** 2, dim=1))
                    landmark_mres[i] += torch.sum(distances).item()
        
        test_mre /= total_samples
        landmark_mres /= total_samples
        
        # Print results
        print(f"Test MRE: {test_mre:.2f}")
        
        # Print per-landmark MRE
        print("\nPer-landmark MRE:")
        landmark_names = [col[:-2] for col in self.config['landmark_cols'][::2]]
        for name, mre in zip(landmark_names, landmark_mres):
            print(f"{name}: {mre:.2f}")
        
        return test_mre 