# Cephalometric Landmark Detection

This project implements a dual-stream model for cephalometric landmark detection using both RGB images and depth maps. The model architecture consists of parallel HRNet backbones for RGB and depth inputs, which generate separate landmark predictions that are then fused using a refinement MLP.

## Dataset

The dataset contains lateral photographs of patients along with their corresponding depth maps and 19 cephalometric landmarks. The data is stored in `data/train_data_pure_depth.pkl`.

## Model Architecture

The model has the following components:

1. **RGB Stream**: Processes RGB images using an HRNet-W32 backbone.
2. **Depth Stream**: Processes depth maps using an HRNet-W32 backbone (with modified input layer).
3. **Refinement MLP**: Fuses predictions from both streams to generate final landmark coordinates.

## Training Strategy

The training process is divided into three phases:

1. **Stream Training**: Each stream (RGB and Depth) is trained separately.
2. **MLP Training**: After the streams are trained, they are frozen, and the Refinement MLP is trained.
3. **Full Model Training**: Finally, all components are unfrozen and fine-tuned together.

## Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

## Usage

To run the full training pipeline:

```bash
python train.py
```

This will automatically run all training stages in sequence.

### Training Specific Stages

To train specific stages individually:

```bash
# Train RGB stream only
python train.py --train_rgb

# Train Depth stream only
python train.py --train_depth

# Train MLP (requires trained RGB and Depth streams)
python train.py --train_mlp

# Fine-tune the full model (requires trained MLP)
python train.py --train_full

# Test the model (requires trained full model)
python train.py --test
```

### Configuration Options

You can customize the training process using the following options:

```bash
# Change batch size
python train.py --batch_size 8

# Change number of epochs for different training stages
python train.py --num_epochs 30 --num_epochs_mlp 15 --num_epochs_finetune 20

# Change learning rates
python train.py --learning_rate 0.0001 --learning_rate_finetune 0.00001

# Specify output directories
python train.py --checkpoint_dir ./checkpoints --log_dir ./logs
```

## Evaluation

The model is evaluated using Mean Radial Error (MRE), which measures the average Euclidean distance between predicted and ground truth landmarks in pixels.

## Files

- `train.py`: Main script for training the model
- `models.py`: Model architecture definitions
- `trainer.py`: Training and testing logic
- `utils.py`: Utility functions for data processing and evaluation 