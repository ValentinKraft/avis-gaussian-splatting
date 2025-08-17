# Volume-Based Gaussian Splatting

This extension to the original 3D Gaussian Splatting framework enables training directly from volumetric data such as CT scans, MRIs, or segmentation masks.

## Key Features

- **Volume Data Training**: Train directly from 3D volumetric data instead of 2D images
- **Intensity-Based Rendering**: Gaussian splats carry intensity values from the original volume
- **Mask-Based Initialization**: Initialize Gaussian positions by sampling from binary or continuous masks
- **Interactive Visualization**: Export models as PLY files for real-time viewing

## Usage

### Basic Training Command

```bash
python train.py --volume_supervision \
  --model_path output/my-volume-model \
  --init_from_mask \
  --mask_path path/to/mask.nii.gz \
  --volume_path path/to/volume.nii.gz
```

### CLI Parameters

#### Volume Supervision Options
- `--volume_supervision`: Enable volume supervision loss (required for volume-based training)
- `--volume_path`: Path to ground truth volume file (.nii, .npy, .mhd)
- `--volume_loss_type`: Type of volume supervision loss (choices: "mse", "dice", "tversky", "kl"; default: "dice")
- `--volume_loss_weight`: Weight for volume supervision loss (default: 1.0)
- `--volume_shape`: Target shape for volume supervision as three integers (default: 64 64 64)

#### Volume Initialization Options
- `--init_from_mask`: Initialize Gaussian points by sampling from segmentation mask
- `--mask_path`: Path to segmentation mask file (.nii, .npy, .mhd)
- `--init_n_points`: Number of Gaussian points to sample (default: 5000)
- `--position_noise`: Standard deviation for position noise (default: 0.01)
- `--volume_transform`: Path to 4x4 transform matrix for volume alignment (.npy)

#### PLY Export Options
- `--save_ply_every`: Save PLY file every N iterations (default: 1)
- `--ply_output_prefix`: Prefix for PLY filenames (default: "gaussians")

## Examples

### Training with 10,000 points from a CT scan
```bash
python train.py --volume_supervision \
  --model_path output/ct-scan \
  --init_from_mask \
  --mask_path data/vessel-mask.nii.gz \
  --volume_path data/ct-volume.nii.gz \
  --init_n_points 10000 \
  --save_ply_every 10 \
  --iterations 5000
```

### Using MSE loss with 2000 points
```bash
python train.py --volume_supervision \
  --model_path output/mri-scan \
  --init_from_mask \
  --mask_path data/brain-mask.nii.gz \
  --volume_path data/mri-volume.nii.gz \
  --volume_loss_type mse \
  --init_n_points 2000 \
  --save_ply_every 100 \
  --iterations 10000
```

## Visualization

After training, the model will be saved as PLY files in the `{model_path}/ply_sequence` directory. You can use the included visualization utility to create animations:

```bash
python utils/create_ply_animation.py \
  --input_dir output/my-volume-model/ply_sequence \
  --output_path my_animation.mp4 \
  --fps 30
```
