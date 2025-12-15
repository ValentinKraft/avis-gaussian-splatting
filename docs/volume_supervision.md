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
python train.py \
  --model_path output/my-volume-model \
  --mask_path path/to/mask.nii.gz \
  --volume_path path/to/volume.nii.gz
```

### CLI Parameters

#### Volume Supervision Options
- `--volume_path`: Path to ground truth volume file (.nii, .npy, .mhd) **(required)**
- `--volume_loss_type`: Type of volume supervision loss (choices: "mse", "dice", "tversky", "kl"; default: "dice")
- `--volume_loss_weight`: Weight for volume supervision loss (default: 1.0)
- `--volume_shape`: Target shape for volume supervision as three integers (default: 64 64 64)

#### Volume Initialization Options
- `--mask_path`: Path to segmentation mask file (.nii, .npy, .mhd) **(required)**
- `--init_n_points`: Number of Gaussian points to sample (default: 5000)
- `--position_noise`: Standard deviation for position noise (default: 0.01)
- `--volume_transform`: Path to 4x4 transform matrix for volume alignment (.npy)

#### PLY Export Options
- `--save_ply_every`: Save PLY file every N iterations (default: 1)
- `--ply_output_prefix`: Prefix for PLY filenames (default: "gaussians")

#### Medical Presets
- `--medical_mode {organ,vessel}`: Applies a simplified preset for smooth medical volumes. `organ` (default) boosts initial point counts and disables densification, while `vessel` keeps a gentle densification window tailored to thin structures.
- `--enable_diversity`: Re-enables the diversity warmup, scale constraints, and related regularizers when you need advanced control beyond the preset.
- `--enable_diagnostics`: Restores verbose monitoring (parameter plots, gradient norms, TensorBoard scalars) for deep debugging runs.

## Examples

### Organ preset (default)
```bash
python train.py \
  --model_path output/ct-scan \
  --mask_path data/vessel-mask.nii.gz \
  --volume_path data/ct-volume.nii.gz \
  --medical_mode organ \
  --iterations 4000
```

### Vessel preset with diagnostics enabled
```bash
python train.py \
  --model_path output/mri-scan \
  --mask_path data/brain-mask.nii.gz \
  --volume_path data/mri-volume.nii.gz \
  --medical_mode vessel \
  --enable_diagnostics \
  --iterations 6000
```

## Visualization

After training, the model will be saved as PLY files in the `{model_path}/ply_sequence` directory. You can use the included visualization utility to create animations:

```bash
python utils/create_ply_animation.py \
  --input_dir output/my-volume-model/ply_sequence \
  --output_path my_animation.mp4 \
  --fps 30
```
