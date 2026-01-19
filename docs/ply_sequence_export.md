# PLY Sequence Export

This feature allows you to save the Gaussian Splatting model as PLY files at specified iterations during training. This is useful for visualizing the training progress and creating animations of the model evolution.

## Usage

The following command line options control PLY export:

```bash
python train.py ... --save_ply_every 10 --ply_output_prefix "my_model"
```

Options:

- `--save_ply_every N`: Save a PLY file every N iterations (default: 1)
- `--ply_output_prefix NAME`: Set the prefix for the PLY filenames (default: "gaussians")

PLY files are saved in the `ply_sequence` subdirectory of your model output directory.

## Creating Animations

You can use the provided utility script to create animations from the PLY sequence:

```bash
python utils/create_ply_animation.py --input_dir output/your_model_id/ply_sequence --output animation.gif
```

Options:

- `--input_dir`: Directory containing the PLY files
- `--pattern`: File pattern to match (default: "gaussians_*.ply")
- `--output`: Output animation path (.gif or .mp4)
- `--fps`: Frames per second (default: 30)
- `--point_size`: Base point size multiplier (default: 5.0)
- `--camera_zoom`: Camera zoom factor (default: 1.5)
- `--rotation_speed`: Camera rotation speed (default: 0.5)
- `--background`: Background color (default: "white")
- `--resolution`: Output resolution (default: 1920 1080)

### Requirements

To create animations, you need to install:

```bash
pip install pyvista tqdm numpy
```

If you want to create MP4 videos, you also need ffmpeg installed.
