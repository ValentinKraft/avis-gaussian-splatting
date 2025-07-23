
# 📘 instructions.md  
**Project Extension: Direct Volume-Based Optimization for 3D Gaussian Splatting**

---

## 🔥 Motivation

The original [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting) method optimizes splats to reconstruct RGB images from multiple views. However, in medical imaging, we often already have **segmentation volumes** (e.g., CT, MRI, or AI-generated masks). This extension aims to optimize the 3D splats **directly on volumetric data** instead of images.

---

## 🎯 Goal

Enable **differentiable volumetric optimization** of Gaussian splats using ground truth **3D segmentations** (binary or float-valued masks) as the target.  

This allows training Gaussian fields to represent **organs, vessels, or other anatomical structures** directly, without rendering to 2D images.

---

## 🏗️ Required Modifications (Copilot Tasks)

### 1. Add Volume Supervision Loss  
Implement a loss function that compares the voxelized splats (`V_hat`) against a segmentation volume (`V`) using one or more of the following:

- ✅ `MSELoss`
- ✅ `DiceLoss`
- ✅ `TverskyLoss` *(optional for vessels)*
- ✅ `KL-Divergence` *(for soft masks)*

➡️ Implement a new file: `gaussian_splatting/losses/volume_loss.py`  
➡️ Create a `VolumeLoss` class with options for loss types and weights.

---

### 2. Splat Rasterization to Volume (Differentiable)  
Add a function that **accumulates all splats into a 3D voxel grid**:

```python
volume_hat = splat_to_volume(splats, volume_shape=(D, H, W))
```

Each splat contributes to nearby voxels via a 3D Gaussian kernel:

```python
gaussian = alpha * exp(-||x - μ||² / (2σ²))
```

➡️ Create this in `gaussian_splatting/utils/splat_to_volume.py`

---

### 3. Integrate Volume Supervision into Training Loop  

In `trainer.py` or `train_step()`:

- Call the voxelization method after updating splats
- Compare it to the reference segmentation mask (`volume_gt`)
- Backpropagate the loss

```python
volume_hat = splat_to_volume(gaussians, volume_shape)
loss_vol = volume_loss(volume_hat, volume_gt)
loss_total = loss_rgb + λ * loss_vol
loss_total.backward()
```

➡️ Add CLI argument or config option for enabling volume supervision  
➡️ Allow training **without** any RGB supervision (image-free mode)

---

### 4. Volume DataLoader Support  
Extend the dataloader to:

- Load 3D volumes or segmentation masks (e.g., `.nii`, `.npy`, or `.mhd`)
- Resample to training resolution (e.g., 64×64×64)
- Align with splat coordinate system

➡️ Add a folder like `data/volumes/`  
➡️ Load volumes in `gaussian_splatting/data/volume_loader.py`

---

## 🧪 Testing Instructions

- Start with synthetic or toy volumes (e.g., spheres, vessel masks)
- Visualize splats and reconstructed voxel volumes (e.g. via `matplotlib`, `napari`, or `vtk`)
- Compare reconstruction with volume ground truth

---

## 📁 Suggested File Structure

```
gaussian_splatting/
├── losses/
│   └── volume_loss.py      # Dice, MSE, Tversky...
├── utils/
│   └── splat_to_volume.py  # Rasterization of Gaussians to voxel grid
├── data/
│   └── volume_loader.py    # Loading and preprocessing volumes
├── trainer.py              # Integrate loss + forward pass
```

---

## 💡 Notes for Copilot

- Use PyTorch with CUDA tensors for volume operations.
- Keep all operations differentiable.
- Allow toggling between RGB loss and volume loss (or both).
- Add appropriate CLI/config arguments (e.g., `--volume-supervision`).
- Match splat scale and voxel resolution (normalize to [0, D] range).

---

## 🧠 Related Keywords (for Copilot prompt enhancement)

```txt
differentiable voxel grid, 3d gaussian accumulation, segmentation supervision, dice loss, vessel mask loss, 3D representation learning, medical image optimization
```

---

## ✅ Example Prompt for Copilot Agent

> "Extend `trainer.py` to add a loss between voxelized splats and a segmentation volume. Use DiceLoss, and accumulate the splats into a 64×64×64 grid using splat_to_volume(). Add CLI flag `--volume-supervision`."
