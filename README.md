# Roboreg IITGN — Eye-to-Hand Camera Calibration

> **Eye-to-hand calibration from RGB / RGB-D images using the robot mesh as the calibration target.**  
> Designed for robotic arms (Franka FR3, xArm, KUKA LBR med7, etc.) with ROS 2 Humble.

[![License: Apache 2.0](https://img.shields.io/github/license/lbr-stack/roboreg)](https://github.com/lbr-stack/roboreg?tab=Apache-2.0-1-ov-file)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Version](https://img.shields.io/badge/version-0.4.6-blue)](https://github.com/lbr-stack/roboreg/releases/tag/0.4.6)
[![ROS 2](https://img.shields.io/badge/ROS2-Humble-orange)](https://docs.ros.org/en/humble/)

---

<!-- ============================================================
     BANNER IMAGE
     Recommended: A wide hero shot of the full robot setup —
     camera mounted above, robot arm in frame.
     Suggested size: 1200 × 400 px
     Replace the path below with your actual image file.
================================================================ -->

<!-- ![Setup Banner](doc/img/banner.png) -->

---

<!-- ============================================================
     RESULT COMPARISON IMAGES
     Show unregistered vs registered mesh + point cloud side by side.
     Suggested size: 600 × 400 px each
     Replace paths below with your actual image files.
================================================================ -->

| Unregistered | Registered |
|:---:|:---:|
| <!-- ![Unregistered](doc/img/unregistered.png) --> `[ doc/img/unregistered.png ]` | <!-- ![Registered](doc/img/registered.png) --> `[ doc/img/registered.png ]` |

---

<!-- ============================================================
     RENDER VERIFICATION IMAGE
     A sample overlay of the robot mesh rendered on a real image.
     Suggested size: 800 × 600 px
     Replace path below with your actual image file.
================================================================ -->

<!-- ![Render Verification](doc/img/render_verification.png) -->
**[ Placeholder: render verification overlay — doc/img/render_verification.png ]**

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Summary](#pipeline-summary)
- [Prerequisites](#prerequisites)
- [CUDA and PyTorch Setup](#cuda-and-pytorch-setup)
- [Installation](#installation)
- [Step 0 — Data Collection](#step-0--data-collection)
  - [0.1 Collect RGB Images](#01-collect-rgb-images)
  - [0.2 Collect Depth Data](#02-collect-depth-data)
  - [0.3 Collect Joint States](#03-collect-joint-states)
  - [0.4 Create Camera Info File](#04-create-camera-info-file)
- [Step 1 — Environment Setup](#step-1--environment-setup)
  - [1.1 Configure Bash Aliases](#11-configure-bash-aliases)
  - [1.2 Activate Environment & Source ROS 2](#12-activate-environment--source-ros-2)
- [Step 2 — Segmentation (rr-sam2)](#step-2--segmentation-rr-sam2)
- [Step 3 — Hydra Robust ICP (rr-hydra)](#step-3--hydra-robust-icp-rr-hydra)
- [Step 4 — Render & Verify (rr-render)](#step-4--render--verify-rr-render)
- [Directory Structure](#directory-structure)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Roboreg** solves the **eye-to-hand calibration** problem — determining the rigid 6-DOF transformation (Homogeneous Transformation matrix **HT**) between a fixed camera and a robot's base frame — without requiring a checkerboard or ArUco marker. Instead, it uses the **robot's own 3D mesh** as the calibration target.

| Before Registration | After Registration |
|---|---|
| Mesh and point cloud misaligned | Mesh overlaid precisely on point cloud |

The pipeline has three main stages:

```
RGB / RGB-D Images  ──►  [SAM2 Segmentation]  ──►  Robot Masks
                                                           │
Depth + Joint States ──►  [Hydra Robust ICP]  ──►  HT Matrix (camera extrinsics)
                                                           │
                          [rr-render]          ──►  Visual Verification Overlay
```

---

## Pipeline Summary

| Stage | Tool | Input | Output |
|---|---|---|---|
| Segmentation | `rr-sam2` | RGB images | Robot masks (`.png`) |
| Registration | `rr-hydra` | Masks + Depth + Joint states | `HT_hydra_robust.npy` |
| Verification | `rr-render` | Images + Joint states + HT | Overlay renders |

---

## Prerequisites

Before installation, ensure the following are available on your system:

| Requirement | Version | Notes |
|---|---|---|
| Ubuntu | 22.04 | Tested environment |
| Python | 3.10 | Required for conda path |
| ROS 2 | Humble | For robot URDF/xacro parsing |
| CUDA Toolkit | 11.x / 12.x | Required for differentiable rendering |
| NVIDIA GPU | Any CUDA-capable | SAM2 segmentation + rendering |
| Robot Description Package | e.g. `franka_description` | Must be on `$ROS_PACKAGE_PATH` |

> **Franka FR3 users:** Ensure `franka_ros2_ws` is built and sourced before running any `rr-*` commands.

---

## CUDA and PyTorch Setup

This section covers the full GPU stack setup using **NVIDIA driver 535**, **CUDA Toolkit 12.2**, and **PyTorch with CUDA 12.1** support. These versions are mutually compatible and tested on Ubuntu 22.04.

---

### 1. Install NVIDIA Driver 535

```bash
# Add the graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install driver 535
sudo apt install -y nvidia-driver-535

# Reboot to load the new driver
sudo reboot
```

After reboot, verify the driver is active:

```bash
nvidia-smi
```

The output should show driver version `535.x.x` and list your GPU(s). Note the `CUDA Version` field in the top-right — this reflects the maximum CUDA version your driver supports (should show 12.2 or higher).

---

### 2. Install CUDA Toolkit 12.2

Download and install the CUDA 12.2 toolkit using the official network installer:

```bash
# Download the CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit 12.2
sudo apt install -y cuda-toolkit-12-2
```

Add CUDA binaries and libraries to your shell environment:

```bash
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify the installation:

```bash
nvcc --version
```

Expected output:

```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.2, V12.2.x
```

---

### 3. Install PyTorch with CUDA 12.1

PyTorch releases CUDA-specific wheels. Install the CUDA 12.1 build of PyTorch (fully compatible with CUDA Toolkit 12.2 due to backward compatibility):

```bash
# Activate your virtual environment first
source roboreg/bin/activate

# Install PyTorch 2.x with CUDA 12.1 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch can detect the GPU:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

Expected output:

```
PyTorch: 2.x.x+cu121
CUDA available: True
Device: NVIDIA GeForce RTX XXXX  (or your specific GPU model)
```

> **Note:** PyTorch CUDA 12.1 wheels are compatible with CUDA Toolkit 12.2 installed on the host. The minor version mismatch is expected and supported.

---

## Installation

Install roboreg into a Python virtual environment after completing the CUDA and PyTorch setup above.

```bash
# Create a Python virtual environment
python3 -m venv roboreg
source roboreg/bin/activate

# Install roboreg
pip3 install roboreg
```

> **Note:** CUDA Toolkit must be installed and `nvcc` must be accessible on `$PATH` before installing roboreg, as some dependencies compile CUDA extensions on install. See [CUDA Toolkit Install Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for reference.

---

## Step 0 — Data Collection

Before running any calibration, collect **50 samples** of the robot at different joint configurations. Each sample consists of three synchronized files.

> **Convention (Franka FR3 example):**
> - Images: `franka_image_<N>.png`
> - Depth: `franka_depth_<N>.npy`
> - Joint states: `franka_joints_<N>.npy`

All files for a given camera should be stored in the **same directory**, e.g.:
```
/home/<user>/<workspace>/Roboreg_test/franka_top/
```

---

### 0.1 Collect RGB Images

Use the provided helper script to save PNG images from the camera topic:

```bash
python3 png_save.py
```

This saves images named `franka_image_0.png`, `franka_image_1.png`, ... `franka_image_49.png` to the output directory.

---

### 0.2 Collect Depth Data

Use the depth saving script to record aligned depth frames as NumPy arrays:

```bash
python3 depth_2_npy.py
```

This saves `franka_depth_0.npy` through `franka_depth_49.npy`.

---

### 0.3 Collect Joint States

Record the robot's joint angles for each configuration:

```bash
python3 joint_state_2_npy.py
```

This saves `franka_joints_0.npy` through `franka_joints_49.npy`.

> **Important:** Ensure the robot is **stationary** at each configuration when you capture the triplet (image + depth + joints). Synchronization errors will degrade calibration accuracy.

---

### 0.4 Create Camera Info File

The `camera_info` YAML file contains the camera's intrinsic parameters (focal length, principal point, distortion coefficients, image dimensions). Roboreg requires this file for both the Hydra ICP and render steps.

With the camera driver running and publishing ROS 2 topics, print the camera info message directly to the terminal:

```bash
ros2 topic echo /camera/color/camera_info --once
```

> **Note:** Replace `/camera/color/camera_info` with the actual camera info topic for your setup. To list all available topics and find the correct one, run:
> ```bash
> ros2 topic list | grep camera_info
> ```

The terminal will print a YAML-formatted message similar to the following:

```yaml
header:
  stamp:
    sec: 1700000000
    nanosec: 0
  frame_id: camera_color_optical_frame
height: 720
width: 1280
distortion_model: plumb_bob
d:
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
k:
- 910.4
- 0.0
- 640.0
- 0.0
- 910.4
- 360.0
- 0.0
- 0.0
- 1.0
r:
- 1.0
- 0.0
- 0.0
- 0.0
- 1.0
- 0.0
- 0.0
- 0.0
- 1.0
p:
- 910.4
- 0.0
- 640.0
- 0.0
- 0.0
- 910.4
- 360.0
- 0.0
- 0.0
- 0.0
- 1.0
- 0.0
binning_x: 0
binning_y: 0
roi:
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: false
```

Copy the entire output, open a new YAML file in the camera info directory, and paste it in:

```bash
# Create the camera_info directory if it does not exist
mkdir -p /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/camera_info

# Open a new file and paste the copied output
nano /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/camera_info/camera_top_info.yaml
```

Paste the content, then save and close with `Ctrl+O`, `Enter`, `Ctrl+X`.

To redirect the output directly to a file without manual copy-paste, you can also use:

```bash
ros2 topic echo /camera/color/camera_info --once > \
    /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/camera_info/camera_top_info.yaml
```

> **Camera Resolution Change Warning:** If the camera resolution is changed at any point (e.g., switching from 1280×720 to 640×480, or swapping to a side camera), you must capture a new `camera_info` message at the new resolution and save it as a separate file. Update the `--camera-info-file` argument in both `rr_hydra` and `rr_render` accordingly.

---

## Step 1 — Environment Setup

### 1.1 Configure Bash Aliases

Add the following convenience functions to your `~/.bashrc` to avoid typing long CLI commands each time. Adjust all paths to match your actual workspace layout.

```bash
nano ~/.bashrc
```

Paste the following block at the end of the file:

```bash
# ─── ROBOREG HELPERS ──────────────────────────────────────────────────────────

## Segmentation (SAM2)
rr_sam2() {
    rr-sam2 \
        --path /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/franka_top \
        --pattern "franka_image_*.png" \
        --n-positive-samples 20 \
        --n-negative-samples 20 \
        --device cuda
}

## Hydra Robust ICP
rr_hydra() {
    rr-hydra \
        --camera-info-file /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/camera_info/camera_top_info.yaml \
        --path /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/franka_top \
        --mask-pattern mask_sam2_franka_image_*.png \
        --depth-pattern franka_depth_*.npy \
        --joint-states-pattern franka_joints_*.npy \
        --ros-package franka_description \
        --xacro-path robots/fr3/fr3_src.urdf.xacro \
        --root-link-name fr3_link0 \
        --end-link-name fr3_hand \
        --number-of-points 20000 \
        --output-file HT_hydra_robust.npy
}

## Render & Verify
rr_render() {
    rr-render \
        --batch-size 1 \
        --num-workers 0 \
        --ros-package franka_description \
        --xacro-path robots/fr3/fr3_src.urdf.xacro \
        --root-link-name fr3_link0 \
        --end-link-name fr3_hand \
        --camera-info-file /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/camera_info/camera_top_info.yaml \
        --extrinsics-file /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/franka_top/HT_hydra_robust.npy \
        --images-path /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/franka_top \
        --joint-states-path /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/franka_top \
        --image-pattern franka_image_*.png \
        --joint-states-pattern franka_joints_*.npy \
        --output-path /home/iitgn-robotics/deepak_ws/Roboreg-IITGN/Roboreg_test/franka_top
}
```

Save and close the file, then reload your shell:

```bash
source ~/.bashrc
```

> **Note:** If the camera resolution changes (e.g., when switching from a top camera to a side camera), update the `--camera-info-file` path to the correct `.yaml` accordingly.

---

### 1.2 Activate Environment & Source ROS 2

Run these commands every time you open a new terminal session before using any `rr-*` tools:

```bash
# If inside a conda environment, deactivate it first
conda deactivate

# Activate the Python virtual environment
source roboreg/bin/activate

# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Source your robot's ROS 2 workspace:
# For Franka FR3:
source /home/iitgn-robotics/Debojit_WS/franka_ros2_ws/install/setup.bash

# For Heal / Cobot:
# source /home/iitgn-robotics/Debojit_WS/cobot_ws/install/setup.bash
```

---

## Step 2 — Segmentation (rr-sam2)

The segmentation step generates a **binary mask** for each image, isolating the robot from the background. You will be shown images one by one and prompted to click:

- **Positive samples** (clicks ON the robot body) — `--n-positive-samples 20`
- **Negative samples** (clicks OFF the robot / on background) — `--n-negative-samples 20`

SAM2 uses these prompts to generate high-quality segmentation masks.

```bash
rr_sam2
```

**What this runs under the hood:**
```bash
rr-sam2 \
    --path /home/.../franka_top \
    --pattern "franka_image_*.png" \
    --n-positive-samples 20 \
    --n-negative-samples 20 \
    --device cuda
```

**Output:** One mask file per image, saved as `mask_sam2_franka_image_<N>.png` in the same directory.

> **Tip:** Be precise with your positive clicks — click clearly on the robot arm links. Place negative clicks on the table, background, and any non-robot objects.

---

## Step 3 — Hydra Robust ICP (rr-hydra)

The Hydra algorithm performs **point-to-plane ICP registration on a Lie algebra**, matching the robot's 3D mesh (sampled as a point cloud from URDF/xacro) against the masked depth point cloud from all 50 images. This step does **not** require GPU and can run on CPU.

```bash
rr_hydra
```

**What this runs under the hood:**
```bash
rr-hydra \
    --camera-info-file .../camera_top_info.yaml \
    --path .../franka_top \
    --mask-pattern mask_sam2_franka_image_*.png \
    --depth-pattern franka_depth_*.npy \
    --joint-states-pattern franka_joints_*.npy \
    --ros-package franka_description \
    --xacro-path robots/fr3/fr3_src.urdf.xacro \
    --root-link-name fr3_link0 \
    --end-link-name fr3_hand \
    --number-of-points 20000 \
    --output-file HT_hydra_robust.npy
```

**Output:** `HT_hydra_robust.npy` — a 4×4 Homogeneous Transformation matrix representing the **camera-to-robot-base extrinsics**.

> **Note:** The `--root-link-name` and `--end-link-name` define the kinematic chain used to compute the robot's forward kinematics at each captured configuration.

---

## Step 4 — Render & Verify (rr-render)

Use the computed extrinsics to **overlay the robot's 3D mesh on the original RGB images**. This is a visual sanity check — if the robot model aligns well with the robot in the images, the calibration is correct.

```bash
rr_render
```

**What this runs under the hood:**
```bash
rr-render \
    --batch-size 1 \
    --num-workers 0 \
    --ros-package franka_description \
    --xacro-path robots/fr3/fr3_src.urdf.xacro \
    --root-link-name fr3_link0 \
    --end-link-name fr3_hand \
    --camera-info-file .../camera_top_info.yaml \
    --extrinsics-file .../franka_top/HT_hydra_robust.npy \
    --images-path .../franka_top \
    --joint-states-path .../franka_top \
    --image-pattern franka_image_*.png \
    --joint-states-pattern franka_joints_*.npy \
    --output-path .../franka_top
```

**Output:** Rendered overlay images saved to the output path. Open them to visually confirm alignment.

> **On first run**, `nvdiffrast` will compile PyTorch extensions — this may take several minutes and use significant RAM (16 GB+ recommended).  
> If you hit memory issues, limit parallel compilation:
> ```bash
> export MAX_JOBS=1
> rr_render
> ```

---

## Directory Structure

After completing the full pipeline, your working directory should look like this:

```
Roboreg_test/
├── camera_info/
│   ├── camera_top_info.yaml          # Camera intrinsics (top camera)
│   └── camera_side_info.yaml         # Camera intrinsics (side camera, if used)
│
└── franka_top/
    ├── franka_image_0.png            # RGB images (50 total)
    ├── franka_image_1.png
    │   ...
    ├── franka_depth_0.npy            # Depth arrays (50 total)
    │   ...
    ├── franka_joints_0.npy           # Joint state arrays (50 total)
    │   ...
    ├── mask_sam2_franka_image_0.png  # SAM2 masks — generated by rr-sam2
    │   ...
    ├── HT_hydra_robust.npy           # 4×4 HT matrix — generated by rr-hydra
    └── render_franka_image_0.png     # Overlay renders — generated by rr-render
        ...
```

---

## CLI Reference

| Command | Description |
|---|---|
| `rr-sam2` | Segment robot from RGB images using SAM2 |
| `rr-hydra` | Robust point-to-plane ICP on Lie algebra (CPU/GPU) |
| `rr-render` | Render robot mesh overlay on images for verification (GPU) |

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `nvdiffrast` compilation hangs / OOM | Too many parallel jobs | `export MAX_JOBS=1` before running |
| `rr-hydra` produces large registration error | Poor segmentation masks | Re-run `rr-sam2` with more careful click placement |
| Robot mesh not found / xacro error | ROS workspace not sourced | Re-run `source .../install/setup.bash` |
| `franka_description` not found | Wrong `--ros-package` or workspace not built | Check `ros2 pkg list \| grep franka` |
| Render overlay is offset | Wrong camera info file (resolution mismatch) | Update `--camera-info-file` to match current camera resolution |
| `nvcc: command not found` | CUDA not on PATH | Re-run `source ~/.bashrc` or verify `/usr/local/cuda-12.2/bin` is on `$PATH` |
| `torch.cuda.is_available()` returns False | PyTorch CPU-only build installed | Reinstall using the `--index-url https://download.pytorch.org/whl/cu121` wheel |

> **Camera Resolution Change Warning:** If you switch camera resolutions or use a different camera (e.g., side vs. top), you **must** update the `--camera-info-file` argument in both `rr_hydra` and `rr_render` to the matching `.yaml` file.

---

## Acknowledgements

Roboreg IITGN builds on the original [lbr-stack/roboreg](https://github.com/lbr-stack/roboreg) project. We gratefully acknowledge the following organizations and funding sources that supported the development of the upstream codebase.

### Organizations and Grants

We would further like to acknowledge the following supporters:

| Logo | Notes |
|:---:|:---|
| [![Wellcome](https://img.shields.io/badge/Wellcome%2FEPSRC-Supporter-blue)](https://wellcome.org/) | This work was supported by core and project funding from the Wellcome/EPSRC [WT203148/Z/16/Z; NS/A000049/1; WT101957; NS/A000027/1]. |
| [![EU Flag](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/320px-Flag_of_Europe.svg.png)](https://ec.europa.eu/) | This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101016985 (FAROS project). |
| [![RViMLab](https://img.shields.io/badge/RViMLab-Built%20Here-lightgrey)](https://rvim.online/) | Built at [RViMLab](https://rvim.online/). |
| [![CAI4CAI](https://img.shields.io/badge/CAI4CAI-Built%20Here-lightgrey)](https://cai4cai.ml/) | Built at [CAI4CAI](https://cai4cai.ml/). |
| [![King's College London](https://img.shields.io/badge/King's%20College%20London-Institution-red)](https://www.kcl.ac.uk/) | Built at [King's College London](https://www.kcl.ac.uk/). |

---

*For issues, open a ticket on the [GitHub Issues page](https://github.com/lbr-stack/roboreg/issues).*  
*Licensed under [Apache 2.0](https://github.com/lbr-stack/roboreg/blob/main/LICENSE).*