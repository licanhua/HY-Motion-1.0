# FBX Export Guide

This guide explains how to export generated motion to FBX files using the standalone wooden boy exporter.

## Overview

The FBX export functionality has been extracted from the ComfyUI nodes and made available as a standalone module. It supports:
- **Wooden Boy Export**: Export to the default wooden boy character template
- **Batch Processing**: Export multiple motions at once
- **NPZ to FBX Conversion**: Convert saved motion NPZ files to FBX

## Files

### 1. `hymotion/utils/fbx_export.py`
Standalone FBX export module that works independently of ComfyUI.

**Key Functions:**
- `export_motion_to_fbx(motion_data, output_path, text_description, template_fbx)` - Export single motion
- `export_motion_batch_to_fbx(motion_data_list, output_dir, ...)` - Export multiple motions
- `load_motion_from_npz(npz_path)` - Load motion from NPZ file

### 2. `test_fbx_export.py`
Test script to verify FBX export functionality.

### 3. Updated `gradio_app.py`
Integrated FBX export function: `export_wooden_boy_fbx(smpl_data_list, output_dir, text)`

## Usage Examples

### 1. Command Line (Direct Export)

Export an NPZ file to FBX:

```bash
python hymotion/utils/fbx_export.py --input output/gradio/motion_xxx_000.npz --output output/test.fbx
```

### 2. Test Script (Single File)

Test export with a motion NPZ file:

```bash
python test_fbx_export.py --motion_npz output/gradio/motion_xxx_000.npz --output_dir output/test_fbx
```

### 3. Test Script (Batch Mode)

Export all NPZ files in a directory:

```bash
python test_fbx_export.py --motion_npz_dir output/gradio --output_dir output/test_fbx
```

### 4. Python API

Use in your Python code:

```python
from hymotion.utils.fbx_export import export_motion_to_fbx, load_motion_from_npz

# Load motion from NPZ
motion_data = load_motion_from_npz("output/gradio/motion_xxx_000.npz")

# Export to FBX
success = export_motion_to_fbx(
    motion_data=motion_data,
    output_path="output/my_animation.fbx",
    text_description="A person walking forward"
)
```

### 5. From Gradio App

The function `export_wooden_boy_fbx()` is integrated into gradio_app.py and can be called after motion generation:

```python
from gradio_app import export_wooden_boy_fbx

# smpl_data_list is the motion output from generate_motion
fbx_files = export_wooden_boy_fbx(
    smpl_data_list=smpl_data_list,
    output_dir="output/fbx",
    text="A person walking"
)
```

## Input Data Format

The motion data should be a dictionary containing:

**Required Keys:**
- `rot6d`: Rotation data in 6D format, shape `(num_frames, 22, 6)`
- `transl`: Translation data, shape `(num_frames, 3)`

**Optional Keys:**
- `text`: Text description of the motion
- `duration`: Duration in seconds
- `seed`: Random seed used for generation
- `keypoints3d`: 3D keypoints (not used for FBX export)
- `root_rotations_mat`: Root rotation matrices (not used for FBX export)

## Workflow Integration

### Typical Workflow:

1. **Generate Motion** (using Gradio or local_infer.py)
   ```bash
   python gradio_app.py
   # Or
   python local_infer.py --text "a person walks forward" --seeds 42
   ```

2. **Motion is Saved as NPZ** (in `output/gradio/` or specified directory)

3. **Export to FBX** (using test script or API)
   ```bash
   python test_fbx_export.py --motion_npz output/gradio/motion_xxx_000.npz --output_dir output/fbx
   ```

4. **Import FBX into 3D Software** (Blender, Maya, Unreal Engine, etc.)

## Troubleshooting

### Issue: "FBX SDK not installed"
**Solution:** Install the FBX Python SDK:
- Download from Autodesk's website
- Or install via pip if available: `pip install fbx`

### Issue: "Template FBX not found"
**Solution:** Ensure the wooden boy template exists at:
```
assets/wooden_models/boy_Rigging_smplx_tex.fbx
```

### Issue: "Import errors"
**Solution:** Make sure you're running from the project root directory:
```bash
cd /path/to/HY-Motion-1.0
python test_fbx_export.py ...
```

## Advanced: Custom FBX Skeletons

For custom FBX skeletons (Mixamo, etc.), use the existing `retarget_fbx.py` module:

```bash
python hymotion/utils/retarget_fbx.py \
    --source output/gradio/motion_xxx_000.npz \
    --target input/custom_character.fbx \
    --output output/retargeted.fbx \
    --yaw 0.0 \
    --scale 0.0
```

## Notes

- The wooden boy export uses the SMPL-H to wooden FBX converter (`smplh2woodfbx.py`)
- Output FBX files include:
  - `.fbx` - The animation file
  - `.txt` - Text description (if provided)
- Frame rate is fixed at 30 FPS
- The template character is a simple wooden mannequin suitable for animation preview

## Example Filenames

After running the test script, you'll get files like:
```
output/test_fbx/
├── motion_20260111_143052_a7f3b1d2_000_export.fbx
├── motion_20260111_143052_a7f3b1d2_000_export.txt
└── ...
```

## Integration with Gradio

The FBX export is automatically available in the Gradio interface:
1. Generate motion using text prompt
2. Motion is saved as NPZ in `output/gradio/`
3. (Optional) Call `export_wooden_boy_fbx()` to also generate FBX files
4. Download FBX files from the interface

The retargeting to custom FBX skeletons is handled separately using the existing `retarget_util.py` module in the Gradio app.
