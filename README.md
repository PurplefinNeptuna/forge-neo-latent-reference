## Forge Neo Latent Reference

A forge-neo extension that adds latent reference input for the Anima DiT model. Reference images are encoded through the VAE and concatenated along the time dimension during the model's forward pass.

## How it works

1. Upload one or more reference images
2. The extension resizes them according to your settings, then encodes them into latents via the model's VAE
3. Generation resolution is automatically overridden to match the resized reference image dimensions
4. During each denoising step, reference latents are concatenated along the time axis (dim=2) before the forward pass, then the output is cropped back to the original length

## UI controls

- **Reference Images**: Upload one or more images to use as references
- **Resize to ~N px**: Target pixel size. Reference images larger than this will be scaled down proportionally to approach this value while preserving aspect ratio
- **Size multiple**: After resizing, width and height are rounded to the nearest multiple of this value (default 8, matching Anima's patch alignment requirement)
- **Resize interpolation method**: The algorithm used when resizing reference images:
  - `bilinear`: Fast, smooth results — good general choice
  - `bicubic`: Sharper than bilinear, slightly slower
  - `nearest`: No smoothing — preserves pixel art / hard edges
  - `area`: Best for downscaling, avoids moiré artifacts
  - `lanczos`: Highest quality resizing, computationally expensive

## Credits

- Inspired by [ComfyUI-Cosmos-Reference](https://github.com/Mirumo0u0/ComfyUI-Cosmos-Reference) for ComfyUI
- Created using Qwen3.6 27B
