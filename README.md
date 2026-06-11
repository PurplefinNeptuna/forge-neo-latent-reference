## Forge Neo Latent Reference

A forge-neo extension that adds latent reference input for the Anima DiT model. Reference image are encoded through the VAE and concatenated along the time dimension during the model's forward pass. Currently only works for single reference image.

## How it works

1. Upload reference image
2. The extension resizes them according to your settings, then encodes them into latents via the model's VAE
3. Generation resolution is automatically overridden to match the resized reference image dimensions
4. During each denoising step, reference latent are concatenated along the time axis (dim=2) before the forward pass, then the output is cropped back to the original length

## UI controls

- **Reference Image**: Upload one image to use as reference
- **Resize to ~N px**: Target pixel size. Reference image larger than this will be scaled down proportionally to approach this value while preserving aspect ratio
- **Size multiple**: After resizing, width and height are rounded to the nearest multiple of this value (default 8, matching Anima's patch alignment requirement)
- **Resize interpolation method**: The algorithm used when resizing reference :
  - `bilinear`: Fast, smooth results — good general choice
  - `bicubic`: Sharper than bilinear, slightly slower
  - `nearest`: No smoothing — preserves pixel art / hard edges
  - `area`: Best for downscaling, avoids moiré artifacts
  - `lanczos`: Highest quality resizing, computationally expensive
- **Background color**: For transparent image, this color fills the background behind the foreground image (default `#000000`)

## Credits

- Inspired by [ComfyUI-Cosmos-Reference](https://github.com/Mirumo0u0/ComfyUI-Cosmos-Reference) for ComfyUI
- Created using Qwen3.6 27B
