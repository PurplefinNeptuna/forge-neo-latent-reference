import gradio as gr
import torch
import numpy as np
import math

from PIL import Image

from modules import scripts, shared, devices
from modules.ui_components import InputAccordion


class LatentReferenceScript(scripts.Script):
    def __init__(self):
        self.ref_latents = []

    def title(self):
        return "Latent Reference"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            ref_images = gr.Image(
                label="Reference Images",
                type="pil",
                image_mode='RGBA', 
                sources=["upload"],
            )

            resize_scale = gr.Slider(
                minimum=64,
                maximum=4096,
                value=1024,
                step=1,
                label="Resize to ~N px (width and height)",
                info="Reference images will be resized so their largest dimension is close to this value. Generation resolution will be overridden to match.",
            )

            size_multiple = gr.Slider(
                minimum=8,
                maximum=128,
                value=8,
                step=1,
                label="Size multiple (alignment)",
                info="After resizing, width and height are rounded to the nearest multiple of this value",
            )

            interpolation = gr.Radio(
                choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
                value="lanczos",
                label="Resize interpolation method",
            )

            bg_color = gr.ColorPicker(
                value="#000000",
                label="Background color",
                info="For transparent imagess: this color fills the background behind the foreground image",
            )

        return (
            enable,
            ref_images,
            resize_scale,
            size_multiple,
            interpolation,
            bg_color,
        )

    def process(
        self,
        p,
        enable,
        ref_images,
        resize_scale,
        size_multiple,
        interpolation,
        bg_color,
        *args,
        **kwargs,
    ):
        if not enable:
            return

        if ref_images is None:
            return

        sd_model = shared.sd_model

        if not getattr(sd_model, "is_wan", False):
            return

        if isinstance(ref_images, list):
            ref_imgs = ref_images
        else:
            ref_imgs = [ref_images]

        self.ref_latents = []

        for img in ref_imgs:
            resized_img, out_h, out_w = self._resize_image(
                img,
                resize_scale,
                size_multiple,
                interpolation,
                bg_color,
            )

            img_tensor = (
                torch.from_numpy(resized_img)
                .float()
                .div_(255.0)
            )

            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            elif img_tensor.ndim == 4:
                img_tensor = img_tensor.permute(0, 3, 1, 2)

            img_tensor = img_tensor.to(
                shared.device,
                dtype=devices.dtype_vae,
            )

            img_tensor = img_tensor * 2.0 - 1.0

            latent = sd_model.encode_first_stage(img_tensor)
            self.ref_latents.append(latent)

        p.width = out_w
        p.height = out_h

        print(
            f"[Latent Reference] Generation size overridden: "
            f"{p.width}x{p.height}"
        )

        unet = p.sd_model.forge_objects.unet
        unet.model_options["latent_ref_images"] = self.ref_latents[:]

    def _resize_image(
        self,
        img: Image.Image,
        resize_scale: float,
        size_multiple: int,
        interpolation: str = "lanczos",
        bg_color: str = "#000000",
    ):
        img = img.convert("RGBA")

        w, h = img.size

        target_area = resize_scale ** 2
        orig_area = w * h

        factor = math.sqrt(target_area / orig_area)

        new_w = int(round((w * factor) / size_multiple) * size_multiple)
        new_h = int(round((h * factor) / size_multiple) * size_multiple)

        new_w = max(new_w, size_multiple)
        new_h = max(new_h, size_multiple)

        bg_rgb = self._hex_to_rgb(bg_color)

        bg = Image.new(
            "RGBA",
            img.size,
            (
                bg_rgb[0],
                bg_rgb[1],
                bg_rgb[2],
                255,
            ),
        )

        result = Image.alpha_composite(bg, img)

        pillow_interp = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
            "area": Image.BOX,
        }

        result = result.resize(
            (new_w, new_h),
            pillow_interp.get(
                interpolation.lower(),
                Image.LANCZOS,
            ),
        )

        result = result.convert("RGB")

        resized_img = np.array(result, dtype=np.uint8)

        return resized_img, new_h, new_w

    def _hex_to_rgb(self, hex_color: str):
        h = hex_color.lstrip("#")

        return (
            int(h[0:2], 16),
            int(h[2:4], 16),
            int(h[4:6], 16),
        )

    def process_before_every_sampling(self, p, *args, **kwargs):
        if not getattr(self, "ref_latents", None):
            return

        unet = p.sd_model.forge_objects.unet.clone()
        refs = self.ref_latents[:]

        prev_wrapper = unet.model_options.get("model_function_wrapper", None)

        def latent_ref_unet_wrapper(model_apply, model_kwargs):
            input_x = model_kwargs.get("input")

            if not isinstance(input_x, torch.Tensor) or input_x.ndim != 5:
                if prev_wrapper is not None:
                    return prev_wrapper(model_apply, model_kwargs)
                else:
                    mk = model_kwargs.copy()
                    x_val = mk.pop("input")
                    t_val = mk.pop("timestep")
                    cond_dict = mk.pop("c", {})
                    return model_apply(x_val, t_val, **cond_dict, **mk)

            orig_T = input_x.shape[2]
            new_x = input_x

            for ref in refs:
                if not isinstance(ref, torch.Tensor):
                    continue

                ref = ref.to(dtype=new_x.dtype, device=new_x.device)

                if ref.ndim == 4:
                    ref = ref.unsqueeze(2)

                bs_x = new_x.shape[0]
                bs_ref = ref.shape[0]
                if bs_ref != bs_x:
                    if bs_x % bs_ref == 0:
                        ref = ref.repeat(bs_x // bs_ref, 1, 1, 1, 1)
                    else:
                        ref = ref.expand(bs_x, -1, -1, -1, -1)

                new_x = torch.cat([new_x, ref], dim=2)

            model_kwargs['input'] = new_x

            if prev_wrapper is not None:
                out = prev_wrapper(model_apply, model_kwargs)
            else:
                mk = model_kwargs.copy()
                x_val = mk.pop("input")
                t_val = mk.pop("timestep")
                cond_dict = mk.pop("c", {})
                out = model_apply(x_val, t_val, **cond_dict, **mk)

            out = out[:, :, :orig_T, :, :]

            return out

        unet.set_model_unet_function_wrapper(latent_ref_unet_wrapper)
        p.sd_model.forge_objects.unet = unet

    def postprocess(self, p, processed, *args, **kwargs):
        self.ref_latents = []
