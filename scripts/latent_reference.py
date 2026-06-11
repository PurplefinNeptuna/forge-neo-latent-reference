import gradio as gr
import torch
import torchvision.transforms.functional as tvf
import numpy as np
import math
import cv2

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
                label="Reference Images (upload multiple)",
                type="numpy",
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
                info="After resizing, width and height are rounded down to the nearest multiple of this value",
            )
            interpolation = gr.Radio(
                choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
                value="bilinear",
                label="Resize interpolation method",
            )

        return enable, ref_images, resize_scale, size_multiple, interpolation

    def process(self, p, enable, ref_images, resize_scale, size_multiple, interpolation, *args, **kwargs):
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
            resized_img, out_h, out_w = self._resize_image(img, resize_scale, size_multiple, interpolation)
            img_tensor = torch.from_numpy(resized_img).float() / 255.0

            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            elif img_tensor.ndim == 4:
                img_tensor = img_tensor.permute(0, 3, 1, 2)

            img_tensor = img_tensor.to(shared.device, dtype=devices.dtype_vae)
            img_tensor = img_tensor * 2 - 1

            latent = sd_model.encode_first_stage(img_tensor)
            self.ref_latents.append(latent)

        p.width = out_w
        p.height = out_h
        print(f"[Latent Reference] Generation size overridden: {p.width}x{p.height}")

        unet = p.sd_model.forge_objects.unet
        unet.model_options["latent_ref_images"] = self.ref_latents[:]

    def _resize_image(self, img: np.ndarray, resize_scale: float, size_multiple: int, interpolation: str = "bilinear"):
        h, w = img.shape[0], img.shape[1]

        target_area = resize_scale ** 2
        orig_area = h * w

        factor = math.sqrt(target_area / orig_area)
        new_h = h * factor
        new_w = w * factor

        new_h = int(round(new_h / size_multiple) * size_multiple)
        new_w = int(round(new_w / size_multiple) * size_multiple)

        if new_h < size_multiple:
            new_h = size_multiple
        if new_w < size_multiple:
            new_w = size_multiple

        str_to_cv = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4
        }
        
        cv_interp = str_to_cv.get(interpolation.lower(), cv2.INTER_LINEAR)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv_interp)
        
        return resized_img, new_h, new_w

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
