import torch
import numpy as np
import comfy.model_management as mm
from PIL import Image

from .customID.pipeline_flux import FluxPipeline
from .customID.transformer_flux import FluxTransformer2DModel
from .customID.model import CustomIDModel


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def convert_preview_image(images):
    # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
    images_tensors = []
    for img in images:
        # 将 PIL.Image 转换为 numpy.ndarray
        img_array = np.array(img)
        # 转换 numpy.ndarray 为 torch.Tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.
        # 转换图像格式为 CHW (如果需要)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        # 添加批次维度并转换为 NHWC
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)

    if len(images_tensors) > 1:
        output_image = torch.cat(images_tensors, dim=0)
    else:
        output_image = images_tensors[0]
    return output_image


class CustomIDModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["FLUX-customID.pt"], {"default": "FLUX-customID.pt"}),
                "load_local_model": ("BOOLEAN", {"default": False}),
                "num_token": ("INT", {"default": 64, "min": 4, "max": 100}),
            }, "optional": {
                "local_flux_model_path": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
                "local_customid_model_path": ("STRING", {"default": "Damo-vision/FLUX-customID"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "customidflux"

    def load_model(self, model, load_local_model, num_token, *args, **kwargs):
        _DTYPE = torch.bfloat16
        device = mm.get_torch_device()
        if load_local_model:
            flux_model_path = kwargs.get("local_flux_model_path", "black-forest-labs/FLUX.1-dev")
            customid_model_path = kwargs.get("local_customid_model_path", "Damo-vision/FLUX-customID")
        else:
            flux_model_path = "black-forest-labs/FLUX.1-dev"
            customid_model_path = "Damo-vision/FLUX-customID"

        transformer = FluxTransformer2DModel.from_pretrained(flux_model_path, subfolder="transformer",
                                                             torch_dtype=_DTYPE).to(device)
        pipe = FluxPipeline.from_pretrained(flux_model_path, transformer=transformer, torch_dtype=_DTYPE).to(device)
        customID_model = CustomIDModel(pipe, customid_model_path, device, _DTYPE, num_token)
        return (customID_model,)


class ApplyCustomIDFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True, "default": ""}),
                "guidance_scale": (
                    "FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.5, "display": "slider"}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_customid"
    CATEGORY = "customidflux"

    def apply_customid(self,
                       model,
                       image,
                       prompt,
                       guidance_scale,
                       num_inference_steps,
                       width,
                       height,
                       seed,
                       batch_size):
        image = tensor2pil(image)
        images = model.generate(pil_image=image,
                                prompt=prompt,
                                num_samples=batch_size,
                                height=height,
                                width=width,
                                seed=seed,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale)
        images = convert_preview_image(images)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "ApplyCustomIDFlux": ApplyCustomIDFlux,
    "CustomIDModelLoader": CustomIDModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomIDModelLoader": "CustomID Model Loader",
    "ApplyCustomIDFlux": "ApplyCustomIDFlux",
}
