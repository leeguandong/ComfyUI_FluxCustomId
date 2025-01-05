"""
Author: LiDongyang(yingtian.ldy@alibaba-inc.com | amo5lee@aliyun.com)
Date: 2024-10
Description: Customized Image Generation Model Based on Facial ID.
"""
import os
from typing import List
# import math
import torch
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import pdb
from .utils import is_torch2_available, get_generator
from .attention_processor import CATRefFluxAttnProcessor2_0
from .resampler import PerceiverAttention, FeedForward
from ..utils.insightface_package import FaceAnalysis2, analyze_faces
import cv2

USE_DAFAULT_ATTN = False  # should be True for visualization_attnmap


class FacePerceiverResampler(torch.nn.Module):
    def __init__(
            self,
            *,
            dim=768,
            depth=4,
            dim_head=64,
            heads=16,
            embedding_dim=1280,
            output_dim=768,
            ff_mult=4,
    ):
        super().__init__()

        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class ProjPlusModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4,
                 output_dim=3072):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=output_dim,
            ff_mult=4,
        )
        self.prj_out_clip = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim * 2, output_dim),
        )

    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return torch.cat([out, self.prj_out_clip(clip_embeds)], dim=1)


class CustomIDModel:
    def __init__(self, sd_pipe, trained_ckpt, device, dtype, num_tokens=4,
                 image_encoder_path="/dev_share/gdli7/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        self.device = device
        self.dtype = dtype
        self.trained_ckpt = trained_ckpt
        self.num_tokens = num_tokens
        self.pipe = sd_pipe
        self.image_encoder_path = image_encoder_path

        # load image encoder
        self.clip_image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device)

        self.set_id_adapter()

        # image proj model
        self.image_proj_model = self.init_proj()
        self.image_proj_model.to(self.device)
        if self.trained_ckpt != None:
            self.load_id_adapter()

        self.face_detector = FaceAnalysis2(name="antelopev2", root="/dev_share/gdli7/models/insightface",
                                           providers=['CUDAExecutionProvider'],
                                           allowed_modules=['detection', 'recognition'])
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))

    def init_proj(self):
        image_proj_model = ProjPlusModel(
            cross_attention_dim=self.image_encoder.config.hidden_size,
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
            num_tokens=self.num_tokens,
            output_dim=self.pipe.transformer.config.num_attention_heads * self.pipe.transformer.config.attention_head_dim,
        ).to(self.device)
        return image_proj_model

    def set_id_adapter(self):
        # init adapter modules
        attn_procs = {}
        transformer_sd = self.pipe.transformer.state_dict()
        for name in self.pipe.transformer.attn_processors.keys():
            if name.startswith("transformer_blocks"):
                attn_procs[name] = CATRefFluxAttnProcessor2_0(
                    self.pipe.transformer.config.num_attention_heads * self.pipe.transformer.config.attention_head_dim,
                    self.pipe.transformer.config.num_attention_heads * self.pipe.transformer.config.attention_head_dim,
                    self.pipe.transformer.config.attention_head_dim,
                    self.num_tokens + 256,  # !
                ).to(self.device, dtype=self.dtype)
            elif name.startswith("single_transformer_blocks"):
                attn_procs[name] = CATRefFluxAttnProcessor2_0(
                    self.pipe.transformer.config.num_attention_heads * self.pipe.transformer.config.attention_head_dim,
                    self.pipe.transformer.config.num_attention_heads * self.pipe.transformer.config.attention_head_dim,
                    self.pipe.transformer.config.attention_head_dim,
                    self.num_tokens + 256,
                ).to(self.device, dtype=self.dtype)
        self.pipe.transformer.set_attn_processor(attn_procs)

    def load_id_adapter(self):
        state_dict = torch.load(self.trained_ckpt, map_location=torch.device('cpu'))
        self.image_proj_model.load_state_dict(state_dict["img_prj_state"], strict=True)
        m, u = self.pipe.transformer.load_state_dict(state_dict["attn_processor_state"], strict=False)
        assert len(u) == 0

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        # image_ = cv2.imread(pil_image)
        image_ = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        # image_ = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)
        faces = analyze_faces(self.face_detector, image_)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).to(self.device)
        faceid_embeds = faceid_embeds.unsqueeze(0)

        # clip
        # face_image = Image.open(pil_image)
        face_image = pil_image
        if isinstance(face_image, Image.Image):
            pil_image = [face_image]
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]

        ip_tokens = self.image_proj_model(faceid_embeds, clip_image_embeds[:, 1:, :])
        assert ip_tokens.shape[1] == self.num_tokens + 256

        return ip_tokens.to(self.device, dtype=self.dtype)

    def generate(
            self,
            pil_image=None,
            prompt=None,
            num_samples=4,
            height=1024,
            width=1024,
            seed=None,
            num_inference_steps=30,
            guidance_scale=3.5,
    ):

        ip_tokens = self.get_image_embeds(pil_image=pil_image)

        bs_embed, seq_len, _ = ip_tokens.shape
        ip_tokens = ip_tokens.repeat(1, num_samples, 1)
        ip_tokens = ip_tokens.view(bs_embed * num_samples, seq_len, -1)
        ip_tokens = ip_tokens.to(self.device).to(self.dtype)

        ip_token_ids = self.pipe._prepare_latent_image_ids(
            1,
            1 * 2,
            (self.num_tokens + 256) * 2,
            self.device,
            self.dtype,
        )
        images = self.pipe(
            prompt,
            ip_token=ip_tokens,
            ip_token_ids=ip_token_ids,
            num_images_per_prompt=num_samples,
            height=height,
            width=width,
            output_type="pil",
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(self.device).manual_seed(seed),
            guidance_scale=guidance_scale,
        ).images
        # import pdb;pdb.set_trace()

        return images
