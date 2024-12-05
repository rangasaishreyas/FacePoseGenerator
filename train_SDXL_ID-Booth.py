#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import contextlib
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    #EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin as LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import re

from Arcface_files.ArcFace_functions import preprocess_image_for_ArcFace, prepare_locked_ArcFace_model
import re 
from facenet_pytorch import MTCNN

import configs.config_train_SDXL as cfg


# from diffusers.models.attention_processor import AttnProcessor2_0

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def atoi(text):
        return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename, revision=revision
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type


def save_model_card(
    repo_id: str,
    use_dora: bool,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    vae_path=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# {'SDXL' if 'playground' not in base_model else 'Playground'} LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    if "playground" in base_model:
        model_description += """\n
## License

Please adhere to the licensing terms as described [here](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++" if "playground" not in base_model else "playground-v2dot5-community",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora" if not use_dora else "dora",
        "template:sd-lora",
    ]
    if "playground" in base_model:
        tags.extend(["playground", "playground-diffusers"])
    else:
        tags.extend(["stable-diffusion-xl", "stable-diffusion-xl-diffusers"])

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {cfg.num_validation_images} images with prompt:"
        f" {cfg.validation_prompt}."
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if not cfg.do_edm_style_training:
        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed) if cfg.seed else None
    # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
    # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
    # inference_ctx = (
    #     contextlib.nullcontext() if "playground" in cfg.pretrained_model_name_or_path else torch.cuda.amp.autocast()
    # )

    phase_name = "test" if is_final_validation else "validation"
    images = []
    #with inference_ctx:
    for i in range(cfg.num_validation_images):
        #with torch.cuda.amp.autocast():
        image = pipeline(**pipeline_args, generator=generator).images[0]
        #print(np.array(image))
        images.append(image)
        folder_path = os.path.join(args.output_dir, phase_name)
        os.makedirs(folder_path,exist_ok=True)
        image_filename = f"{folder_path}/{epoch}_validation_img_{i}.jpg"
        image.save(image_filename)

    #with inference_ctx:
    #    images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(cfg.num_validation_images)]

    for tracker in accelerator.trackers:
        
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")

    del pipeline
    torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def latents_to_pil_images(latents, vae):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def previous_latents_to_image_for_mtcnn(latents, vae):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample #[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).round()[0]#.dtype("uint8")
    image = torch.permute(image, (1, 2, 0))
    return image

def latents_to_image_for_mtcnn(latents, vae):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    #with torch.no_grad():
    image = vae.decode(latents).sample #[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255)[0]#.round()[0]#.dtype("uint8")
    image = torch.permute(image, (1, 2, 0))
    return image

# convert a normalized image back to RGB format suitable for PIL
def reverse_normalized_image(img, multiply_255=True):
    mean = 0.5
    std = 0.5 

    denorm = transforms.Normalize(
        mean=[-mean / std],
        std=[1.0 / std],
    )

    img = denorm(img).clip(0, 1)    

    if multiply_255:
        img = (img * 255).to(dtype=torch.uint8)    
    return img 


def save_normalized_image(images, path):
    images = images[0]
    images = reverse_normalized_image(images, multiply_255=True)
    img_save = transforms.functional.to_pil_image(images, mode="RGB")
    img_save.save(path)
    return True 

def latents_decode(latents, vae):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents  # 0.18215
    with torch.no_grad():
        image = vae.decode(latents).sample #[0]

    return image


# def latents_to_image_for_arcface(latents, vae):
#     # bath of latents -> list of images
#     latents = (1 / 0.18215) * latents
#     latents = latents # .to(dtype=torch.float16)
#     with torch.no_grad():
#         image = vae.decode(latents).sample #[0]
        
#     image = torch.nn.functional.adaptive_avg_pool2d(image, (112,112))
#     return image

def cropped_image_to_arcface_input(img):
    # might not have square dimensions
    #  transform to (1, 3, X, X)
    img = torch.permute(img, (2, 0, 1))
    #img = torch.nn.functional.adaptive_avg_pool2d(img, (112,112)) # TODO 
    # img = transforms.functional.resize(img, (112, 112), antialias=None) # TODO 
    #print(img.shape)
    img = torch.unsqueeze(img, 0)
    img = torch.nn.functional.interpolate(img, size=[112, 112])#[3, 112,112]) 
    #print(img.shape)
    img = ((img / 255) - 0.5) / 0.5 

    #img = img[None, :, :, :]
    
    return img




def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != cfg.local_rank:
        cfg.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
        self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if cfg.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if center_crop:
                y1 = max(0, int(round((image.height - cfg.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - cfg.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (cfg.resolution, cfg.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            self.crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        self.instance_identity_embeds_path = sorted(list(Path(instance_data_root.replace("images", "ArcFace_embeds")).iterdir()))
        

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_identity_embeds_path = sorted(list(Path(class_data_root.replace("images", "ArcFace_embeds")).iterdir()))

        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        original_size = self.original_sizes[index % self.num_instance_images]
        crop_top_left = self.crop_top_lefts[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["original_size"] = original_size
        example["crop_top_left"] = crop_top_left

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # costum prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        example["instance_identity_embeds"] = torch.load(self.instance_identity_embeds_path[index % self.num_instance_images]) 
        

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            example["class_prompt"] = self.class_prompt
            example["class_identity_embeds"] = torch.load(self.class_identity_embeds_path[index % self.num_class_images]) 
        
        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    identity_embed = [example["instance_identity_embeds"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        original_sizes += [example["original_size"] for example in examples]
        crop_top_lefts += [example["crop_top_left"] for example in examples]

        identity_embed += [example["class_identity_embeds"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    identity_embed = torch.cat(identity_embed, dim=0)
    

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "identity_embed": identity_embed
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):

    if cfg.do_edm_style_training and cfg.snr_gamma is not None:
        raise ValueError("Min-SNR formulation is not supported when conducting EDM-style training.")

    logging_dir = Path(args.output_dir, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs()#find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Generate class images if prior preservation is enabled.
    if cfg.with_prior_preservation:
        class_images_dir = Path(cfg.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < cfg.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if cfg.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif cfg.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif cfg.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                cfg.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=cfg.revision,
                variant=cfg.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = cfg.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(cfg.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=cfg.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=cfg.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=cfg.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    scheduler_type = determine_scheduler_type(cfg.pretrained_model_name_or_path, cfg.revision)
    if "EDM" in scheduler_type:
        cfg.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
        logger.info("Performing EDM-style training!")
    elif cfg.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )
        logger.info("Performing EDM-style training!")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")

    #print("NOISE SCHEDULER:", noise_scheduler, "\n")
    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision, variant=cfg.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=cfg.revision, variant=cfg.variant
    )
    vae_path = (
        cfg.pretrained_model_name_or_path
        if cfg.pretrained_vae_model_name_or_path is None
        else cfg.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if cfg.pretrained_vae_model_name_or_path is None else None,
        revision=cfg.revision,
        variant=cfg.variant,
    )
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, variant=cfg.variant
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32) # TODO  #dtype=weight_dtype)#dtype=torch.float32) # TODO FIX weight type

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if cfg.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=cfg.lora_rank,
        use_dora=cfg.use_dora,
        lora_alpha=cfg.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # unet.set_attn_processor(AttnProcessor2_0())
    # unet.set_default_attn_processor()
    
    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if cfg.train_text_encoder:
        text_lora_config = LoraConfig(
            r=cfg.lora_rank,
            use_dora=cfg.use_dora,
            lora_alpha=cfg.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(model, type(unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if cfg.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if cfg.mixed_precision == "fp16":
            models = [unet_]
            if cfg.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if cfg.mixed_precision == "fp16":
        models = [unet]
        if cfg.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))

    if cfg.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))

    # Optimization parameters
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": cfg.learning_rate}
    if cfg.train_text_encoder:
        # different learning rate for text encoder and unet
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": cfg.adam_weight_decay_text_encoder,
            "lr": cfg.text_encoder_lr if cfg.text_encoder_lr else cfg.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": cfg.adam_weight_decay_text_encoder,
            "lr": cfg.text_encoder_lr if cfg.text_encoder_lr else cfg.learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]

    # Optimizer creation
    if not (cfg.optimizer.lower() == "prodigy" or cfg.optimizer.lower() == "adamw"):
        logger.warn(
            f"Unsupported choice of optimizer: {cfg.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        cfg.optimizer = "adamw"

    if cfg.use_8bit_adam and not cfg.optimizer.lower() == "adamw":
        logger.warn(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {cfg.optimizer.lower()}"
        )

    if cfg.optimizer.lower() == "adamw":
        if cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,
        )

    if cfg.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if cfg.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if cfg.train_text_encoder and cfg.text_encoder_lr:
            logger.warn(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {cfg.text_encoder_lr} and learning_rate: {cfg.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = cfg.learning_rate
            params_to_optimize[2]["lr"] = cfg.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            beta3=cfg.prodigy_beta3,
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,
            decouple=cfg.prodigy_decouple,
            use_bias_correction=cfg.prodigy_use_bias_correction,
            safeguard_warmup=cfg.prodigy_safeguard_warmup,
        )


    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=cfg.instance_data_dir,
        instance_prompt=cfg.instance_prompt,
        class_prompt=cfg.class_prompt,
        class_data_root=cfg.class_data_dir if cfg.with_prior_preservation else None,
        class_num=cfg.num_class_images,
        size=cfg.resolution,
        #repeats=cfg.repeats,
        #center_crop=cfg.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, cfg.with_prior_preservation),
        num_workers=cfg.dataloader_num_workers,
    )

    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids

    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (cfg.resolution, cfg.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    if not cfg.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not cfg.train_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
            cfg.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if cfg.with_prior_preservation:
        if not cfg.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
                cfg.class_prompt, text_encoders, tokenizers
            )
    
        
    # Clear the memory here
    if not cfg.train_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        del tokenizer_one, tokenizer_two
        del text_encoder_one, text_encoder_two
        gc.collect()
        torch.cuda.empty_cache()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if not cfg.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds
            if cfg.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                unet_add_text_embeds = torch.cat([unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
        # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
        # batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, cfg.instance_prompt)
            tokens_two = tokenize_prompt(tokenizer_two, cfg.instance_prompt)
            if cfg.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, cfg.class_prompt)
                class_tokens_two = tokenize_prompt(tokenizer_two, cfg.class_prompt)
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.lr_num_cycles,
        power=cfg.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if cfg.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = (
            "dreambooth-lora-sd-xl"
            if "playground" not in cfg.pretrained_model_name_or_path
            else "dreambooth-lora-playground"
        )
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

        ###########################################################
    # ArcFaceModel for Loss 
    arcface_model = prepare_locked_ArcFace_model()
    arcface_model.to(device=accelerator.device)

    # TODO 
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def cosine_distance(x, y):
        sim = F.cosine_similarity(x, y) 
        distance = 1 - sim
        return distance
    
    triplet_loss_function = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance)
    
    # MTCNN model for face detection
    mtcnn_model = MTCNN(image_size=112,device=accelerator.device, margin=0)



    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    final_state = False

    for epoch in range(first_epoch, cfg.num_train_epochs):
        unet.train()
        avg_combined_loss = []; avg_id_loss = []; avg_instance_loss = []; avg_prior_loss = []
        
        if cfg.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

            # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_two).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                gt_arcface_embed = batch["identity_embed"].to(dtype=vae.dtype) # shape [2,512]
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    if not cfg.train_text_encoder:
                        prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts)
                        tokens_two = tokenize_prompt(tokenizer_two, prompts)

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()

                if latents_mean is None and latents_std is None:
                    model_input = model_input * vae.config.scaling_factor
                    if cfg.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                else:
                    latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                    latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                    model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
                    model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                if not cfg.do_edm_style_training:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    # timesteps = torch.randint(
                    #     1, 10, (bsz,), device=model_input.device
                    # )
                    timesteps = timesteps.long()
                else:
                    # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
                    # instead of discrete timesteps, so here we sample indices to get the noise levels
                    # from `scheduler.timesteps`
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                # For EDM-style training, we first obtain the sigmas based on the continuous timesteps.
                # We then precondition the final model inputs based on these sigmas instead of the timesteps.
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                if cfg.do_edm_style_training:
                    sigmas = get_sigmas(timesteps, len(noisy_model_input.shape), noisy_model_input.dtype)
                    if "EDM" in scheduler_type:
                        inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_model_input, sigmas)
                    else:
                        inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

                # time ids
                add_time_ids = torch.cat(
                    [
                        compute_time_ids(original_size=s, crops_coords_top_left=c)
                        for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                    ]
                )

                # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                if not train_dataset.custom_instance_prompts:
                    elems_to_repeat_text_embeds = bsz // 2 if cfg.with_prior_preservation else bsz
                else:
                    elems_to_repeat_text_embeds = 1

                # Predict the noise residual
                if not cfg.train_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids,
                        "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                    }
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        inp_noisy_latents if cfg.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]
                else:
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        inp_noisy_latents if cfg.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]

                weighting = None
                if cfg.do_edm_style_training:
                    # Similar to the input preconditioning, the model predictions are also preconditioned
                    # on noised model inputs (before preconditioning) and the sigmas.
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    if "EDM" in scheduler_type:
                        model_pred = noise_scheduler.precondition_outputs(noisy_model_input, model_pred, sigmas)
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_model_input
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                noisy_model_input / (sigmas**2 + 1)
                            )
                    # We are not doing weighting here because it tends result in numerical problems.
                    # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                    # There might be other alternatives for weighting as well:
                    # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                    if "EDM" not in scheduler_type:
                        weighting = (sigmas**-2.0).float()

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = model_input if cfg.do_edm_style_training else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        model_input
                        if cfg.do_edm_style_training
                        else noise_scheduler.get_velocity(model_input, noise, timesteps)
                    )
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if cfg.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    if weighting is not None:
                        print("Prior ", weighting)
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                                target_prior.shape[0], -1
                            ),
                            1,
                        )
                        prior_loss = prior_loss.mean()
                    else:
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                if cfg.snr_gamma is None:
                    if weighting is not None:
                        print("SNR gamma", weighting)
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                                target.shape[0], -1
                            ),
                            1,
                        )
                        instance_loss = loss.mean()
                    else:
                        instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                loss = instance_loss 

                if cfg.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + cfg.prior_loss_weight * prior_loss
                
                #print(cfg.which_loss)
                if cfg.which_loss == "identity": 
                    # save_normalized_image(pixel_values, f"images_during_training/input_{step}.jpg")
                    latent_x0 = noise_scheduler.step(model_pred, timesteps[0], noisy_model_input[0]).pred_original_sample

                    # tmp_img = latents_decode(latent_x0, vae)
                    # save_normalized_image(tmp_img, f"images_during_training/latent_decoded_{timesteps[0]}.jpg")
                    # exit()
                    # Perform face detection first 
                    img = latents_to_image_for_mtcnn(latent_x0.to(vae.dtype), vae) 
                    
                    bboxs, probs = mtcnn_model.detect(img, landmarks=False)
                    if bboxs is not None: #and bboxs_prior is not None:  
                        bbox = bboxs[0].astype(int) 
                        initial_size = img.shape[0]
                        img_cropped = img[max(0,bbox[1]): min(bbox[3], initial_size ) , max(0, bbox[0]): min(bbox[2], initial_size)] 
                    
                        img_cropped = cropped_image_to_arcface_input(img_cropped)
                        pred_arcface_features = arcface_model(img_cropped)
                        
                        # compare arcface features between predicted face and gt face
                        arcface_cos_similarity = cos(pred_arcface_features, gt_arcface_embed[0])  
                        identity_loss = 1 - arcface_cos_similarity #((1 - arcface_cos_similarity) + (1 - arcface_cos_similarity_prior)) / 2 # TODO check is this ok                         

                        identity_noise_level_weight =  (1 -  timesteps[0] / noise_scheduler.config.num_train_timesteps) ** 2
                        if not cfg.timestep_loss_weighting: identity_noise_level_weight = 1 
                        loss = loss + identity_noise_level_weight * identity_loss
                    

                elif cfg.which_loss == "triplet_prior": 
                        
                    latent_x0 = noise_scheduler.step(model_pred, timesteps[0], noisy_model_input[0]).pred_original_sample
                    
                    # Perform face detection first 
                    img = latents_to_image_for_mtcnn(latent_x0.to(vae.dtype), vae) 
                    bboxs, probs = mtcnn_model.detect(img, landmarks=False)
                    
                    #latent_x0_prior = noise_scheduler.step(model_pred_prior, timesteps[1], noisy_model_input[1]).pred_original_sample
                    #img_prior = latents_to_image_for_mtcnn(latent_x0_prior.to(weight_dtype), vae) 
                    #bboxs_prior, probs_prior = mtcnn_model.detect(img_prior, landmarks=False)

                    if bboxs is not None: #and bboxs_prior is not None:  
                        bbox = bboxs[0].astype(int) 
                        initial_size = img.shape[0]
                        img_cropped = img[max(0,bbox[1]): min(bbox[3], initial_size ) , max(0, bbox[0]): min(bbox[2], initial_size)] 
                        
                        img_cropped = cropped_image_to_arcface_input(img_cropped)
                        pred_arcface_features = arcface_model(img_cropped)                   

                        identity_noise_level_weight =  (1 - timesteps[0] / noise_scheduler.config.num_train_timesteps) ** 2
                        if not cfg.timestep_loss_weighting: identity_noise_level_weight = 1 

                        # input: anchor, positive, negative    
                        triplet_loss = triplet_loss_function(pred_arcface_features, gt_arcface_embed[0][None, :], gt_arcface_embed[1][None, :])
                        loss = loss + identity_noise_level_weight * triplet_loss
                # if cfg.subloss == "only_triplet":
                #     loss = 0 * loss + 0 * prior_loss + triplet_loss #identity_noise_level_weight * identity_loss
                
                # elif cfg.subloss == "without_prior":
                #     loss = loss + triplet_loss

                # elif cfg.subloss == "only_reconstruction":
                #     loss =  loss # + 0 * prior_loss + triplet_loss
                # else: 
                # loss = loss + cfg.prior_loss_weight * prior_loss + triplet_loss
                # loss = loss + 1000
                # print("total loss:", loss)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
                        if cfg.train_text_encoder
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            
            step_loss = loss.detach().item()
            step_instance_loss = instance_loss.detach().item()
            step_prior_loss = prior_loss.detach().item()

            if cfg.which_loss == "identity":
                step_id_loss = identity_loss.detach().item()

            elif cfg.which_loss == "triplet_prior":
                step_id_loss = triplet_loss.detach().item()

            else: step_id_loss = 0
            
            avg_combined_loss.append(step_loss)
            avg_instance_loss.append(step_instance_loss)
            avg_prior_loss.append(step_prior_loss)
            avg_id_loss.append(step_id_loss)

            logs = {"Step Loss/Reconstruction": step_instance_loss, "Step Loss/ID": step_id_loss,  
                    "Step Loss/Prior": step_prior_loss, "Step Loss/Combined": step_loss, "LR": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if cfg.max_train_steps is not None and global_step >= cfg.max_train_steps:
                final_state = True
                print("Stop training")
                break

        if accelerator.is_main_process:
            print("Perform validation during training")
            if cfg.validation_prompt is not None and epoch % cfg.validation_epochs == 0:
                # create pipeline
                # if not cfg.train_text_encoder:
                #     text_encoder_one = text_encoder_cls_one.from_pretrained(
                #         cfg.pretrained_model_name_or_path,
                #         subfolder="text_encoder",
                #         revision=cfg.revision,
                #         variant=cfg.variant,
                #     )
                #     text_encoder_two = text_encoder_cls_two.from_pretrained(
                #         cfg.pretrained_model_name_or_path,
                #         subfolder="text_encoder_2",
                #         revision=cfg.revision,
                #         variant=cfg.variant,
                #     )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    cfg.pretrained_model_name_or_path,
                    vae=vae,
                    #text_encoder=accelerator.unwrap_model(text_encoder_one),
                    #text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    unet=accelerator.unwrap_model(unet),
                    revision=cfg.revision,
                    variant=cfg.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline_args = {"prompt": cfg.validation_prompt, "num_inference_steps": 30}

                images = log_validation(
                    pipeline,
                    args,
                    accelerator,
                    pipeline_args,
                    epoch,
                )
        
        if accelerator.is_main_process:
            if epoch % cfg.checkpointing_epochs == 0 or final_state:
            #if final_state:
                #print("Global step", epoch)
                #print("Checkpointing:", cfg.checkpointing_epochs)
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if cfg.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= cfg.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

        epoch_logs = {"Epoch Loss/Reconstruction": np.mean(np.array(avg_instance_loss)), "Epoch Loss/ID": np.mean(np.array(avg_id_loss)),
                      "Epoch Loss/Prior": np.mean(np.array(avg_prior_loss)), "Epoch Loss/Combined": np.mean(np.array(avg_combined_loss)),}
        accelerator.log(epoch_logs, step=global_step)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        #unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if cfg.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            )
            text_encoder_two = unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two.to(torch.float32))
            )
        
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        # Final inference
        # Load previous pipeline
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if cfg.pretrained_vae_model_name_or_path is None else None,
            revision=cfg.revision,
            variant=cfg.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            cfg.pretrained_model_name_or_path,
            vae=vae,
            revision=cfg.revision,
            variant=cfg.variant,
            torch_dtype=weight_dtype,
        )

        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if cfg.validation_prompt and cfg.num_validation_images > 0:
            pipeline_args = {"prompt": cfg.validation_prompt, "num_inference_steps": 30}
            images = log_validation(
                pipeline,
                args,
                accelerator,
                pipeline_args,
                epoch,
                is_final_validation=True,
            )


    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    
    id_folders = os.listdir(cfg.source_folder)
    
    cfg.timestep_loss_weighting = True
    
    #cfg.which_loss = ""

    for which_loss in cfg.losses_to_test: #, "identity", "triplet_prior"]:
        print("Loss:", which_loss)
        # cfg.subloss = subloss
        cfg.which_loss = which_loss
        output_folder = cfg.output_folder
        if cfg.which_loss != "": 
            output_folder += cfg.which_loss + "_loss"
            if not cfg.timestep_loss_weighting:
                output_folder += "_NoIDWeighting"
            else: 
                output_folder += "_WithIDWeighting"
        else: 
            output_folder += "No_new_Loss"
        
        if cfg.train_text_encoder: 
            output_folder += "_WithTextEncoder"
    
        cfg.instance_data_dir = cfg.source_folder
        args.output_dir = output_folder
        
        print("Args:", vars(args))
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save args to json
        json_output_file = os.path.join(args.output_dir, "training_config.json")
        with open(json_output_file, 'w') as fp:

            config_vars={var:vars(cfg)[var] for var in dir(cfg) if not var.startswith('_')}
            args_vars = vars(args) 
            all_args = config_vars | args_vars
            json.dump(all_args, fp, indent=4)

        id_folders.sort(key=natural_keys)
        
        for i, id_folder in enumerate(id_folders): 
            #print(id_folder)
            cfg.instance_data_dir = os.path.join(cfg.source_folder, id_folder) # "./DATASETS/TUFTS_TEST_512/images_id_1"
            args.output_dir = os.path.join(output_folder, id_folder) 
            if os.path.exists(os.path.join(args.output_dir,"checkpoint-1000")):
                print("Already trained:", args.output_dir)
                continue  
            main(args)
            #break 
        #break