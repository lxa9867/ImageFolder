""" This file contains some utils functions for visualization.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
import torchvision.transforms.functional as F
from einops import rearrange

def make_viz_from_samples(
    original_images,
    reconstructed_images
):
    """Generates visualization images from original images and reconstructed images.

    Args:
        original_images: A torch.Tensor, original images.
        reconstructed_images: A torch.Tensor, reconstructed images.

    Returns:
        A tuple containing two lists - images_for_saving and images_for_logging.
    """
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    reconstructed_images = reconstructed_images * 255.0
    reconstructed_images = reconstructed_images.cpu()
    
    original_images = torch.clamp(original_images, 0.0, 1.0)
    original_images *= 255.0
    original_images = original_images.cpu()

    diff_img = torch.abs(original_images - reconstructed_images)
    to_stack = [original_images, reconstructed_images, diff_img]

    images_for_logging = rearrange(
            torch.stack(to_stack),
            "(l1 l2) b c h w -> b c (l1 h) (l2 w)",
            l1=1).byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging


def make_viz_from_samples_generation(
    generated_images,
):
    generated = torch.clamp(127.5 * generated_images + 128.0, 0, 255)
    images_for_logging = rearrange(
        generated, 
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2)

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    return images_for_saving, images_for_logging