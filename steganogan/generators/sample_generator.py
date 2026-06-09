import os
from typing import List, Tuple
from imageio import imread
from PIL import Image
import torch
import torch.nn as nn


class SampleGenerator:
    """Generates sample stego images during training for visual quality monitoring."""

    def __init__(self, encoder: nn.Module, payload_generator: object,
                 device: torch.device) -> None:
        self.encoder:           nn.Module     = encoder
        self.payload_generator: object        = payload_generator
        self.device:            torch.device  = device

    def generate_samples(self, samples_path: str, epoch: int,
                         text_to_encode: str, data_depth: int) -> None:
        """
        Encode a text message into up to 8 callback images and save a grid PNG.

        Parameters
        ----------
        samples_path    : directory where the grid image will be saved
        epoch           : current training epoch (used in the output filename)
        text_to_encode  : secret text to embed into each sample image
        data_depth      : bits per pixel used during encoding
        """
        callback_images_path = os.path.join('data', 'callback_images')
        if not os.path.exists(callback_images_path):
            os.makedirs(callback_images_path)
            raise ValueError(
                'callback_images directory not found. '
                'Please add images to generate samples.'
            )

        image_filenames = sorted(os.listdir(callback_images_path))
        if len(image_filenames) < 8:
            raise ValueError('Expected at least 8 images in callback_images.')

        reshaped_tensors: List[torch.Tensor] = []

        for filename in image_filenames[:8]:
            path = os.path.join(callback_images_path, filename)

            image  = imread(path, pilmode='RGB') / 127.5 - 1.0
            tensor = torch.FloatTensor(image).permute(2, 0, 1)

            if text_to_encode:
                cover      = tensor.unsqueeze(0).to(self.device)
                cover_size = cover.size()

                payload = self.payload_generator.make_payload(
                    cover_size[3], cover_size[2], data_depth, text_to_encode
                ).to(self.device)

                raw_out = self.encoder(cover, payload)
                # Support both standard and iterative encoders
                encoded = raw_out[-1] if isinstance(raw_out, (list, tuple)) else raw_out
                tensor  = encoded.squeeze(0).clamp(-1.0, 1.0).to(self.device)

            resized = nn.functional.interpolate(
                tensor.unsqueeze(0), size=(360, 360),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            reshaped_tensors.append(resized)

        self._create_and_save_grid(reshaped_tensors, samples_path, epoch)

    def _create_and_save_grid(self, reshaped_tensors: List[torch.Tensor],
                              samples_path: str, epoch: int) -> None:
        """
        Arrange tensors into a 4×2 grid image and save it as a PNG.

        Parameters
        ----------
        reshaped_tensors : list of 8 image tensors, each (3, H, W) in [-1, 1]
        samples_path     : output directory
        epoch            : epoch number for filename
        """
        batch  = torch.stack(reshaped_tensors).clamp(-1.0, 1.0)
        batch  = ((batch + 1.0) / 2.0 * 255.0).byte()
        images = [Image.fromarray(t.permute(1, 2, 0).cpu().numpy()) for t in batch]

        grid_cols = 4
        grid_rows = 2
        gap       = 20
        img_w, img_h = images[0].size
        total_w = grid_cols * img_w + (grid_cols - 1) * gap
        total_h = grid_rows * img_h + (grid_rows - 1) * gap

        grid_img = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))
        for idx, img in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            grid_img.paste(img, (col * (img_w + gap), row * (img_h + gap)))

        grid_img.save(os.path.join(samples_path, f'grid_epoch_{epoch}.png'))
