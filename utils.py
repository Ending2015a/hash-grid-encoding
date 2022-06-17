# --- built in ---
import os
# --- 3rd party ---
import matplotlib.pyplot as plt
import numpy as np
import imageio
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 10000000000

# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def srgb_to_linear(img):
  limit = 0.04045
  return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)
    
def linear_to_srgb(img):
  limit = 0.0031308
  return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def read_image_imageio(filename):
  image = imageio.imread(filename)
  image = np.asarray(image).astype(np.float32)
  if len(image.shape) == 2:
    image = image[:, :, np.newaxis]
  return image / 255.0

def write_image_imageio(filename, image, quality=95):
  image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
  kwargs = {}
  if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg']:
    if image.ndim >= 3 and image.shape[2] > 3:
      image = image[:, :, :3]
    kwargs['quality'] = quality
    kwargs['subsampling'] = 0
  imageio.imwrite(filename, image, **kwargs)

def read_image(filename):
  if os.path.splitext(filename)[1] == '.npy':
    image = np.load(filename)
  else:
    image = read_image_imageio(filename)
    image = srgb_to_linear(image)
  return image

def write_image(filename, image, quality=95):
  if os.path.splitext(filename)[1] == '.npy':
    # here we encode the image to npy format
    np.save(filename, image)
  else:
    image = linear_to_srgb(np.clip(image, 0.0, 1.0))
    write_image_imageio(filename, image, quality=quality)
