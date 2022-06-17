# --- built in ---
import os
import time
import argparse
# --- 3rd party ---
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import imageio
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 10000000000
# --- 3rd party ---
import utils

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str)
  parser.add_argument('-o', '--output', type=str, default=None)
  parser.add_argument('-s', '--scale', type=float, default=0.5)
  return parser.parse_args()

def main():
  a = get_args()
  filename = a.input
  scale = a.scale

  start_time = time.time()
  image = utils.read_image(filename)
  print('Took {} seconds to load image'.format(time.time() - start_time))

  h, w, c = image.shape
  print(f"{w}x{h} pixels, {c} channels")
  if scale != 1.0:
    h = int(h*scale)
    w = int(w*scale)
    print(f"Scaling image to {w}x{h} pixels")
    image = resize(image, (h, w))

  output = a.output
  if a.output is None:
    output = os.path.splitext(filename)[0] + '.npy'

  utils.write_image(output, image.astype(np.float16))



if __name__ == '__main__':
  main()
    