# --- built in ---
import os
import argparse
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import tqdm
# --- my module ---
from train import Task
import utils

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt', type=str)
  parser.add_argument('-o', '--output', type=str, default='output/frame_{:06d}.png')
  parser.add_argument('--zoom', type=float, default=10.0)
  parser.add_argument('--center_x', type=float, default=0.5)
  parser.add_argument('--center_y', type=float, default=0.5)
  parser.add_argument('--width', type=int, default=480)
  parser.add_argument('--height', type=int, default=640)
  parser.add_argument('--n_frames', type=int, default=150)
  return parser.parse_args()


def smoothstep(e0, e1, x):
  t = np.clip((x-e0)/(e1-e0), 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)

def lerp(x, y, a):
  return x * (1-a) + y * a

@torch.no_grad()
def render(a, model, grid, frame_idx, zoom_factor, center):
  grid = (grid-center) / zoom_factor + center
  bound_min = np.min(grid, axis=0)
  bound_max = np.max(grid, axis=0)
  move = np.maximum(0.0 - bound_min, 0.0)
  move = move + np.minimum(1.0 - (bound_max + move), 0.0)
  grid = grid + move
  outs = model(torch.from_numpy(grid)).cpu().numpy()
  return outs.reshape((a.height, a.width, -1))


def main():
  a = get_args()
  model = Task.load_from_checkpoint(a.ckpt).to(device='cuda')

  # generate grid
  u_res = 0.0
  v_res = 0.0
  u = np.linspace(u_res, 1-u_res, a.height)
  v = np.linspace(v_res, 1-v_res, a.width)
  u, v = np.meshgrid(u, v, indexing='ij')
  grid = np.stack((v.flatten(), u.flatten()), axis=0).T # (n, 2)
  grid = grid.astype(np.float32)

  stay = 25
  zoom_in = stay + int(0.8 * (a.n_frames-stay))
  zoom_out = a.n_frames

  target_center = np.array((a.center_x, a.center_y), dtype=np.float32)

  for frame_idx in tqdm.tqdm(range(a.n_frames)):
    if frame_idx < stay:
      zoom_factor = 1.0
    elif frame_idx < zoom_in:
      frame_time = smoothstep(stay, zoom_in, frame_idx)
      zoom_factor = lerp(1.0, a.zoom, frame_time)
    else:
      frame_time = smoothstep(zoom_in, zoom_out, frame_idx)
      zoom_factor = lerp(a.zoom, 1.0, frame_time)
    canvas = render(a, model, grid.copy(), frame_idx, zoom_factor, target_center)
    path = a.output.format(frame_idx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.write_image(path, canvas, quality=95)

if __name__ == '__main__':
  main()


