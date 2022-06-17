# --- built in ---
from typing import Any, List, Dict, Union, Tuple, Optional, Callable
import os
import math
import argparse
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import imageio
import tqdm
# --- my module ---
import utils
import encoding

# --- datasets ---

class Sampler2D(nn.Module):
  def __init__(self, filename: str):
    super().__init__()
    data = torch.from_numpy(utils.read_image(filename)).to(dtype=torch.float16)
    self.register_buffer('data', data, persistent=False)
    mesh = self.get_mesh().float()
    self.register_buffer('mesh', mesh, persistent=False)
    self.shape = self.data.shape
    h, w, c = self.shape
    self.h = h
    self.w = w
    self.c = c
    self.num_pixels = h * w
    resolution = torch.tensor((self.shape[1], self.shape[0]), dtype=torch.float32)
    self.register_buffer('resolution', resolution, persistent=True)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    shape = self.data.shape
    x = x * self.resolution
    ind = x.long()
    w = x - ind.float()
    x0 = ind[:, 0].clamp(min=0, max=shape[1]-1)
    y0 = ind[:, 1].clamp(min=0, max=shape[0]-1)
    x1 = (x0 + 1).clamp(max=shape[1]-1)
    y1 = (y0 + 1).clamp(max=shape[0]-1)
    return (
      self.data[y0, x0].to(dtype=torch.float32) * (1.0 - w[:,0:1]) * (1.0 - w[:,1:2]) +
      self.data[y0, x1].to(dtype=torch.float32) * w[:,0:1] * (1.0 - w[:,1:2]) +
      self.data[y1, x0].to(dtype=torch.float32) * (1.0 - w[:,0:1]) * w[:,1:2] +
      self.data[y1, x1].to(dtype=torch.float32) * w[:,0:1] * w[:,1:2]
    )

  def get_mesh(self) -> torch.Tensor:
    h, w, c = self.data.shape
    n_pixels = h * w
    u_res = 0.5 / h
    v_res = 0.5 / w
    u = np.linspace(u_res, 1-u_res, h)
    v = np.linspace(v_res, 1-v_res, w)
    u, v = np.meshgrid(u, v, indexing='ij')
    xy = np.stack((v.flatten(), u.flatten()), axis=0).T # (n, 2)
    xy = xy.astype(np.float32)
    xy = torch.from_numpy(xy)
    return xy

class TaskDataset(Dataset):
  def __init__(
    self,
    sampler: Sampler2D,
    batch_size: int,
    n_samples: int
  ):
    super().__init__()
    self.sampler = sampler
    self.batch_size = batch_size
    self.n_samples = n_samples
    self.jit_sampler = None
  
  def setup(self):
    self.jit_sampler = torch.jit.trace(self.sampler, self.get_rand())

  def __len__(self):
    return self.n_samples

  def get_rand(self):
    return torch.rand([self.batch_size, 2],
        dtype=torch.float32, device=self.sampler.data.device)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.jit_sampler is None:
      self.setup()
    x = self.get_rand()
    return x, self.jit_sampler(x)

# --- networks ---

class MLP(nn.Module):
  def __init__(
    self,
    dim: int,
    out_dim: int = 1,
    mlp_units: List[int] = [64, 64]
  ):
    super().__init__()
    layers = []
    self.input_dim = dim
    self.output_dim = out_dim
    in_dim = dim
    for out_dim in mlp_units:
      layers.append(nn.Linear(in_dim, out_dim))
      layers.append(nn.ReLU(inplace=True))
      in_dim = out_dim
    layers.append(nn.Linear(in_dim, self.output_dim))
    self.model = nn.Sequential(*layers)
  
  def forward(self, x: torch.tensor):
    return self.model(x)

class ToyNet(nn.Module):
  def __init__(
    self,
    dim: int,
    out_dim: int = 1,
    mlp_units: List[int] = [64, 64],
    enc_method: str = 'freq',
    enc_kwargs: dict = {}
  ):
    super().__init__()
    if enc_method == 'freq':
      self.enc = encoding.Frequency(dim, **enc_kwargs)
      dim = self.enc.output_dim
    elif enc_method == 'hashgrid':
      self.enc = encoding.MultiResHashGrid(dim, **enc_kwargs)
      dim = self.enc.output_dim
    else:
      print(f'Disable encoding: {enc_method}')
      self.enc = None
    self.mlp = MLP(dim, out_dim=out_dim, mlp_units=mlp_units)

  def forward(self, x: torch.Tensor):
    if self.enc is not None:
      x = self.enc(x)
    return self.mlp(x)


class Task(pl.LightningModule):
  def __init__(
    self,
    filename: str,
    batch_size: int = 65536,
    n_samples: int = 10,
    lr: float = 1e-3,
    mlp_units: List[int] = [64, 64],
    relative_l2: bool = False,
    enc_method: Optional[str] = None,
    enc_kwargs: Dict[str, Any] = {},
    channels: int = None,
    vis_freq: Callable = None,
    inference_only: bool = False
  ):
    super().__init__()

    self.vis_freq = vis_freq
    self.inference_only = inference_only

    if not inference_only:
      self.sampler = Sampler2D(filename)
      channels = self.sampler.c
    self.save_hyperparameters(ignore=['inference_only', 'vis_freq'])

    if not inference_only:
      self.setup_dataset()
    self.setup_model()

  def setup_dataset(self):
    self.trainset = TaskDataset(
      self.sampler,
      batch_size = self.hparams.batch_size,
      n_samples = self.hparams.n_samples
    )

  def setup_model(self):
    self.model = ToyNet(
      dim = 2,
      out_dim = self.hparams.channels,
      mlp_units = self.hparams.mlp_units,
      enc_method = self.hparams.enc_method,
      enc_kwargs = self.hparams.enc_kwargs
    )

  def configure_optimizers(self):
    optim = torch.optim.Adam(
      self.model.parameters(),
      lr = self.hparams.lr,
      weight_decay = 1e-8,
      eps = 1e-8,
      betas = (0.9, 0.99),
    )
    return optim

  def train_dataloader(self):
    return DataLoader(
      self.trainset,
      batch_size = None, # manual batching
      num_workers = 0, # main thread
    )

  def forward(
    self,
    x: torch.Tensor,
  ):
    x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    return self.model(x)

  def l2_loss(self, y, y_, relative=False):
    if relative:
      return ((y-y_)**2.0) / (y_.detach()**2.0 + 0.01)
    else:
      return ((y-y_)**2.0)

  def training_step(self, batch, batch_idx: int):
    x, y = batch
    y_ = self(x)
    loss = self.l2_loss(y, y_, relative=self.hparams.relative_l2).mean()

    self.log(
      "train/loss",
      loss.item(),
      on_step = True,
      on_epoch = True,
      sync_dist = True,
      prog_bar = True
    )
    return loss

  @torch.no_grad()
  def _preview(self):
    batch_size = self.hparams.batch_size * 8
    num_batches = self.sampler.num_pixels // batch_size + 1
    start_idx = 0
    pixels = []
    mesh = self.sampler.mesh
    for _ in range(num_batches):
      if start_idx >= self.sampler.num_pixels:
        break
      stop_idx = min(start_idx + batch_size, self.sampler.num_pixels)
      mesh_slice = mesh[start_idx:stop_idx]
      outs = self(mesh_slice)
      pixels.append(outs.cpu())
      start_idx = stop_idx
    pixels = torch.cat(pixels, dim=0)
    canvas = pixels.reshape(self.sampler.shape).detach().cpu().numpy()

    path = os.path.join(
      self.logger.log_dir,
      f"predictions/steps_{self.global_step:06d}.jpg"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.write_image(path, canvas, quality=95)
  
  def on_save_checkpoint(self, checkpoint):
    if self.trainer.is_global_zero:
      res = (self.vis_freq is not None
          and self.vis_freq(self.current_epoch, self.global_step))
      if res:
        print('Visualizing results...')
        self._preview()


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, help='Path to input image (.jpg/.npy)')
  parser.add_argument('--root', type=str, default='./logs')
  parser.add_argument('--trace', type=str, default='experiments')
  parser.add_argument('--batch_size', type=int, default=65536)
  parser.add_argument('--epochs', type=int, default=400, help='100 steps per epoch')
  parser.add_argument('--device', type=int, default=0)
  parser.add_argument('--enc_method', choices=['freq', 'hashgrid', 'none'])
  parser.add_argument('--n_levels', type=int, default=16)
  parser.add_argument('--n_features_per_level', type=int, default=2)
  parser.add_argument('--log2_hashmap_size', type=int, default=15)
  parser.add_argument('--base_resolution', type=int, default=16)
  parser.add_argument('--finest_resolution', type=int, default=8192)
  parser.add_argument('--visualize', action='store_true', default=False)
  return parser.parse_args()

if __name__ == '__main__':

  a = get_args()

  def vis_func(epoch, step):
    # [1, 2, 4, 8, 10, 20, 30, 40, ...]
    epoch += 1
    if epoch < 10:
      return (epoch & (epoch-1)) == 0
    if epoch < 100:
      return epoch % 10 == 0
    if epoch < 1000:
      return epoch % 100 == 0

  root_dir = a.root
  image_file = os.path.basename(a.input)
  trace_name = a.trace
  image_name = image_file.split(".")[0]

  dir_path = os.path.join(root_dir, trace_name, image_name)

  if a.enc_method == 'freq':
    enc_kwargs = dict(
      n_levels = a.n_levels
    )
  elif a.enc_method == 'hashgrid':
    enc_kwargs = dict(
      n_levels = a.n_levels,
      n_features_per_level = a.n_features_per_level,
      log2_hashmap_size = a.log2_hashmap_size,
      base_resolution = a.base_resolution,
      finest_resolution = a.finest_resolution
    )
  elif a.enc_method == 'none':
    a.enc_method = None
    enc_kwargs = dict()

  model = Task(
    filename = a.input,
    batch_size = a.batch_size,
    n_samples = 100,
    lr = 1e-3,
    relative_l2 = True,
    enc_method = a.enc_method,
    enc_kwargs = enc_kwargs,
    vis_freq = vis_func if a.visualize else None
  )

  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    every_n_epochs = 1
  )

  trainer = pl.Trainer(
    callbacks = checkpoint_callback,
    max_epochs = a.epochs,
    accelerator = "gpu",
    devices = [a.device],
    default_root_dir = dir_path
  )

  trainer.fit(model)
