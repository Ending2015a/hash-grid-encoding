# Hash Grid Encoding

This repo contains an implementation of NVidia's hash grid encoding from [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/) and a runnable example of gigapixel tasks. The hash grid is implemented in ***pure PyTorch*** hense it's more human friendly than [NVidia's original implementation (C++/CUDA)](https://github.com/NVlabs/instant-ngp).


Some features:
* Implemented in ***pure PyTorch***.
* Supports ***arbitrary dimensions***.

## How To Use

### MultiResHashGrid
To use the [`MultiResHashGrid`](encoding.py#L129) in your own project, you can simply copy-paste the code in `encoding.py` into your project. For example:
```python
import torch
import encoding

enc = encoding.MultiResHashGrid(2)  # 2D image data
enc = encoding.MultiResHashGrid(3)  # 3D data

dim = 3
batch_size = 100

# The input value must be within the range [0, 1]
input = torch.rand((batch_size, dim), dtpye=torch.float32)
enc_input = enc(input)

# Then you can forward into your network
model = MyMLP(dim=enc_input.output_dim, out_dim=1)
output = model(enc_input)

# Move to other devices
enc = enc.to(dtype='cuda')
```

### Gigapixel task
This repo also contains a runnable gigapixel image task, which is implemented based on [PyTorch Lightning](https://www.pytorchlightning.ai/). For more instructions of running this code, see [Examples](#Examples).

## Examples

### Albert

![](https://github.com/Ending2015a/hash-grid-encoding/blob/master/data/albert-compare.gif)

Run this example:
```
python train.py -i data/albert.jpg --enc_method hashgrid --visualize
```


### Tokyo

![](https://github.com/Ending2015a/hash-grid-encoding/blob/master/data/tokyo-compare.gif)

https://user-images.githubusercontent.com/18180004/174231919-16705ae3-357e-4c50-832c-bae6f1d92556.mp4

Download [the tokyo image](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837) and place it at `data/tokyo.jpg`.

To run the tokyo example in its original size (56718 x 21450 pixels), your GPU must have memory at least 20GB. If your GPU have no such amount of memory, you can use the `convert.py` script to scale down the image size into half. By converting to `.npy` format can also increase the loading speed:

```shell
python convert.py -i data/tokyo.jpg -o data/tokyo.npy --scale 0.5
```

Then run the experiment

```shell
python train.py -i data/tokyo.npy --enc_method hashgrid --finest_resolution 32768 --visualize
```

