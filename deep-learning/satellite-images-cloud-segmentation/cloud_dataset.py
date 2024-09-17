import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CloudDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, transform=None):
        super().__init__()
        # nir: near-infrared band
        # gt: ground truth label
        # loop red dir files and combine into a dictionary with the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir)
                      for f in r_dir.iterdir() if not f.is_dir()]

        self.transform = transform

    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        files = {
            'red': r_file,
            'green': g_dir / r_file.name.replace('red', 'green'),
            'blue': b_dir / r_file.name.replace('red', 'blue'),
            'nir': nir_dir / r_file.name.replace('red', 'nir'),
            'gt': gt_dir / r_file.name.replace('red', 'gt')
        }
        return files

    def __len__(self):
        return len(self.files)

    def open_as_nparray(self, idx, include_nir=False, invert=False):

        rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                        np.array(Image.open(self.files[idx]['green'])),
                        np.array(Image.open(self.files[idx]['blue']))], axis=2)
        # shape is HxWx3

        if include_nir:
            # add extra dimension to get HxWx1
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), axis=2)
            rgb = np.concatenate((rgb, nir), axis=2)

        if invert:  # instead of HxWxC, get CxHxW
            rgb = rgb.transpose((2, 0, 1))

        # return normalized rgb, values between [0, 1]
        return rgb / np.iinfo(rgb.dtype).max

    def __getitem__(self, idx):
        # NCHW format
        x = self.open_as_nparray(idx, include_nir=True, invert=False)
        y = self.open_mask(idx, add_dim=False)

        x = Image.fromarray((x * 255).astype(np.uint8))  # assuming x is normalized between 0 and 1
        y = Image.fromarray(y.astype(np.uint8))

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y.squeeze()

    def open_mask(self, idx, add_dim=False):
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask == 255, 1, 0)
        return np.expand_dims(raw_mask, 0) if add_dim else raw_mask

    def open_as_pil(self, idx):
        arr = 256 * self.open_as_nparray(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')

    def __repr__(self):
        s = f'Dataset class, {self.__len__()} files'
        return s