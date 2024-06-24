import os
import pickle

import albumentations
from matplotlib import pyplot as plt
import pywt
import torch.nn.functional as F
import cv2
import einops
import numpy
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import transforms as T

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()






class flowersPaths(Dataset):
    def __init__(self, path, size=256):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # example = self.myturenimg(self.images[i])
        image = Image.open(self.images[i])
        t = transforms.Compose(

            [ transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
              transforms.ToTensor(),
              # transforms.Normalize(
              #     mean=torch.tensor([0.485, 0.456, 0.406]),
              #     std=torch.tensor([0.229, 0.224, 0.225])),
              ])
        return t(image)

def save_flower(x,path):

    x = einops.rearrange(x, "b c w h ->c (b w)  h ")
    tu = transforms.ToPILImage()
    x = tu(x)
    x.save(path)



def sava_radar(x,path):
    x = x*10
    x = x.cpu().detach().numpy()
    x = einops.rearrange(x,"b c w h-> (b w) (c h)")


    plt.imshow(x,vmax=10,vmin=0,cmap="jet")
    plt.savefig(path,dpi=600)


    







CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}


def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)

            yield img.convert(mode)
        except EOFError:
            break
        i += 1

def gif_to_tensor1(path, channels = 3, transform = T.Compose([T.Resize(size=(16, 16)), T.ToTensor()])):
    img = Image.open(path)

    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))

    return torch.stack(tensors, dim = 1)

class Get_tager_sample_gif(Dataset):
    def __init__(self, path):
        self.img_path = os.listdir(path)
        self.path = path

    def __getitem__(self, idx):
        img_name = self.img_path[idx]

        path = os.path.join(self.path, img_name)

        gif = gif_to_tensor1(path)

        return gif,self.img_path[idx]


    def __len__(self):
        return len(self.img_path)
def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images







transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),

])











    


import torch.nn.functional as F
from torchvision import transforms as T, utils

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}
from torch.utils import data
from pathlib import Path
from functools import partial

# gif -> (channels, frame, height, width) tensor
def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]
    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1
def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


    
class Get_tager_sample_h5npy(Dataset):
    def __init__(self,path):
        with open(path, 'r') as validation_file:
            self.name = [line.strip() for line in validation_file.readlines()]
        self.t = transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    def __getitem__(self, idx):
        x = numpy.load(os.path.join("somedata",self.name[idx]))
        x = self.t(x)
       
    
       
        image_array = einops.rearrange(x,"(c t) w h -> c t w h",t=20)
        image_array = einops.repeat(image_array,"c t w h-> (c a) t w h",a=3)
        # t c w h
        img_low = F.interpolate(image_array[:,:4], size=(32, 32), mode='bilinear', align_corners=False)
       
        return img_low, image_array[:,:4,...], image_array[:,4:,...],self.name[idx]

      
    def __len__(self):
        return len(self.name)
    






































    
    
    
    
    
    
    
    
    
    
    
    
