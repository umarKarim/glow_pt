import torch 
import torch.utils.data as udata
import torchvision.transforms as transforms 
import os 
from PIL import Image 


class Dataset():
    def __init__(self, path, im_size, nbits):
        self.path = path 
        self.nbits = nbits 
        self.im_size = im_size 
        self.im_paths = self.gather_im_paths()
        self.tfms = transforms.Compose([ 
            transforms.Resize(self.im_size),
            transforms.CenterCrop(self.im_size),
            transforms.ToTensor()
        ])

    def gather_im_paths(self):
        im_paths = sorted([os.path.join(self.path, x) for x in os.listdir(self.path)
                            if x.endswith('.jpg')])
        return im_paths 

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, i):
        im = Image.open(self.im_paths[i])
        im = self.tfms(im)
        im = im * 255.0
        im = torch.floor(im / (2 ** (8 - self.nbits))) / (2 ** self.nbits) - 0.5
        return im 

def get_dataloader(name=None, path='/isize/umar/datasets/celebA/class1/',
                    batch_size=16, im_size=64, 
                    nbits=5):
    dataset = Dataset(
        path=path,
        im_size=im_size,
        nbits=nbits
    )
    dataloader = udata.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return dataloader

