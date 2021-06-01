import albumentations as A
import numpy as np
import glob
import os

from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2

transform = {
    'train': A.Compose([A.CenterCrop(height=650, width=1110, p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Rotate(limit=45, p=0.3),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.5),
                        A.Resize(height=720, width=1280, p=1.0),
                        A.Cutout(num_holes=15, max_h_size=50, max_w_size=50, p=0.5),
                        ToTensorV2(p=1.0)], p=1.0, bbox_params={'format': 'pascal_voc',
                                                        'label_fields': ['labels'],
                                                        'min_area': 0,
                                                        'min_visibility': 0}),

    'valid': A.Compose([A.Resize(height=720, width=1280, p=1.0),
                        ToTensorV2(p=1.0)], p=1.0, bbox_params={'format': 'pascal_voc',
                                                               'label_fields': ['labels'],
                                                               'min_area': 0,
                                                               'min_visibility': 0,}),
    
    'test': A.Compose([A.Resize(height=720, width=1280, p=1.0),
                       ToTensorV2(p=1.0)], p=1.0),
                        
}

class TestDataset(Dataset):
    '''
    Dataset for inference
    
    Input:
    image_root: The root directory of images of one side camera
    depthmaps_path: The related deathmaps path with .npy format ###should be in the same order with the images in the image_root!!!!!
    isleft: Is it the left camera data?

    Return:
    image: Tensor -> Image after preprocessing with shape (channel, height, width)
    depthmap: numpy -> The depthmap of the image returned
    isleft: bool -> Is it the left camera data?
    '''
    def __init__(self, images_root, depthmaps_root, transform=transform['test']):
        self.images_root = images_root
        self.depthmaps_root = depthmaps_root
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(images_root, '*'))
        self.dm_paths = glob.glob(os.path.join(depthmaps_root, '*'))
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        
        dm = np.load(self.dm_paths[index])
        tsfm = self.transform(**{'image':image})

        return tsfm['image'], dm
    
    def __len__(self):
        return len(self.image_paths)