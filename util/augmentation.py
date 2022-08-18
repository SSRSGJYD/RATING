import cv2
import numpy as np
from PIL import Image
import random
from skimage import util
import torch
from torchvision import transforms


class ROICropResize:
    def __init__(self, roi, input_size, policy='roi', xshift=(-10, 60), yshift=(-10, 30), xscale=(0.8, 1.2), yscale=(0.8, 1.2), train=True):
        self.roi = roi
        self.input_size = input_size
        self.policy = policy
        self.xshift = xshift
        self.yshift = yshift
        self.xscale = xscale
        self.yscale = yscale
        self.train = train
        
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            PIL_image = transforms.ToPILImage()(img)
        else:
            PIL_image = img

        x1, y1, x2, y2 = self.roi
        input_w, input_h = self.input_size

        if self.policy == 'roi':
            if self.train:
                xshift1 = random.randint(*self.xshift)
                xshift2 = random.randint(*self.xshift)
                yshift1 = random.randint(*self.yshift)
                yshift2 = random.randint(*self.yshift)
                x1 = max(0, x1-xshift1)
                y1 = max(0, y1-yshift1)
                x2 = min(PIL_image.size[0], x2+xshift2)
                y2 = min(PIL_image.size[1], y2+yshift2)
                crop_image = PIL_image.crop((x1, y1, x2, y2))
            else:
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(PIL_image.size[0], x2)
                y2 = min(PIL_image.size[1], y2)
                crop_image = PIL_image.crop((x1, y1, x2, y2))

        elif self.policy == 'roi_random':
            if self.train:
                width = (x2 - x1)
                height = (y2 - y1)
                xshift = (self.xshift[1] - self.xshift[0]) * random.random() + self.xshift[0]
                yshift = (self.yshift[1] - self.yshift[0]) * random.random() + self.yshift[0]
                xc = (x1 + x2) // 2 + int(xshift * width)
                yc = (y1 + y2) // 2 + int(yshift * height)

                xscale = (self.xscale[1] - self.xscale[0]) * random.random() + self.xscale[0]
                yscale = (self.yscale[1] - self.yscale[0]) * random.random() + self.yscale[0]

                x1 = max(0, xc - int(xscale * width // 2))
                y1 = max(0, yc - int(yscale * height // 2))
                x2 = min(PIL_image.size[0], xc + int(xscale * width // 2))
                y2 = min(PIL_image.size[1], yc + int(yscale * height // 2))
                crop_image = PIL_image.crop((x1, y1, x2, y2))
            else:
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(PIL_image.size[0], x2)
                y2 = min(PIL_image.size[1], y2)
                crop_image = PIL_image.crop((x1, y1, x2, y2))

        elif self.policy == 'roi_extend':
            if self.train:
                width_half = (x2 - x1) // 2
                height_half = (y2 - y1) // 2
                xshift = random.randint(*self.xshift)
                yshift = random.randint(*self.xshift)
                xc = (x1 + x2) // 2 + xshift
                yc = (y1 + y2) // 2 + yshift
                width_half += np.abs(xshift)
                height_half += np.abs(yshift)
                aspect_ratio = input_w / input_h
                if width_half / height_half > aspect_ratio:
                    height_half = int(width_half / aspect_ratio)
                else:
                    width_half = int(height_half * aspect_ratio)
                crop_image = PIL_image.crop((max(0, xc-width_half), max(0, yc-height_half), 
                                        min(PIL_image.size[0], xc+width_half), min(PIL_image.size[1], yc+height_half)))
            else:
                xc = (x1 + x2) // 2
                yc = (y1 + y2) // 2
                width_half = (x2 - x1) // 2
                height_half = (y2 - y1) // 2
                aspect_ratio = input_w / input_h
                if width_half / height_half > aspect_ratio:
                    height_half = int(width_half / aspect_ratio)
                else:
                    width_half = int(height_half * aspect_ratio)
                crop_image = PIL_image.crop((max(0, xc-width_half), max(0, yc-height_half), 
                                        min(PIL_image.size[0], xc+width_half), min(PIL_image.size[1], yc+height_half)))

        elif self.policy == 'noroi':
            if self.train:
                xshift1 = random.randint(0, 50)
                xshift2 = random.randint(PIL_image.size[0]-50, PIL_image.size[0])
                yshift1 = random.randint(0, 30)
                yshift2 = random.randint(PIL_image.size[1]-30, PIL_image.size[1])
                crop_image = PIL_image.crop((xshift1, yshift1, xshift2, yshift2))
            else:
                crop_image = PIL_image

        resize_image = crop_image.resize(self.input_size, Image.BILINEAR)
        if isinstance(img, torch.Tensor):
            return transforms.ToTensor()(resize_image)
        else:
            return resize_image

    def __repr__(self):
        return self.__class__.__name__

class MaskedDoppler:
    def __init__(self):
        pass
            
    def __call__(self, img):
        assert isinstance(img, Image.Image)
        np_array = np.array(img)
        img_HSV = cv2.cvtColor(np_array, cv2.COLOR_BGR2HSV)
        mask = img_HSV[:, :, 1] > 40 / 100 * 255
        mask = mask.astype(np.uint8)
        np_array[mask == 1] = [0, 0, 255]
        return Image.fromarray(np_array.astype(np.uint8))
    
    def __repr__(self):
        return self.__class__.__name__
