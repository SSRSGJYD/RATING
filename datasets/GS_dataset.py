import json
import random

import cv2
import numpy as np
from torchvision import transforms
import tqdm
import torch

from .base_dataset import BaseDataset
from util import read_PIL_image
import util.augmentation as aug

class GSDataset(BaseDataset):
    """ GSUS image dataset. """
    def __init__(self, opt, task):
        super().__init__(opt, task)

        if isinstance(self.opt.classes, str):
            self.opt.classes = self.opt.classes.split(',')

        self.prepare_dataset()
        self.prepare_new_epoch()
        print('total sample:', len(self.sample_list))

    def prepare_dataset(self):
        if self.task == 'train':
            self.samples = dict()
            for c in range(len(self.opt.classes[0])):
                self.samples[c] = []
            for dataset in self.datasets:
                with open(dataset, 'r') as f:
                    record = json.load(f)
                for sample in tqdm.tqdm(record):
                    if self.opt.label == 'SH':
                        class_label = self.get_class_label(sample['SH_label'])
                    elif self.opt.label == 'vascularity':
                        class_label = self.get_class_label(sample['vascularity_label'])
                    elif self.opt.label == 'combined':
                        class_label = self.get_class_label(max(sample['SH_label'], sample['vascularity_label']))
                    if class_label is None: 
                        continue
                    self.samples[class_label].append(sample)
            for k, v in self.samples.items():
                print('sample num of class {}:'.format(k), len(v))
        else:
            self.sample_list = []
            for dataset in self.datasets:
                with open(dataset, 'r') as f:
                    record = json.load(f)
                for sample in tqdm.tqdm(record):
                    if self.opt.label == 'SH':
                        class_label = self.get_class_label(sample['SH_label'])
                    elif self.opt.label == 'vascularity':
                        class_label = self.get_class_label(sample['vascularity_label'])
                    elif self.opt.label == 'combined':
                        class_label = self.get_class_label(max(sample['SH_label'], sample['vascularity_label']))
                    if class_label is None: 
                        continue
                    self.sample_list.append((sample, class_label))

    def prepare_new_epoch(self):
        if self.task == 'train':
            self.sample_list = []
            if self.opt.sample_strategy == 'original':
                for k, v in self.samples.items():
                    for sample in v:
                        self.sample_list.append((sample, k))
            elif self.opt.sample_strategy == 'resample':
                max_class_num = max(len(v) for k, v in self.samples.items())
                for k, v in self.samples.items():
                    repeat = max_class_num // len(v)
                    candidates = [(sample, k) for sample in v]
                    self.sample_list.extend(candidates * repeat)
                    self.sample_list.extend(random.sample(candidates,  max_class_num % len(v)))
                    
            random.shuffle(self.sample_list)

    def _load_sample(self, sample):
        image = read_PIL_image(sample['GS_path'])
        if self.task == 'train':
            if self.opt.policy == 'noroi':
                center = None
            else:
                center = ((sample['GS_roi_anno'][0]+sample['GS_roi_anno'][2]) // 2, (sample['GS_roi_anno'][1]+sample['GS_roi_anno'][3]) // 2)
            augmentation = transforms.Compose([
                transforms.Grayscale(1),
                transforms.RandomRotation(5, center=center),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
                aug.ROICropResize(sample['GS_roi_anno'], (self.opt.input_w, self.opt.input_h), self.opt.policy, 
                                xshift=(-0.1, 0.1), yshift=(-0.1, 0.1), xscale=(0.8, 1.1), yscale=(0.8, 1.1), train=True),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ])
        else:
            augmentation = transforms.Compose([
                transforms.Grayscale(1),
                aug.ROICropResize(sample['GS_roi_anno'], (self.opt.input_w, self.opt.input_h), self.opt.policy, train=False),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ])
        tensor = augmentation(image)

        if self.opt.visualize:
            vis_image = np.array(transforms.Grayscale(3)(image))
            x1, y1, x2, y2 = sample['GS_roi_anno']
            crop_image = vis_image[y1:y2, x1:x2].copy()
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0,0,255), 2)
        else:
            crop_image = None
            vis_image = None
        
        tensor = self.norm_data(tensor)

        return tensor, vis_image, crop_image

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample, label = self.sample_list[idx]
        data, vis_image, crop_image = self._load_sample(sample)
        return data, vis_image, crop_image, label, str(idx)+'(level={})'.format(sample['GS_label'])

    def cal_target(self, model):
        return model.instance_ys[0], model.instance_scores[0], model.instance_preds[0]

    @staticmethod
    def collate_fn(data):
        """ data: [(image, vis_image, crop_image, label, id)] """

        images = []
        instance_label = []
        bbox_list = []
        whole_images = []
        crop_images = []
        sample_id_list = []

        for batch in data:
            image, vis_image, crop_image, label, id = batch
            images.append(image)
            instance_label.append(label)
            bbox_list.append([])
            whole_images.append(vis_image)
            crop_images.append(crop_image)
            sample_id_list.append(id)
        
        collate_data = {
            'input': [torch.stack(images, dim=0)],
            'vis_image': [crop_images],
            'instance_label': [torch.LongTensor(instance_label)] * 2,
            'bbox_list': bbox_list,
            'id': sample_id_list
        }

        return collate_data