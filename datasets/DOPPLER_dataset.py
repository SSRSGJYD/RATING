import json
import random
import cv2
import numpy as np
import tqdm
import torch
from torchvision import transforms

from .base_dataset import BaseDataset
from util import read_PIL_image
import util.augmentation as aug

class DOPPLERDataset(BaseDataset):
    """ Doppler US image dataset. """
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
                    self.samples[class_label].append((sample, class_label, class_label))
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
                    self.sample_list.append((sample, class_label, class_label))

    def prepare_new_epoch(self):
        if self.task == 'train':
            self.sample_list = []
            if self.opt.sample_strategy == 'original':
                for k, v in self.samples.items():
                    for sample in v:
                        self.sample_list.append(sample)
            elif self.opt.sample_strategy == 'resample':
                max_class_num = max(len(v) for k, v in self.samples.items())
                for k, v in self.samples.items():
                    repeat = max_class_num // len(v)
                    self.sample_list.extend(v * repeat)
                    self.sample_list.extend(random.sample(v,  max_class_num % len(v)))
                    
            random.shuffle(self.sample_list)

    def _load_sample(self, sample):
        image = read_PIL_image(sample['DOPPLER_path'])
        if self.task == 'train':
            if self.opt.policy == 'noroi':
                center = None
            else:
                center = ((sample['DOPPLER_roi_anno'][0]+sample['DOPPLER_roi_anno'][2]) // 2, (sample['DOPPLER_roi_anno'][1]+sample['DOPPLER_roi_anno'][3]) // 2)
            augmentation = transforms.Compose([
                transforms.RandomRotation(5, center=center),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
                aug.ROICropResize(sample['DOPPLER_roi_anno'], (self.opt.input_w, self.opt.input_h), self.opt.policy, 
                                (0, 0), (0, 0), (0.9, 1.1), (0.9, 1.1), True),

                aug.MaskedDoppler(), 
                transforms.ToTensor(),
            ])
        else:
            augmentation = transforms.Compose([
                aug.ROICropResize(sample['DOPPLER_roi_anno'], (self.opt.input_w, self.opt.input_h), self.opt.policy, 
                                (0, 60), (0, 30), train=False),
                aug.MaskedDoppler(), 
                transforms.ToTensor(),
            ])
        tensor = augmentation(image)
        tensor = self.norm_data(tensor)

        crop_image = None
        if self.opt.visualize:
            vis_image = np.array(image)
            x1, y1, x2, y2 = sample['DOPPLER_roi_anno']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0,0,255), 2)
        else:
            vis_image = None
        return tensor, vis_image, crop_image

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        result = self.sample_list[idx]
        sample = result[0]
        labels = result[1:]
        data, vis_image, crop_image = self._load_sample(sample)
        return data, vis_image, crop_image, labels, str(idx)+'(level={})'.format(labels[0])

    def cal_target(self, model):
        return model.instance_ys[0], model.instance_scores[0], model.instance_preds[0]

    @staticmethod
    def collate_fn(data):
        """ data: [(image, vis_image, crop_image, labels, id)] """

        images = []
        instance_labels = [[] for _ in range(len(data[0][3]))]
        bbox_list = []
        whole_images = []
        crop_images = []
        sample_id_list = []

        for batch in data:
            image, vis_image, crop_image, labels, id = batch
            images.append(image)
            for i, label in enumerate(labels):
                instance_labels[i].append(label)
            bbox_list.append([])
            whole_images.append(vis_image)
            crop_images.append(crop_image)
            sample_id_list.append(id)
        
        collate_data = {
            'input': [torch.stack(images, dim=0)],
            'vis_image': [whole_images, crop_images],
            'instance_label': [torch.LongTensor(instance_label) for instance_label in instance_labels],
            'bbox_list': bbox_list,
            'id': sample_id_list
        }

        return collate_data