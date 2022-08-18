import itertools
import random
import numpy as np
import torch

from .base_dataset import BaseDataset


class JigsawPuzzle(BaseDataset):
    """
    Implementation of 'Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles'.
    """

    def __init__(self, opt, task, dataset, permutation_num, permutation_file=None, patch_width=64, patch_height=64, width_crops=3, height_crops=3):
        super().__init__(opt, task)
        self.dataset = dataset
        self.permutation_num = permutation_num
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.width_crops = width_crops
        self.height_crops = height_crops
        if permutation_file is not None:
            all_permutations = np.load(permutation_file)
            print('Jigsaw permutation Hamming set loaded')
        else:
            all_permutations = np.array(list(itertools.permutations(list(range(self.width_crops * self.height_crops)))))

        self.max_hamming_set = np.array(all_permutations[:self.permutation_num], dtype=np.uint8)
        self.opt.classes = [[str(i) for i in range(permutation_num)]] * 2

    def prepare_dataset(self):
        pass

    def _create_jigsaws(self, image):
        '''
        image: torch tensor of (C, H, W)
        '''
        perm_index = random.randrange(self.permutation_num)
        height, width = image.shape[-2:]
        cell_height = height // self.height_crops
        cell_width = width // self.width_crops

        final_crops = []
        for idx in self.max_hamming_set[perm_index]:
            row = idx // self.width_crops
            col = idx % self.width_crops
            x_start = col * cell_width + random.randrange(cell_width - self.patch_width)
            y_start = row * cell_height + random.randrange(cell_height - self.patch_height)            
            final_crops.append(image[..., y_start:y_start + self.patch_height, x_start:x_start + self.patch_width])
        
        return final_crops, perm_index

    def prepare_new_epoch(self):
        self.dataset.prepare_new_epoch()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        items = self.dataset.__getitem__(idx)
        tensor = items[0] # (C, H, W)
        tensors, label = self._create_jigsaws(tensor) # a list of (C, H, W)
        return tensors, label, str(idx)

    def cal_target(self, model):
        return model.instance_ys[0], model.instance_scores[0], model.instance_preds[0]

    @staticmethod
    def collate_fn(data):
        """ data: [(tensors, label, idx)] """

        images = []
        instance_label = []
        sample_id_list = []

        for batch in data:
            tensors, label, id = batch
            images.extend(tensors)
            instance_label.append(label)
            sample_id_list.append(id)
        
        collate_data = {
            'input': [torch.stack(images, dim=0)],
            'instance_label': [torch.LongTensor(instance_label)] * 2,
            'id': sample_id_list
        }

        return collate_data