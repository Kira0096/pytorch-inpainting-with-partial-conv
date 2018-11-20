import random
import torch
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train', label_dict=None):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        # if split == 'train':
        self.paths = glob('{:s}/{:s}/**/*.jpg'.format(img_root, split),
                              recursive=True)
        self.mask_paths = glob('{:s}/*.jpg'.format(mask_root))
        self.N_mask = len(self.mask_paths)
        self.label_dict = label_dict
        self.no_mask = not(self.label_dict == None)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))
        label = self.paths[index].split('/')[-2]

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB')) if not self.no_mask \
            else torch.ones_like(gt_img)

        if self.label_dict == None:
            return gt_img * mask, mask, gt_img, torch.FloatTensor(0)
        else:
            return gt_img * mask, mask, gt_img, torch.FloatTensor(self.label_dict[label]) 

    def __len__(self):
        return len(self.paths)
