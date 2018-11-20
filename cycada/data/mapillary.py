import torch.utils.data as data
from PIL import Image
import numpy as np
import os

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams
from .cityscapes import remap_labels_to_train_ids
from .cityscapes import id2label as LABEL2TRAIN


@register_data_params('mapillary')
class MapillaryParams(DatasetParams):
    num_channels = 3
    image_size   = 1024
    mean         = 0.5
    std          = 0.5
    num_cls      = 19
    target_transform = None


@register_dataset_obj('mapillary')
class Mapillary(data.Dataset):
    def __init__(self, root, split='train', remap_labels=True, transform=None,
                 target_transform=None):
        self.root_dir = root
        self.transform = transform
        self.remap_labels = remap_labels
        self.target_transform = target_transform
        self.length = None
        self._load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Returns an image and its label"""
        img = Image.open(self.img_files[idx]).convert('RGB')
        label = Image.open(self.label_files[idx])

        if self.transform is not None:
            img = self.transform(img)
        if self.remap_labels:
            label = np.asarray(label)
            label = remap_labels_to_train_ids(label)
            label = Image.fromarray(np.uint8(label), 'L')
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return img, label

    def _load_data(self):
        ann_dir = os.path.join(self.root_dir, 'annotations')
        img_dir = os.path.join(self.root_dir, 'images')
        label_dir = os.path.join(self.root_dir, 'labels')

        ann_files_list = os.listdir(ann_dir)

        self.ann_files = []
        self.img_files = []
        self.label_files = []
        for ann in ann_files_list:
            fname, ext = os.path.splitext(ann)
            if ext.lower() != '.xml':
                continue

            self.ann_files.append(os.path.join(ann_dir, ann))
            self.img_files.append(os.path.join(img_dir, fname + '.jpg'))
            self.label_files.append(os.path.join(label_dir, fname + '.png'))
        self.length = len(self.ann_files)


if __name__ == '__main__':
    dataset = Mapillary(root_dir='/usr/local/data/mapillary/train_small/')
    print(len(dataset))

    for idx in range(len(dataset)):
        img, lbl = dataset[idx]
        print(img.shape, lbl.shape)

