import torch.utils.data as data
from imageio import imread
import os

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams


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
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        self.length = None
        self._load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Returns an image and its label"""
        img = imread(self.img_files[idx])
        label = imread(self.label_files[idx])
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

