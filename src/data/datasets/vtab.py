"""
Borrowed form https://github.com/KMnP/vpt/blob/main/src/data/datasets/tf_dataset.py
"""

import functools
import tensorflow.compat.v1 as tf
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm

from collections import Counter
from torch import Tensor
from PIL import Image

from ..vtab_datasets import base

# pylint: disable=unused-import
from ..vtab_datasets import caltech
from ..vtab_datasets import cifar
from ..vtab_datasets import clevr
from ..vtab_datasets import diabetic_retinopathy
from ..vtab_datasets import dmlab
from ..vtab_datasets import dsprites
from ..vtab_datasets import dtd
from ..vtab_datasets import eurosat
from ..vtab_datasets import kitti
from ..vtab_datasets import oxford_flowers102
from ..vtab_datasets import oxford_iiit_pet
from ..vtab_datasets import patch_camelyon
from ..vtab_datasets import resisc45
from ..vtab_datasets import smallnorb
from ..vtab_datasets import sun397
from ..vtab_datasets import svhn
from ..vtab_datasets.registry import Registry

tf.config.experimental.set_visible_devices(
    [], "GPU"
)  # set tensorflow to not use gpu  # noqa
DATASETS = [
    "caltech101",
    "cifar(num_classes=100)",
    "dtd",
    "oxford_flowers102",
    "oxford_iiit_pet",
    "patch_camelyon",
    "sun397",
    "svhn",
    "resisc45",
    "eurosat",
    "dmlab",
    'kitti(task="closest_vehicle_distance")',
    'smallnorb(predicted_attribute="label_azimuth")',
    'smallnorb(predicted_attribute="label_elevation")',
    'dsprites(predicted_attribute="label_x_position",num_classes=16)',
    'dsprites(predicted_attribute="label_orientation",num_classes=16)',
    'clevr(task="closest_object_distance")',
    'clevr(task="count_all")',
    'diabetic_retinopathy(config="btgraham-300")',
]


class TFDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, transform=None):
        assert split in {
            "train",
            "val",
            "test",
            "trainval",
        }, "Split '{}' not supported for {} dataset".format(split, cfg.DATA.NAME)
        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME

        self.img_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.img_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        self.get_data(cfg, split)
        self._transform = transform

    def get_data(self, cfg, split):
        tf_data = build_tf_dataset(cfg, split)
        # enhance speed by using prefetch
        tf_data = tf_data.prefetch(1)

        for i, data in enumerate(tqdm(tf_data, desc="Loading data", unit="batch")):
            if i == 0:
                self._image_list = [Image.fromarray(data[0].numpy().squeeze())]
                self._targets = [data[1].numpy()[0]]
            else:
                self._image_list.append(
                    Image.fromarray(data[0].numpy().squeeze())
                )
                self._targets.append(data[1].numpy()[0])
        self._class_ids = sorted(set(self._targets))
        print("Number of images: {}".format(len(self._image_list)))
        print(
            "Number of classes: {} / {}".format(
                len(self._class_ids), self.get_class_num()
            )
        )

        del tf_data

    def get_info(self):
        num_imgs = len(self._image_tensor_list)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, "
                + "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == "inv":
            mu = -1.0
        elif weight_type == "inv_sqrt":
            mu = -0.5
        weight_list = num_per_cls**mu
        weight_list = np.divide(weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        label = self._targets[index]
        im = self._transform(self._image_list[index])

        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        return im, label

    def __len__(self):
        return len(self._targets)


def preprocess_fn(data, size=224, input_range=(0.0, 1.0)):
    image = data["image"]
    image = tf.image.resize(image, [size, size])

    image = tf.cast(image, tf.float32) / 255.0
    image = image * (input_range[1] - input_range[0]) + input_range[0]

    data["image"] = image
    return data


def build_tf_dataset(cfg, mode):
    """
    Builds a tf data instance, then transform to a list of tensors and labels
    """

    if mode not in ["train", "val", "test", "trainval"]:
        raise ValueError(
            "The input pipeline supports `train`, `val`, `test`."
            "Provided mode is {}".format(mode)
        )

    vtab_dataname = cfg.DATA.NAME.split("vtab-")[-1]
    data_dir = cfg.DATA.DATAPATH
    if vtab_dataname in DATASETS:
        data_cls = Registry.lookup("data." + vtab_dataname)
        vtab_tf_dataloader = data_cls(data_dir=data_dir)
    else:
        raise ValueError(
            'Unknown type for "dataset" field: {}'.format(type(vtab_dataname))
        )

    split_name_dict = {
        "dataset_train_split_name": "train",
        "dataset_val_split_name": "val200",
        "dataset_trainval_split_name": "train800val200",
        "dataset_test_split_name": "test",
    }

    def _dict_to_tuple(batch):
        return batch["image"], batch["label"]

    return vtab_tf_dataloader.get_tf_data(
        batch_size=1,  # data_params["batch_size"],
        drop_remainder=False,
        split_name=split_name_dict[f"dataset_{mode}_split_name"],
        for_eval=mode != "train",  # handles shuffling
        shuffle_buffer_size=1000,
        prefetch=1,
        train_examples=None,
        epochs=1,  # setting epochs to 1 make sure it returns one copy of the dataset
    ).map(_dict_to_tuple)


def to_torch_imgs(img: np.ndarray, mean: Tensor, std: Tensor) -> Tensor:
    t_img: Tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    t_img -= mean
    t_img /= std

    return t_img
