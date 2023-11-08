import tensorflow_datasets as tfds
import argparse

_dict = {
    "caltech101": "caltech101:3.*.*",
    "cifar100": "cifar100:3.*.*",
    "clevr": "clevr:3.*.*",
    "dmlab": "dmlab:2.0.1",
    "dsprites": "dsprites:2.*.*",
    "dtd": "dtd:3.*.*",
    "eurosat": "eurosat/{}:2.*.*",
    "oxford_flowers": "oxford_flowers102:2.*.*",
    "oxford_pet": "oxford_iiit_pet:3.*.*",
    "pcam": "patch_camelyon:2.*.*",
    "smallnorb": "smallnorb:2.*.*",
    "svhn": "svhn_cropped:3.*.*",
    "sun397": "sun397:4.*.*",
    "kitti": "kitti:3.*.*",
}


class builder:
    def __init__(self, dataset_name, data_dir):
        self.dataset_name = dataset_name
        self.data_dir = "."
        self.dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)

    def build(self):
        self.dataset_builder.download_and_prepare()


# main function
parser = argparse.ArgumentParser(description="Select Dataset to prepare")
parser.add_argument(
    "--dataset",
    type=str,
    default="caltech101",
    help="For Saving and loading the current Model",
)
args = parser.parse_args()
dataset_name = args.dataset

if dataset_name in _dict.keys():
    dataset_builder = builder(_dict[dataset_name], data_dir)
    dataset_builder.build()
else:
    print("DATASET NOT FOUND")
    print("Available datasets are: %s ".format(_dict.keys()))