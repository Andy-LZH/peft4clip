import tensorflow_datasets as tfds
import argparse

_dict = {
    "vtab-caltech101": "caltech101:3.0.1",
    "vtab-cifar100": "cifar100:3.*.*",
    "vtab-clevr": "clevr:3.*.*",
    "vtab-dmlab": "dmlab:2.0.1",
    "vtab-dsprites": "dsprites:2.*.*",
    "vtab-dtd": "dtd:3.*.*",
    "vtab-eurosat": "eurosat:2.*.*",
    "vtab-oxford_flowers": "oxford_flowers102:2.*.*",
    "vtab-oxford_pet": "oxford_iiit_pet:3.*.*",
    "vtab-pcam": "patch_camelyon:2.*.*",
    "vtab-smallnorb": "smallnorb:2.*.*",
    "vtab-svhn": "svhn_cropped:3.*.*",
    "vtab-sun397": "sun397:4.*.*",
    "vtab-kitti": "kitti:3.*.*",
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
    dataset_builder = builder(_dict[dataset_name], ".")
    dataset_builder.build()
else:
    print("DATASET NOT FOUND")
    print("Available datasets are: {}".format(_dict.keys()))
