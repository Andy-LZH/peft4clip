# Prompts for VTab-1K

Follow what CLIP did in their repo, we as well share some of our manually crafted prompt and motivations here.

## Template we used
```py
_DATASET_TEMPLATE = {
        "vtab-oxford_flowers": "a photo of a {}",
        "vtab-caltech101": "a photo of a {}",
        "vtab-cifar100": "a photo of a {}",
        "vtab-dtd": "a photo of a {}",
        "vtab-eurosat": "a photo of a {}",
        "vtab-oxford_pet": "a photo of a {}",
        "vtab-pcam": "a photo of a {}",
        "vtab-svhncropped": "a photo of a digital number {}",
        "vtab-sun397": "a photo of a {}",
        "vtab-clevr_count": "a photo of a {}",
        "vtab-clevr_distance": "a photo of a {}",
        "vtab-dmlab": "this is {}",
        "vtab-kitti": "a photo with {}",
        "vtab-smallnorb_azimuth": "a photo of a {}",
        "vtab-smallnorb_elevation": "a photo of a {}",
        "vtab-dSprites_location": "a photo of a {}",
        "vtab-dSprites_orientation": "a photo of a {}",
    }
```

## [vtab-pcam](https://www.tensorflow.org/datasets/catalog/patch_camelyon)
The PatchCamelyon benchmark is a new and challenging image classification dataset. It consists of 327.680 color images (96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue. PCam provides a new benchmark for machine learning models: bigger than CIFAR10, smaller than Imagenet, trainable on a single GPU.
```py
classes = [
    "normal lymph node tissue",
    "lymph node metastasis"
]
```

## [vtab-dmlab](https://www.tensorflow.org/datasets/catalog/dmlab)

The Dmlab dataset consists of 360x480 color images in 6 classes. The classes are {close, far, very far} x {positive reward, negative reward} respectively.

```py
classes = [
    "a close object with postive reward",
    "a close object with negative reward",
    "a far object with postive reward",
    "a far object with negative reward"
    "a very far object with postive reward",
    "a very far object with negative reward"
]
```

## [vtab-kitti](https://www.tensorflow.org/datasets/catalog/kitti)
The dataset contains 7481 training images annotated with 3D bounding boxes. A full description of the annotations can be found in the readme of the object development kit readme on the Kitti homepage.
```py
classes = [
    "a car on my left or right side.",
    "a car nearby.",
    "a car in the distance.",
    "no car."
]
```