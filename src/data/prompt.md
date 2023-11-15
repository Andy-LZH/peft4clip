# Prompts for VTAB-1k

Follow what CLIP did in their repo, we as well share some of our manually crafted prompt and motivations here.

## [vtab-svhncropped](https://www.tensorflow.org/datasets/catalog/svhn_cropped)
```py
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]
template = "a photo of a digital number {}"
```

## [vtan-sun397](https://www.tensorflow.org/datasets/catalog/sun397)
The SUN397 dataset is a popular image recognition benchmark composed of 108,754 images in 397 categories. The images are very diverse and often contain complex scenes with several objects (more than 10 per image) and/or contextual details. The dataset is divided into 3 subsets: 50,000 for training, 5,000 for validation and 53,754 for testing.
see [here](datasets/prompts/sun397/classes.txt) for the full list of classes.
```py
template = "a photo of a {}"
```

## [vtab-pcam](https://www.tensorflow.org/datasets/catalog/patch_camelyon)
The PatchCamelyon benchmark is a new and challenging image classification dataset. It consists of 327.680 color images (96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue. PCam provides a new benchmark for machine learning models: bigger than CIFAR10, smaller than Imagenet, trainable on a single GPU.
```py
classes = [
    "normal lymph node tissue",
    "lymph node metastasis"
]
template = "a photo of {}"
```

## [vtab-resisc45](https://www.tensorflow.org/datasets/catalog/resisc45)
The dataset contains 31,500 images in 45 classes, with 700 images per class. The dataset is divided into 3 subsets: 25,000 for training, 1,000 for validation and 5,500 for testing. The images are in high resolution (256x256 pixels) in .tif format. The dataset is a subset of the UC Merced Land Use Dataset.
```py
classes = [
    "airplane",
    "airport",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular_farmland",
    "cloud",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "medium_residential",
    "mobile_home_park",
    "mountain",
    "overpass",
    "palace",
    "parking_lot",
    "railway",
    "railway_station",
    "rectangular_farmland",
    "river",
    "roundabout",
    "runway",
    "sea_ice",
    "ship",
    "snowberg",
    "sparse_residential",
    "stadium",
    "storage_tank",
    "tennis_court",
    "terrace",
    "thermal_power_station",
    "wetland"
]
template = "satellite imagery of {}"
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
template = "a photo of {}"
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
template = "a photo with {}"
```

## [vtab-clevr_count](https://www.tensorflow.org/datasets/catalog/clevr)
Created as part of the VTAB benchmark. Count for the number of objects in the image.
```py
classes = [
    "three things",
    "four things",
    "five things",
    "six things",
    "seven things",
    "eight things",
    "nine things",
    "ten things"
]
template = "there are {} objects in the image"
```

## [vtab-clevr_distance](https://www.tensorflow.org/datasets/catalog/clevr)
Created as part of the VTAB benchmark. Distance to the closest object in the image.
```py
classes = [
    "0-8.0M",
    "8.0-8.5M",
    "8.5-9.0M",
    "9.0-9.5M",
    "9.5-10.0M",
    "10.0-100M",
]
template = "the closest object is {} away from the camera"
```

## [vtab-smallnorb_azimuth](https://www.tensorflow.org/datasets/catalog/smallnorb)
```py
classes = [
    "0",
    "20",
    "40",
    "60",
    "80",
    "100",
    "120",
    "140",
    "160",
    "180",
    "200",
    "220",
    "240",
    "260",
    "280",
    "300",
    "320",
    "340",
]

template = "the camera azimuth is {} degrees"
```

## [vtab-smallnorb_elevation](https://www.tensorflow.org/datasets/catalog/smallnorb)
```py
classes = [
    "30",
    "35",
    "40",
    "45",
    "50",
    "55",
    "60",
    "65",
    "70",
]
template = "the camera elevation is {} degrees"
```

## Full Template we used in VTAB-1k
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
    "vtab-clevr_count": "there are {} objects in the image",
    "vtab-clevr_distance": "the closest object is {} away from the camera",
    "vtab-dmlab": "a photo of {}",
    "vtab-kitti": "a photo with {}",
    "vtab-smallnorb_azimuth": "the camera azimuth is {} degrees",
    "vtab-smallnorb_elevation": "the camera elevation is {} degrees",
    "vtab-dSprites_location": "a photo of a {}",
    "vtab-dSprites_orientation": "a photo of a {}",
    "vtab-resisc45": "satellite imagery of {}",
}
```