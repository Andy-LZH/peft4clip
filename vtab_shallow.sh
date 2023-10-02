# check if vtab-caltech101 exists
python train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "ViT-B32"
python train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "ViT-B16"
python train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "ViT-L14"

python train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "ViT-B32"
python train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "ViT-B16"
python train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "ViT-L14"

python train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "ViT-B32"
python train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "ViT-B16"
python train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "ViT-L14"

python train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "ViT-B32"
python train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "ViT-B16"
python train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "ViT-L14"

python train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "ViT-B32"
python train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "ViT-B16"
python train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "ViT-L14"

python train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "ViT-B32"
python train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "ViT-B16"
python train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "ViT-L14"

python train.py --data "vtab-eurosat" --model "VPT-CLIP-Shallow" --backbone "ViT-B32"
python train.py --data "vtab-eurosat" --model "VPT-CLIP-Shallow" --backbone "ViT-B16"
python train.py --data "vtab-eurosat" --model "VPT-CLIP-Shallow" --backbone "ViT-L14"