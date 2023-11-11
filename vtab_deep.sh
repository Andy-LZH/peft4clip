# check if vtab-caltech101 exists
python train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "ViT-B32"
python train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "ViT-B16"
python train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "ViT-L14"

python train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "ViT-B32"
python train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "ViT-B16"
python train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "ViT-L14"

python train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "ViT-B32"
python train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "ViT-B16"
python train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "ViT-L14"

python train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "ViT-B32"
python train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "ViT-B16"
python train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "ViT-L14"

python train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "ViT-B32"
python train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "ViT-B16"
python train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "ViT-L14"

python train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "ViT-B32"
python train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "ViT-B16"
python train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "ViT-L14"

python train.py --data "vtab-eurosat" --model "VPT-CLIP-Deep" --backbone "ViT-B32"
python train.py --data "vtab-eurosat" --model "VPT-CLIP-Deep" --backbone "ViT-B16"
python train.py --data "vtab-eurosat" --model "VPT-CLIP-Deep" --backbone "ViT-L14"