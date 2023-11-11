# CLIP-Adapter

for num_shot in 2 4 8 16

do
    for type in "vision" "vision-language"
    do
    # Caltech101
    # python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type

    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "ViT-B32" --shots $num_shot --type $type 
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "ViT-B16" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "ViT-L14" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type

    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "ViT-B32" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "ViT-B16" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "ViT-L14" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type
    # python3 train.py --data "vtab-caltech101" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type

    # vtab-pcam
    # python3 train.py --data "vtab-pcam" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-pcam" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # kitti
    # python3 train.py --data "vtab-kitti" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Shallow" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Shallow" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Shallow" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Deep" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Deep" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Deep" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-kitti" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True


    # svhncropped
    # python3 train.py --data "vtab-svhncropped" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-svhncropped" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # Oxford_Pets
    # python3 train.py --data "vtab-oxford_pet" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    # python3 train.py --data "vtab-oxford_pet" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # cifar100
    python3 train.py --data "vtab-cifar100" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-cifar100" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # dtd
    python3 train.py --data "vtab-dtd" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-dtd" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    # Oxford_flowers
    python3 train.py --data "vtab-oxford_flowers" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Shallow" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Shallow" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Shallow" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Shallow" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Deep" --backbone "ViT-B32" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Deep" --backbone "ViT-B16" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Deep" --backbone "ViT-L14" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type --wandb True
    python3 train.py --data "vtab-oxford_flowers" --model "VPT-CLIP-Deep" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type --wandb True

    done

done