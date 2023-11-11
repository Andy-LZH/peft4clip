# CLIP-Adapter

for num_shot in 2 4 8 16

do
    for type in "vision" "vision-language"
    do
    # Caltech101
    python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot --type $type
    python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot --type $type
    python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot --type $type
    python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot --type $type
    python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot --type $type
    python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot --type $type
    python3 train.py --data "vtab-caltech101" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot --type $type

    # # patch_camelyon
    # python3 train.py --data "vtab-patch_camelyon" --model "CLIP-Adapter" --backbone "ViT-B32" --shots $num_shot
    # python3 train.py --data "vtab-patch_camelyon" --model "CLIP-Adapter" --backbone "ViT-B16" --shots $num_shot
    # python3 train.py --data "vtab-patch_camelyon" --model "CLIP-Adapter" --backbone "ViT-L14" --shots $num_shot
    # python3 train.py --data "vtab-patch_camelyon" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-400M" --shots $num_shot
    # python3 train.py --data "vtab-patch_camelyon" --model "CLIP-Adapter" --backbone "MetaCLIP-B32-2.5B" --shots $num_shot
    # python3 train.py --data "vtab-patch_camelyon" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-400M" --shots $num_shot
    # python3 train.py --data "vtab-patch_camelyon" --model "CLIP-Adapter" --backbone "MetaCLIP-B16-2.5B" --shots $num_shot
    done

done