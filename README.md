# Parameter Efficient Fine-Tuning for CLIP

An emprical study of parameter efficient fine-tuning for adapting CLIP to downstream tasks:
![peftclip_figure](https://github.com/Andy-LZH/peft4clip/assets/40489953/8ae5e6a6-d700-4fa4-971e-55560988e92a)


- [x] **Dataset**: [VTAB-1k](https://google-research.github.io/task_adaptation/)

- [ ] **Fine-tuning Strategy**

- [ ] **Backbone**


### Environment Setup 
All the code is tested on **python 3.9+**, **CUDA 11.7/12.0**
```bash
# create a new conda environment
conda create -n peft_clip python=3.9
conda activate peft_clip

pip install -r requirements.txt
```

Optional: Install and configure [wandb](https://wandb.ai/site) for logging and visualization.
```bash
pip install wandb
wandb login
```


### Supported Tasks(Dataset) and Backbone
See [prepare_vtab.md](src/data/prompt.md) and [prompt.md](src/data/prompt.md) for prepare and learn the dataset
| Task | Backbone
| :---: |  :---: |
| vtab-caltech101 | ViT-B32 |
| vtab-cifar100 | ViT-B16 |
| vtba-dtd | ViT-L14 |
| vtab-oxford_flowers | MetaCLIP-B32-400m |
| vtab-oxford_pet | MetaCLIP-B32-2.5B |
| vtab-svhn | MetaCLIP-B16-400M |
| vtab-sun397 | MetaCLIP-B16-2.5B |
| vtba-pcam |
| vtab-eurosat |
| vtab-resisc45 |
| vtab-clevr_count |
| vtab-clevr_distance |
| vtba-dmlab |
| vtab-kitti |
| vtab-dSprites_location |
| vtab-dSprites_orientation |
| vtab-smallnorb_azimuth |
| vtab-smallnorb_distance |

### Supported Strategy
| Model |
| :---: |
| [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter) |
| [CoOP](https://github.com/KaiyangZhou/CoOp) |
| [VPT-CLIP-Shallow](https://github.com/KMnP/vpt) |
| [VPT-CLIP-Deep](https://github.com/KMnP/vpt)|
| Continued |

### Running
```bash
python train.py \
      --data "<dataset_name>" \     # Specify the dataset(task) name from table in Supported Tasks
      --backbone "<backbone_name>" \ # Choose the backbone architecture from table in Supported backbone
      --model "<strategy_name>" \   # Define the strategy model from table in Supported Strategy
      --type "<inference_type>" \   # Set the inference type to either "vision" or "vision-language"
      --shots "<num_shots>" \       # Indicate the number of shots
      --seeds "<seed>"              # Provide the seed value for reproducibility
```

