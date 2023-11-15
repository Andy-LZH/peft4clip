# Parameter Efficient Fine-Tuning for CLIP

A systematic study of parameter efficient fine-tuning for CLIP, including the following aspects:
- [x] **Dataset**: [Vtab-1k](https://google-research.github.io/task_adaptation/)

- [ ] **Fine-tuning Strategy**

- [ ] **Backbone**


#### Environment Setup 
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

#### Supported Tasks(Dataset) and Backbone
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
| vtab-smallnorb_azimuth |

#### Supported Strategy
| Model |
| :---: |
| [CLIP-Adapter]() |
| [CoOP]() |
| [VPT-CLIP-Shallow]() |
| [VPT-CLIP-Deep] |
| Continued |

#### Running
```bash
python train.py \
      --data "<dataset_name>" \     # Specify the dataset(task) name from table in Supported Tasks
      --backbone "<backbone_name>" \ # Choose the backbone architecture from table in Supported backbone
      --model "<strategy_name>" \   # Define the strategy model from table in 
      --type "<inference_type>" \   # Set the inference type
      --shots "<num_shots>" \       # Indicate the number of shots
      --seeds "<seed>"              # Provide the seed value for reproducibility
```

