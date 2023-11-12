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

#### Datasets and Backbones
| Dataset | Prompts | Note
| :---: | :--- | :--- |
| vtab-caltech101 | [Link](/src/data/prompt.md)| |
| vtab-cifar100 | [Link](/src/data/prompt.md)| |
| vtab-clevr | [Link](/src/data/prompt.md) | Use **vtab-clevr_count** or **vtab-clevr_distance** in <dataset_name> |
| vtba-dmlab | [Link](/src/data/prompt.md#vtab-dmlab)| |
| vtab-dsprites | [Link](/src/data/prompt.md)| Use **vtab-dSprites_location** or **vtab-dSprites_orientation** in <dataset_name> |
| vtba-dtd | [Link](/src/data/prompt.md)| |
| vtab-eurosat | [Link](/src/data/prompt.md)| |
| vtab-oxford_flowers | [Link](/src/data/prompt.md)| |
| vtab-oxford_pet |[Link](/src/data/prompt.md)| |
| vtba-pcam | [Link](/src/data/prompt.md#vtab-pcam)| Tensorflow dataset sometime broken, submit issue when encountered |
| vtab-smallnorb | [Link](/src/data/prompt.md)| Use **vtab-smallnorb_azimuth** or **vtab-smallnorb_elevation** in <dataset_name> |
| vtab-svhn | [Link](/src/data/prompt.md)| |
| vtab-sun397 | [Link](/src/data/prompt.md)| Tensorflow dataset sometime broken, submit issue when encountered |
| vtab-kitti | [Link](/src/data/prompt.md#vtab-kitti)| |

```bash
# download dataset (pwd: src/data)
cd src/data
python prepare_vtab.py --data <Dataset>

# i.e download caltech101 dataset
python prepare_vtab.py --data caltech101
```

#### Running
```bash
python train.py --data <dataset_name> --backbone <backbone_name> --model <strategy_name> --type <inferece_type> --shots <num_shots> --seeds <seed>
```
