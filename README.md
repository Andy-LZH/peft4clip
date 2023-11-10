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

#### Register Dataset and Backbone
Avaliable datasets to be registered:
| Dataset | # Classes 
| :---: | :---: |
| caltech101 | 101 |
| cifar100 | 100 |
| clevr | TODO |
| dmlab | TODO |
| dsprites | TODO |
| dtd | TODO |
| eurosat | TODO |
| oxford_flowers | TODO |
| oxford_pet | TODO |
| pcam | TODO |
| smallnorb | TODO |
| svhn | TODO |
| sun397 | TODO |
| kitti | TODO |

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
