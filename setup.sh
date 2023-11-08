# Install all dependecies
pip install -r requirements.txt

# Downloading Datasets from Vtab
cd src/data

# prepare Caltech101 dataset
python prepare_vtab.py --dataset caltech101

# prepare Cifar100 dataset
python prepare_vtab.py --dataset cifar100

# prepare clevr dataset
python prepare_vtab.py --dataset clevr

# prepare dmlab dataset
python prepare_vtab.py --dataset dmlab

# prepare dtd dataset
python prepare_vtab.py --dataset dtd

# prepare eurosat dataset
python prepare_vtab.py --dataset eurosat

# prepare oxford_flowers dataset
python prepare_vtab.py --dataset oxford_flowers

# preapre oxford_pet dataset
python prepare_vtab.py --dataset oxford_pet

# prepare Pcam dataset
python prepare_vtab.py --dataset pcam

# prepare smallnorb dataset
python prepare_vtab.py --dataset smallnorb

# prepare svhn dataset
python prepare_vtab.py --dataset svhn

# prepare sun397 dataset
python prepare_vtab.py --dataset sun397

# prepare kitti dataset
python prepare_vtab.py --dataset kitti