# Auto-Adapter
[[Overleaf]](https://www.overleaf.com/project/645420d94a40fe5e86b5bcb5) [[Experiment Result]](https://docs.google.com/spreadsheets/d/1VdLHwpSc6WaDBDBAlwTeeVHV2-NfAFiPXYJk1-J5hxs)

M-Mux(Multimodality Multiplexer) is a complex neural network capable of determine bottlenecks of the foundation model, and adapt the corresponding component accordingly. 

## Installation
### install depdencies 
```
pip install -r requirements.txt
```

## Prepare Dataset
### Kaggle

#### Environment Setup
Follow [this](https://github.com/Kaggle/kaggle-api#readme) to prepare kaggle api. 

#### Install Rice_Image_Dataset
```
cd ./data
kaggle datasets download -d muratkokludataset/rice-image-dataset
unzip *.zip
```
