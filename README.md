# CLIP-VPT
[[Overleaf]]() [[Experiment]](https://docs.google.com/spreadsheets/d/1VdLHwpSc6WaDBDBAlwTeeVHV2-NfAFiPXYJk1-J5hxs)

Visual Prompting CLIP as Fine-Tuning method, guiding CLIP to adapt to new domain based on text supervision. 

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
