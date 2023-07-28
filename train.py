from model.CLIP.vpt_clip import VisionPromptCLIP
from model.vpt.src.configs.vit_configs import get_b32_config
from model.vpt.src.configs.config import get_cfg
from data.Rice_Image_Dataset.Rice import Rice_Dataset
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
from time import sleep

# main function to call from workflow
def main():
    backbone, preprocess = clip.load("ViT-B/32", device="cuda")
    config = get_b32_config()
    prompt_config = get_cfg().MODEL.PROMPT
    prompt_config.PROJECT = 768
    rice_dataset_test = Rice_Dataset(csv_file='data/Rice_Image_Dataset/test_meta.csv', root_dir='data/Rice_Image_Dataset/', transform=preprocess)
    rice_dataset_train = Rice_Dataset(csv_file='data/Rice_Image_Dataset/train_meta.csv', root_dir='data/Rice_Image_Dataset/', transform=preprocess)

    # define data loaders
    train_loader, test_loader = DataLoader(rice_dataset_train, batch_size=64, shuffle=True), DataLoader(rice_dataset_test, batch_size=1, shuffle=True)
    img_size = rice_dataset_test.__getitem__(0)[0].shape[1]
    num_classes = len(rice_dataset_test.classes)

    model = VisionPromptCLIP(backbone=backbone, config=config, prompt_config=prompt_config, img_size=img_size, num_classes=num_classes)

    # TODO: encapsulate into trainer
    model.train()
    for img, label, idx in tqdm(train_loader):
        raw_feature, clip_feature = model(img)
        sleep(1)
    # TODO: encapsulate into evlauator
    # model.eval()
main()