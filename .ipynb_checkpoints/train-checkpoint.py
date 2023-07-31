from src.model.CLIP.vpt_clip import VisionPromptCLIP

from src.data.Rice_Image_Dataset.Rice import Rice_Dataset
from src.utils.utils import setup_clip
from torch.utils.data import DataLoader
import argparse
import clip
from tqdm import tqdm
from time import sleep

# main function to call from workflow
def main():
    # set up arg parser
    parser = argparse.ArgumentParser(description='Train Vision Prompt CLIP')
    # check cuda availability
    parser.add_argument('--model', type=str, default="ViT-B/32",
                        help='For Saving and loading the current Model')
    parser.add_argument('--device', type=str, default="cuda",
                        help='For Saving and loading the current Model')
    parser.add_argument('--data', type=str, default="Rice_Image_Dataset",
                        help='For Saving and loading the current Model')
    
    args = parser.parse_args()

    # set up cfg and args
    backbone, preprocess, config, prompt_config = setup_clip(args)

    rice_dataset_test = Rice_Dataset(csv_file='src/data/Rice_Image_Dataset/test_meta.csv', root_dir='src/data/Rice_Image_Dataset/', transform=preprocess)
    rice_dataset_train = Rice_Dataset(csv_file='src/data/Rice_Image_Dataset/train_meta.csv', root_dir='src/data/Rice_Image_Dataset/', transform=preprocess)

    # define data loaders
    train_loader, test_loader = DataLoader(rice_dataset_train, batch_size=64, shuffle=True), DataLoader(rice_dataset_test, batch_size=1, shuffle=True)
    img_size = rice_dataset_test.__getitem__(0)[0].shape[1]
    num_classes = len(rice_dataset_test.classes)

    model = VisionPromptCLIP(backbone=backbone, config=config, prompt_config=prompt_config, img_size=img_size, num_classes=num_classes)

    # TODO: encapsulate into trainer
    model.train()
    for img, label, idx in tqdm(train_loader):
        raw_feature, clip_feature = model(img)
        sleep(0.1)

if __name__ == '__main__':
    main()
