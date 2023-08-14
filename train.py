import os
import clip
import torch
import argparse
import numpy as np
from tqdm import tqdm
from time import sleep
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from src.model.vpt_clip.vpt_clip import VisionPromptCLIP
from src.utils.utils import setup_clip


# main function to call from workflow
def main():
    # set up arg parser
    parser = argparse.ArgumentParser(description="Train Vision Prompt CLIP")
    # check cuda availability
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="For Saving and loading the current Model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="For Saving and loading the current Model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="food-101",
        help="For Saving and loading the current Model",
    )

    args = parser.parse_args()
    print(args)
    # set up cfg and args
    (
        backbone,
        config,
        prompt_config,
        train_loader,
        test_loader,
        dataset_config,
    ) = setup_clip(args)

    print("Training Vision Prompt CLIP...")
    print("Press Ctrl+C to stop training")
    print(dataset_config)
    sleep(1)

    # construct text input
    text_input = torch.cat(
        [clip.tokenize(f"a photo of {c}") for c in dataset_config.DATA.CLASSES]
    ).to(args.device)

    # define data loaders
    img_size = dataset_config.DATA.CROPSIZE
    num_classes = dataset_config.DATA.NUMBER_CLASSES

    model = VisionPromptCLIP(
        backbone=backbone,
        config=config,
        prompt_config=prompt_config,
        img_size=img_size,
        num_classes=num_classes,
        prompts=text_input,
    )
    model = model.to(args.device)
    model.train()

    predicted = []
    labels = []

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    pbar = tqdm(train_loader)
    for epoch in range(1):
        loss_step = []
        accuracy_step = []
        for img, label, idx in pbar:
            # TODO check how this improve linear probe accuracy
            # dynamically cast to fp16 to save memory and comatible with clip

            # forward pass
            logits = model(img.to(args.device))
            logits = logits.float()
            print(logits.shape)
            print(label.shape)
            loss = loss_fn(logits, label.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # find the highest logit for each image in the batch
            _, indices = logits.max(1)
            predicted.append(indices.cpu().numpy())
            labels.append(label.cpu().numpy())

            accuracy = (indices == label.to(args.device)).float().mean().item()

            # draw loss and accuracy
            loss_step.append(loss.item())
            accuracy_step.append(accuracy)

            # print loss and accuracy in each batch inside tqdm
            pbar.set_description(
                "Epoch [{}/{}] Loss: {:.4f} Accuracy: {:.2f}%".format(
                    epoch + 1,
                    1,
                    loss.item(),
                    accuracy * 100,
                )
            )

    # calculate accuracy
    predicted = np.concatenate(predicted)
    labels = np.concatenate(labels)

    accuracy = (predicted == labels).mean()
    print(f"Train Accuracy = {accuracy}")

    # check if logs folder exists
    if not os.path.exists("./src/logs"):
        os.makedirs("./src/logs")

    # draw loss and accuracy using matplotlib
    plt.plot(loss_step)
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("./src/logs/loss_{}.png".format(args.data))

    plt.plot(accuracy_step)
    plt.title("Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.savefig("./src/logs/accuracy_{}.png".format(args.data))


if __name__ == "__main__":
    main()
