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
from src.model.vpt.src.solver.lr_scheduler import WarmupCosineSchedule


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
    parser.add_argument(
        "--deep",
        type=bool,
        default=False,
        help="Whether to use deep prompt or not",
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

    # define optimizer and loss function, update only one parameter linear in the model
    prompt_parameters = (
        list(model.head.parameters())
        + list(model.prompt_dropout.parameters())
        + list(model.prompt_proj.parameters())
    )

    optimizer = torch.optim.AdamW(prompt_parameters, lr=1e-3, weight_decay=1e-5)

    max_epochs = 30
    warm_up_epochs = 10
    loss_fn = torch.nn.CrossEntropyLoss()

    torch.autograd.set_detect_anomaly(True)
    # scheduler = WarmupCosineSchedule(
    #     optimizer, warmup_steps=warm_up_epochs, t_total=max_epochs
    # )
    loss_step = []
    accuracy_step = []

    for epoch in range(warm_up_epochs + max_epochs):
        pbar = tqdm(train_loader)
        for img, label, idx in pbar:
            # mixed precision training
            with autocast():
                # calculate logits
                logits = model.linear_probe(img.to(args.device))
                assert logits.dtype == torch.float16

                # calculate loss
                loss = loss_fn(logits, label.to(args.device))
                loss = torch.sum(loss) / logits.shape[0]
                assert loss.dtype == torch.float32

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
                "Warmup Epoch [{}/{}] Loss: {:.4f} Accuracy: {:.2f}%".format(
                    epoch + 1,
                    warm_up_epochs,
                    loss.item(),
                    accuracy * 100,
                )
                if epoch < warm_up_epochs
                else "Epoch [{}/{}] Loss: {:.4f} Accuracy: {:.2f}%".format(
                    epoch + 1 - warm_up_epochs,
                    max_epochs,
                    loss.item(),
                    accuracy * 100,
                )
            )
        # update learning rate
        # scheduler.step()

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
