import clip
import torch
import argparse
from src.model.CLIP_VPT.VisionPromptCLIP import VisionPromptCLIP
from src.utils.utils import setup_clip
from src.engine.engines import Engine


# main function to call from workflow
def main():
    # set up arg parser
    parser = argparse.ArgumentParser(description="Train Vision Prompt CLIP")
    # check cuda availability
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B32",
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
    parser.add_argument(
        "--evluate",
        type=bool,
        default=False,
        help="Whether to train or not",
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

    print(dataset_config)

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
        dataset_config=dataset_config,
        prompt_config=prompt_config,
        img_size=img_size,
        num_classes=num_classes,
        prompts=text_input,
    ).to(args.device)

    # setup engine
    engine = Engine(
        model=model,
        device=args.device,
        train_loader=train_loader,
        test_loader=test_loader,
        configs=dataset_config,
    )

    if not args.evluate:
        # evluate the model
        engine.train()

    # evaluate the model
    engine.evaluate()


if __name__ == "__main__":
    main()
