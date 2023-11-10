import os
import argparse
from src.model.CLIP_VPT.VisionPromptCLIP import VisionPromptCLIP
from src.utils.utils import setup_model
from src.engine.engines import Engine


# main function to call from workflow
def main():
    # set up arg parser
    parser = argparse.ArgumentParser(description="Train Vision Prompt CLIP")
    # check cuda availability
    parser.add_argument(
        "--model",
        type=str,
        default="VPT-CLIP-Shallow",
        help="For Saving and loading the current Model",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-B16",
        help="For Saving and loading the current Model",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="vtab-caltech101",
        help="For Saving and loading the current Model",
    )

    parser.add_argument(
        "--type",
        type=str,
        default="vision",
        help="Specify the type of inference, vision or vision-language",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=8,
        help="Specify the number of shots",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Specify the seed",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="For Saving and loading the current Model",
    )

    parser.add_argument(
        "--evluate",
        type=bool,
        default=False,
        help="Whether to train or not",
    )

    parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        help="Whether to save the model or not",
    )

    args = parser.parse_args()
    print(args)
    # set up cfg and args
    (
        model,
        train_loader,
        test_loader,
        dataset_config,
    ) = setup_model(args)

    # setup engine
    engine = Engine(
        model=model,
        device=args.device,
        train_loader=train_loader,
        test_loader=test_loader,
        configs=dataset_config,
    )

    # train the model
    model_path = "./src/logs/{}/{}/{}/epochs{}/model.pth".format(
        dataset_config.DATA.NAME,
        dataset_config.MODEL.TYPE,
        dataset_config.MODEL.BACKBONE,
        dataset_config.SOLVER.TOTAL_EPOCH,
    )

    if not args.evluate or not os.path.exists(model_path):
        # evluate the model
        engine.train(save_model=args.save_model)
        # engine.evaluate()

    else:
        # evaluate the model
        engine.evaluate()


if __name__ == "__main__":
    main()
