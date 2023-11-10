import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import sleep
import matplotlib.pyplot as plt
from loguru import logger


class Engine:
    """
    Trainer class for training the model.
    """

    def __init__(self, model, device, train_loader, test_loader, configs):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.configs = configs

        # setup hyperparameters
        ## setup optimizer

        ## setup loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        ## setup epochs
        self.max_epochs = configs.SOLVER.TOTAL_EPOCH
        self.warm_up_epochs = configs.SOLVER.WARMUP_EPOCH
        self.shots = configs.DATA.SHOTS

        ## setup logger
        self.dataset_name = configs.DATA.NAME
        self.model_name = configs.MODEL.TYPE
        self.type = configs.MODEL.TRANSFER_TYPE
        logger.info("Model: {}".format(self.model_name))

        # setup optimizer
        self.optimizer = self.model.build_optimizer(configs)
        self.text_encoder = self.model.model.encode_text

    def train(self, save_model: bool = False):
        """
        Train the model.
        """

        # for logger traning loss and accuracy
        predicted = []
        labels = []
        loss_step = []
        accuracy_step = []

        # train the model
        logger.info("Training Vision Prompt CLIP...")
        logger.info("Press Ctrl+C to stop training")

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.warm_up_epochs + self.max_epochs):
            pbar = tqdm(self.train_loader)
            for img, label in pbar:
                with autocast():
                    if self.type == "vision":
                        # calculate logits
                        logits = self.model(img.to(self.device))
                        assert logits.dtype == torch.float16

                    elif self.type == "vision-language":
                        logits = self.model.vision_language_forward(img.to(self.device))
                        assert logits.dtype == torch.float16

                    # calculate loss
                    loss = self.criterion(logits, label.to(self.device))
                    loss = torch.sum(loss) / logits.shape[0]
                    assert loss.dtype == torch.float32
                    # show label
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # find the highest logit for each image in the batch
                _, indices = logits.max(1)
                predicted.append(indices.cpu().numpy())
                labels.append(label.cpu().numpy())

                accuracy = (indices == label.to(self.device)).float().mean().item()

                # draw loss and accuracy
                loss_step.append(loss.item())
                accuracy_step.append(accuracy)

                # print loss and accuracy in each batch inside tqdm
                pbar.set_description(
                    "Warmup Epoch [{}/{}] Loss: {:.4f} Accuracy: {:.2f}%".format(
                        epoch + 1,
                        self.warm_up_epochs,
                        loss.item(),
                        accuracy * 100,
                    )
                    if epoch < self.warm_up_epochs
                    else "Epoch [{}/{}] Loss: {:.4f} Accuracy: {:.2f}%".format(
                        epoch + 1 - self.warm_up_epochs,
                        self.max_epochs,
                        loss.item(),
                        accuracy * 100,
                    )
                )

        # check if logs folder exists
        if not os.path.exists("./src/logs"):
            os.makedirs("./src/logs")

        if not os.path.exists(
            "./src/logs/{}/{}/shots{}/epochs{}/".format(
                self.dataset_name, self.model_name, self.shots, self.max_epochs
            )
        ):
            os.makedirs(
                "./src/logs/{}/{}/shots{}/epochs{}/".format(
                    self.dataset_name, self.model_name, self.shots, self.max_epochs
                )
            )

        self.base_dir = "./src/logs/{}/{}/shots{}/epochs{}/".format(
            self.dataset_name, self.model_name, self.shots, self.max_epochs
        )

        if save_model:
            # save model
            torch.save(
                self.model.state_dict(),
                self.base_dir + "model.pth".format(self.dataset_name, self.max_epochs),
            )

        # calculate training accuracy
        predicted = np.concatenate(predicted)
        labels = np.concatenate(labels)

        accuracy = (predicted == labels).mean()
        logger.info(f"Train Accuracy = {accuracy}")

        # draw loss and accuracy using matplotlib
        plt.plot(loss_step)
        plt.title("Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(
            self.base_dir + "loss.png".format(self.dataset_name, self.max_epochs)
        )

        plt.plot(accuracy_step)
        plt.title("Accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.savefig(
            self.base_dir + "accuracy.png".format(self.dataset_name, self.max_epochs)
        )

        # run evaluation
        self.evaluate(train_val_flag=True)

    def evaluate(self, train_val_flag: bool = False):
        """
        Evaluate the model.
        """

        # test model
        self.model.eval()

        if not train_val_flag:
            # check if logs folder or model.path exists
            if not os.path.exists("./src/logs"):
                raise ValueError("No logs folder found, please train the model first")

            if not os.path.exists(
                "./src/logs/{}/{}/epochs{}/model.pth".format(
                    self.dataset_name, self.model_name, self.max_epochs
                ),
            ):
                raise ValueError("No model found, please train the model first")

            self.model.load_state_dict(
                torch.load(
                    "./src/logs/{}/{}/shots{}/epochs{}/model.pth".format(
                        self.dataset_name, self.model_name, self.shots, self.max_epochs
                    ),
                    map_location=self.device,
                )
            )

        predicted = []
        labels = []
        logger.info("Evaluating Vision Prompt CLIP...")
        logger.info("Press Ctrl+C to stop evaluation")
        with torch.no_grad():
            for img, label in tqdm(self.test_loader):
                with autocast():
                    if self.type == "vision":
                        # calculate logits
                        logits = self.model(img.to(self.device))
                    elif self.type == "vision-language":
                        logits = self.model.vision_language_forward(img.to(self.device))
                _, indices = logits.max(1)
                predicted.append(indices.cpu().numpy())
                labels.append(label.cpu().numpy())
        # calculate accuracy
        predicted = np.concatenate(predicted)
        labels = np.concatenate(labels)

        accuracy = accuracy_score(labels, predicted)
        precision = precision_score(labels, predicted, average="macro")
        recall = recall_score(labels, predicted, average="macro")
        f1 = f1_score(labels, predicted, average="macro")

        logger.info("Accuracy: {}\n".format(accuracy))
        logger.info("Precision: {}\n".format(precision))
        logger.info("Recall: {}\n".format(recall))
        logger.info("F1: {}\n".format(f1))
