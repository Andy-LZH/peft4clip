import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        self.prompt_parameters = model.head.parameters()
        self.optimizer = torch.optim.AdamW(
            params=self.prompt_parameters,
            lr=configs.SOLVER.BASE_LR,
            weight_decay=configs.SOLVER.WEIGHT_DECAY,
        )

        ## setup loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        ## setup epochs
        self.max_epochs = configs.SOLVER.TOTAL_EPOCH
        self.warm_up_epochs = configs.SOLVER.WARMUP_EPOCH

        ## setup logging
        self.dataset_name = configs.DATA.NAME

    def train(self):
        """
        Train the model.
        """

        # for logging traning loss and accuracy
        predicted = []
        labels = []
        loss_step = []
        accuracy_step = []

        # train the model
        print("Training Vision Prompt CLIP...")
        print("Press Ctrl+C to stop training")

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.warm_up_epochs + self.max_epochs):
            pbar = tqdm(self.train_loader)
            for img, label, idx in pbar:
                # mixed precision training
                with autocast():
                    # calculate logits
                    logits = self.model(img.to(self.device))
                    assert logits.dtype == torch.float16

                    # calculate loss
                    loss = self.criterion(logits, label.to(self.device))
                    loss = torch.sum(loss) / logits.shape[0]
                    assert loss.dtype == torch.float32

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
            "./src/logs/{}/epochs{}/".format(self.dataset_name, self.max_epochs)
        ):
            os.makedirs(
                "./src/logs/{}/epochs{}/".format(self.dataset_name, self.max_epochs)
            )

        # save model
        torch.save(
            self.model.state_dict(),
            "./src/logs/{}/epochs{}/model.pth".format(
                self.dataset_name, self.max_epochs
            ),
        )
        # calculate training accuracy
        predicted = np.concatenate(predicted)
        labels = np.concatenate(labels)

        accuracy = (predicted == labels).mean()
        print(f"Train Accuracy = {accuracy}")

        # draw loss and accuracy using matplotlib
        plt.plot(loss_step)
        plt.title("Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(
            "./src/logs/{}/epochs{}/loss.png".format(self.dataset_name, self.max_epochs)
        )

        plt.plot(accuracy_step)
        plt.title("Accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.savefig(
            "./src/logs/{}/epochs{}/accuracy.png".format(
                self.dataset_name, self.max_epochs
            )
        )

    def evaluate(self):
        """
        Evaluate the model.
        """

        # test model
        self.model.eval()

        # check if logs folder or model.path exists
        if not os.path.exists("./src/logs"):
            raise ValueError("No logs folder found, please train the model first")

        if not os.path.exists(
            "./src/logs/{}/epochs{}/model.pth".format(
                self.dataset_name, self.max_epochs
            ),
        ):
            raise ValueError("No model found, please train the model first")

        self.model.load_state_dict(
            torch.load(
                "./src/logs/{}/epochs{}/model.pth".format(
                    self.dataset_name, self.max_epochs
                ),
                map_location=self.device,
            )
        )
        predicted = []
        labels = []
        print("Evaluating Vision Prompt CLIP...")
        print("Press Ctrl+C to stop evaluation")
        with torch.no_grad():
            for img, label, idx in tqdm(self.test_loader):
                with autocast():
                    logits = self.model.linear_probe(img.to(self.device))
                    _, indices = logits.max(1)
                    predicted.append(indices.cpu().numpy())
                    labels.append(label.cpu().numpy())

        # calculate accuracy
        accuracy = accuracy_score(labels, predicted)
        precision = precision_score(labels, predicted, average="macro")
        recall = recall_score(labels, predicted, average="macro")
        f1 = f1_score(labels, predicted, average="macro")

        # save results
        with open(
            "./src/logs/{}/epochs{}/results.txt".format(
                self.dataset_name, self.max_epochs
            ),
            "w",
        ) as f:
            f.write("Accuracy: {}\n".format(accuracy))
            f.write("Precision: {}\n".format(precision))
            f.write("Recall: {}\n".format(recall))
            f.write("F1: {}\n".format(f1))
