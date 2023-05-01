import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data.Rice_Image_Dataset.Rice import Rice_Dataset



# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, idx in tqdm(DataLoader(dataset, batch_size=256, shuffle=True)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def print_metrics(preds: list, labels: list):
    print("Precision: ", precision_score(labels, preds, average='macro'))
    print("Recall: ", recall_score(labels, preds, average='macro'))
    print("F1: ", f1_score(labels, preds, average='macro'))
    print("Accuracy: ", accuracy_score(labels, preds))

# Load the dataset
train = Rice_Dataset(csv_file='data/Rice_Image_Dataset/train_meta.csv', root_dir='data/Rice_Image_Dataset/', transform=preprocess)
test = Rice_Dataset(csv_file='data/Rice_Image_Dataset/test_meta.csv', root_dir='data/Rice_Image_Dataset/', transform=preprocess)

train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression, training linear classifier on top of frozen features
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
print_metrics(predictions, test_labels)