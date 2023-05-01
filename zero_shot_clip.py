import torch
import clip
from tqdm import tqdm
from data.Rice_Image_Dataset.Rice import Rice_Dataset
from torch.utils.data import DataLoader
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def print_metrics(preds: list, labels: list):
    print("Precision: ", precision_score(labels, preds, average='macro'))
    print("Recall: ", recall_score(labels, preds, average='macro'))
    print("F1: ", f1_score(labels, preds, average='macro'))
    print("Accuracy: ", accuracy_score(labels, preds))


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
# define model and preprocess method used
model, preprocess = clip.load("RN50", device=device)

rice_dataset_test = Rice_Dataset(csv_file='data/Rice_Image_Dataset/test_meta.csv', root_dir='data/Rice_Image_Dataset/', transform=preprocess)

prompt_file = open("data/Rice_Image_Dataset/prompt.json", "r")
prompt = json.load(prompt_file)['Prompt']
prompt_file.close()

# encode prompt
text = clip.tokenize(prompt).to(device)

# Zero-Shot test on test set
preds = []
labels = []
with torch.no_grad():
    for image, label, idx in tqdm(DataLoader(rice_dataset_test)):

        # No minibatching here, just single image, hence unsqueeze
        image = image.to(device)
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # cosine similarity as logits
        logits_per_image, logits_per_text = model(image, text)

        # flatten the logits when mini-batching
        pred = logits_per_image.argmax().cpu().numpy()
        label = label.cpu().numpy()
        preds.append(pred)
        labels.append(label)

print_metrics(preds, labels)


        

