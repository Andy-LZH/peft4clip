import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

files = []
labels = []

# creating dataframe from files and labels
df_train = pd.DataFrame(columns=['file', 'label'])
df_test = pd.DataFrame(columns=['file', 'label'])

# path to the folder containing the images
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        if(filename.endswith('.jpg')):
            print(dirname.split('/')[-1])
            files.append(os.path.join(dirname.split('/')[-1], filename))
            labels.append(dirname.split('/')[-1])

# unique dictionary of labels
dictionary = {}

print(len(files))
for i in range(len(files)):
    if labels[i] not in dictionary:
        dictionary[labels[i]] = [files[i]]
    else:
        dictionary[labels[i]].append(files[i])

x_train = []
y_train = []
x_test = []
y_test = []

for key in dictionary.keys():
    train, test = train_test_split(dictionary[key], test_size=0.2, random_state=42)
    x_train.extend(train)
    y_train.extend([key]*len(train))
    x_test.extend(test)
    y_test.extend([key]*len(test))

df_train['file'] = x_train
df_train['label'] = y_train

df_train.to_csv('train_meta.csv', index=True)

df_test['file'] = x_test
df_test['label'] = y_test

df_test.to_csv('test_meta.csv', index=True)


