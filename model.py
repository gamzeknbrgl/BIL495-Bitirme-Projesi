from __future__ import print_function
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
#import libraries
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



with open('settings/settings.json') as file:    
	folder = json.load(file)

#settings for paths
test_size 		= folder["test_size"]
seed 			= folder["seed"]
deep_features 	= folder["deep_features"]
labels_path 	= folder["labels_path"]
results 		= folder["results"]
classifier_path = folder["classifier_path"]
train_path 		= folder["train_path"]
num_classes 	= folder["num_classes"]
classifier_path = folder["classifier_path"]

h5f_data  = h5py.File(deep_features, 'r')
h5f_label = h5py.File(labels_path, 'r')
features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']
features = np.array(features_string)
labels   = np.array(labels_string)
h5f_data.close()
h5f_label.close()

(dataset, test, dataset_labels, test_labels) = train_test_split(np.array(features),np.array(labels),test_size=test_size,random_state=seed)

#logistic regression model
model = LogisticRegression(random_state=seed)
model.fit(dataset, dataset_labels)

#results
file = open(results, "w")
for (label, features) in zip(test_labels, test):
	
	predictions = model.predict_proba(np.atleast_2d(features))[0]
	predictions = np.argsort(predictions)[::-1][:5]


model_prediction = model.predict(test)
file.write("{}\n".format(classification_report(test_labels, model_prediction)))
file.close()
#classifier
pickle.dump(model, open(classifier_path, 'wb'))

labels = sorted(list(os.listdir(train_path)))

#confusion matrix
cm = confusion_matrix(test_labels, model_prediction)
sns.heatmap(cm,annot=True,cmap="Set2")
plt.show()