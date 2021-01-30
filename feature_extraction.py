
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import os
import json
import cv2
import h5py



with open('settings.json') as f:    
	configuration = json.load(f)

#settings for paths
m_name 		    = configuration["model"]
weights 		= configuration["weights"]
include_top 	= configuration["include_top"]
train_path 		= configuration["train_path"]
deep_features 	= configuration["deep_features"]
labels_path 	= configuration["labels_path"]
test_size 		= configuration["test_size"]
results 		= configuration["results"]
model_path 		= configuration["model_path"]

#use Xception model for feature extraction

if m_name == "xception":
	pre_trained = Xception(weights=weights)
	model = Model(pre_trained.input, pre_trained.get_layer('avg_pool').output)
	image_size = (299, 299)
else:
	pre_trained = None


model_labels = os.listdir(train_path)

#encoder for labels
label_encoder = LabelEncoder()
label_encoder.fit([tl for tl in model_labels])

features = []
labels   = []

#feature extaction with model
count = 1
for i, label in enumerate(model_labels):
	cur_path = train_path + "/" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		feature = model.predict(x)
		flat = feature.flatten()
		features.append(flat)
		labels.append(label)
	

label_encoder = LabelEncoder()
lenc_labels = label_encoder.fit_transform(labels)

#write features in the file
h5f_data = h5py.File(deep_features, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(lenc_labels))

h5f_data.close()
h5f_label.close()


model_json = model.to_json()
with open(model_path + str(test_size) + ".json", "w") as json_file:
	json_file.write(model_json)


model.save_weights(model_path + str(test_size) + ".h5")
