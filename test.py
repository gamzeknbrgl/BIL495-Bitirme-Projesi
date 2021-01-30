from __future__ import print_function

#tkinter imports
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile
import os

# keras imports
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

# other imports
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import pickle
from PIL import Image 


root=Tk()
root.geometry("750x720")
root.resizable(width=True,height=True)


#logo
logo = Image.open('logo1.png')
logo=logo.resize((200,170), Image.ANTIALIAS)
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(column=2, row=0)
logo_label.pack( anchor=N)


def openfn():
    global filename
    filename=filedialog.askopenfilename(title='Select Image')
    
    return filename

def clear():
    global main, root
    main.destroy()
    main=Frame(root)
    main.pack()


def open_img():
    global main, root
    main.destroy()
    main=Frame(root)
    main.pack()
    x=openfn()
    print(x)
    img=Image.open(x)
    img=img.resize((400,400), Image.ANTIALIAS)
    img=ImageTk.PhotoImage(img)
    panel=Label(main,image=img)
    panel.image=img
    panel.pack(anchor=S)

def predict():

    img2=Image.open(filename)
    with open('settings/settings.json') as f:    
        config = json.load(f)

    m_name      = config["model"]
    weights         = config["weights"]
    include_top     = config["include_top"]
    train_path      = config["train_path"]
    test_path       = config["test_path"]
    deep_features   = config["deep_features"]
    labels_path     = config["labels_path"]
    test_size       = config["test_size"]
    results         = config["results"]
    model_path      = config["model_path"]
    seed            = config["seed"]
    classifier_path = config["classifier_path"]


    classifier = pickle.load(open(classifier_path, 'rb'))

    classifier = pickle.load(open(classifier_path, 'rb'))

    if m_name == "xception":
        pre_trained = Xception(weights=weights)
        model = Model(pre_trained.input, pre_trained.get_layer('avg_pool').output)
        image_size = (299, 299)
    else:
        pre_trained = None

    
    train_labels = os.listdir(train_path)

    img         = img2
    x           = image.img_to_array(img)
    x           = np.expand_dims(x, axis=0)
    x           = preprocess_input(x)
    feature     = model.predict(x)
    flat        = feature.flatten()
    flat        = np.expand_dims(flat, axis=0)
    preds       = classifier.predict(flat)
    prediction  = train_labels[preds[0]]
    print ("" + train_labels[preds[0]])

    x = train_labels[preds[0]].split("#")
    x = [item.replace(",", " #") for item in x]
    explanation = "Recommended Hashtags:\n #" +" #".join(x)
    w2 =tk.Label(main, justify=tk.CENTER,padx = 100, text=explanation)
    w2.config(font=("Courier", 12,'bold'),fg="DarkOrchid4")
    w2.pack()

main=Frame(root)
main.pack(side=tk.LEFT,padx=10)
root.title("Instagram Hashtag Recommender")

button=Button(root,text='Select Image',command=open_img,font=("Courier",10,'bold'), bg="MediumOrchid4", fg="white", width=15)
button.pack()
button1=Button(root,text='Generate Hashtags',command=predict,font=("Courier",10,'bold'), bg="MediumOrchid4", fg="white", width=20)
button1.pack()
button3= Button(root, text="Exit", command=lambda:exit(),font=("Courier",10,'bold'), bg="MediumOrchid4", fg="white", width=10)
button3.pack()


root.mainloop()