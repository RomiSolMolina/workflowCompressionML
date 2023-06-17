
import os

import numpy as np

import skimage.data
import skimage.transform
from skimage import io
from sklearn.utils import shuffle
import pandas as pd

from tensorflow.keras.utils import to_categorical


# def load_data(data_dir):

#     directories = [d for d in os.listdir(data_dir) 
#                    if os.path.isdir(os.path.join(data_dir, d))]
#     labels = []
#     images = []
#     for d in directories:
#         label_dir = os.path.join(data_dir, d)
#         file_names = [os.path.join(label_dir, f) 
#                       for f in os.listdir(label_dir) if f.endswith(".jpg")]
#         for f in file_names:
#             images.append(io.imread(f)/255.0)
#             labels.append(int(d))
#     return images, labels

import os
from PIL import Image
import numpy as np

def load_data(data_dir):
    labels = []
    images = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(".jpg"):
                    file_path = os.path.join(label_dir, file)
                    image = Image.open(file_path)
                    image = np.array(image) / 255.0
                    images.append(image)
                    labels.append(int(label))
    
    return images, labels

def loadDataset_2D(root_path, classLabels, rows, cols):

    datasetFolderName = root_path
   
    sourceFiles=[]
    X=[]
    Y=[]

    train_path = datasetFolderName + '/train/'
    validation_path = datasetFolderName + '/validation/'
    test_path = datasetFolderName + '/test/'

    # Load training, validation, and testing datasets.
    images_train, labels_train = load_data(train_path)
    images_validation, labels_validation = load_data(validation_path)
    images_test, labels_test = load_data(test_path)

    print("Label: {0}\n Total Images: {1}".format(len(set(labels_train)), len(images_train)))
    print("Label: {0}\n Total Images: {1}".format(len(set(labels_validation)), len(images_validation)))
    print("Label: {0}\n Total Images: {1}".format(len(set(labels_test)), len(images_test)))

    # Resize images
    images_train = ([skimage.transform.resize(image, (80, 80), mode='constant')
                    for image in images_train])
    
    images_validation = ([skimage.transform.resize(image, (80, 80), mode='constant')
                         for image in images_validation])

    images_test = ([skimage.transform.resize(image, (80, 80), mode='constant')
                    for image in images_test])


    labels_train = np.array(labels_train)
    images_train = np.array(images_train)

    labels_validation = np.array(labels_validation)
    images_validation = np.array(images_validation)

    labels_test = np.array(labels_test)
    images_test = np.array(images_test)

    y_train = to_categorical(labels_train)
    y_test = to_categorical(labels_test)

    print("Dataset loaded!")
    return images_train, images_validation, images_test, y_train, y_test



def preproc_dataset(signal_df):
    
    # Remove cols that does not belong to the signal shape
    UNUSED_COLUMNS = ['Energy', 'FCI']
    df = signal_df.drop(columns=UNUSED_COLUMNS)
   
   
    # Label in csv file corresponds to the signal class
    _LABEL_COLUMN = 'class'
    
    dfTest = pd.DataFrame()
    dfTrain = pd.DataFrame()
    
    # Divide dataset

    for k in range(0,2):
     
        df2 = df[df[_LABEL_COLUMN].isin([k])]
        
        df_tr = df2[:10000]
        df_t = df2[10001:10900]
        
        dfTrain = pd.concat([df_tr, dfTrain])   
        dfTest = pd.concat([df_t, dfTest])
    
    dfTrain = shuffle(dfTrain)
    dfTest = shuffle(dfTest)
       
    return dfTrain, dfTest


def loadDataset_1D(root_path, classLabels, samples):
    tmp = 0

    
    
    return tmp
