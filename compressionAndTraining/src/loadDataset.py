
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sn

import skimage.data
import skimage.transform
from skimage import io

import tensorflow as tf 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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

    print("Label: {0}\n # Images: {1}".format(len(set(labels_train)), len(images_train)))
    print("Label: {0}\n # Images: {1}".format(len(set(labels_validation)), len(images_validation)))
    print("Label: {0}\n # Images: {1}".format(len(set(labels_test)), len(images_test)))

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



def preproc_dataset_1D(signal_df):
    
    # --- If needed, remove cols that does not belong to the signal shape
   

    # UNUSED_COLUMNS = ['Unnamed: 0']
    # df = signal_df.drop(columns=UNUSED_COLUMNS)
   
   # --- 

    # Label in csv file corresponds to the signal class
    _LABEL_COLUMN = 'cluster'
    
    dfTest = pd.DataFrame()
    dfTrain = pd.DataFrame()
 
    
    # Divide dataset

    for k in range(0,4):
        df2 = signal_df[signal_df[_LABEL_COLUMN].isin([k])]

        df_tr = df2[:45000]
        df_t = df2[45001:49000]
       
        dfTrain = pd.concat([df_tr, dfTrain])   
        dfTest = pd.concat([df_t, dfTest])
    
    dfTrain = shuffle(dfTrain)
    dfTest = shuffle(dfTest)


    return dfTrain, dfTest


def loadDataset_1D(root_path, nLabels, samples):
    tmp = 0

    all_files = glob.glob(root_path + "/csv/*.csv")
    li = []

    df = pd.DataFrame()

    dfTrain = pd.DataFrame()
    dfTest = pd.DataFrame()

    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Append all the .csv files stored in the folder root_path
    for filename in all_files:
        signal_df_complete = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    
### ---------------------------------
    SIGNAL_SAT_1 = root_path+"/sat/df_sub_raw_c2_sat1.csv"
    SIGNAL_SAT_2 = root_path+"/sat/df_sub_raw_c2_sat2.csv"
    signal_SAT_1 = pd.read_csv(SIGNAL_SAT_1)
    signal_SAT_2 = pd.read_csv(SIGNAL_SAT_2)

    signal_df = pd.concat(li, axis=0, ignore_index=True)
    signal_df_sat = pd.concat([signal_SAT_1, signal_SAT_2])

    UNUSED_COLUMNS = ['Unnamed: 0']
    signal_df = signal_df.drop(columns=UNUSED_COLUMNS)
    signal_df_sat = signal_df_sat.drop(columns=UNUSED_COLUMNS)

    signal_df_original = signal_df
    signal_df_cluster = signal_df

    signal_df = signal_df.astype(int) 
    
    signal_df_sat['cluster'] = signal_df_sat['cluster'].replace([2],3)

    
    signal_df_sat = signal_df_sat.astype(int) 

    signal_df['cluster'] = signal_df['cluster'].replace([6],0)
    signal_df['cluster'] = signal_df['cluster'].replace([5],1)
    signal_df['cluster'] = signal_df['cluster'].replace([2],2)

    signal_df_complete = pd.concat([signal_df, signal_df_sat])
    
### ---------------------------------
    # Plot distribution per class
    # sn.countplot(x = 'cluster', data = df)

    dfTrain, dfTest = preproc_dataset_1D(signal_df_complete)
    
    y = dfTrain.cluster
    x = dfTrain.drop('cluster',axis=1)

    yTest_df_Final = dfTest.cluster
    xTest_df_Final = dfTest.drop('cluster',axis=1)

    ## Split dataset into train and validation (test obtained from manual division) 
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = seed)
   
    yTrain = to_categorical(yTrain)
    yTest = to_categorical(yTest)
    yTest_Final = to_categorical(yTest_df_Final, nLabels)

    print(len(x))
    print(len(y))


    return xTrain, xTest, xTest_df_Final, yTrain, yTest, yTest_Final
