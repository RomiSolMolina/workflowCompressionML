# src/dataset_loader.py

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
from tensorflow.keras.datasets import cifar10

from src.config.config import DatasetConfig


def load_images_from_directory(data_dir, target_size=(80, 80)):
    labels, images = [], []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.lower().endswith(".jpg"):
                    path = os.path.join(label_dir, file)
                    image = np.array(Image.open(path)) / 255.0
                    image = resize(image, target_size, mode='constant')
                    images.append(image)
                    labels.append(int(label))
    return np.array(images), np.array(labels)


def loadDataset_2D(root_path, classLabels, rows, cols):
    train_path = os.path.join(root_path, 'train')
    val_path = os.path.join(root_path, 'validation')
    test_path = os.path.join(root_path, 'test')

    images_train, labels_train = load_images_from_directory(train_path, (rows, cols))
    images_val, labels_val = load_images_from_directory(val_path, (rows, cols))
    images_test, labels_test = load_images_from_directory(test_path, (rows, cols))

    print(f"Train: {len(images_train)} images - {len(set(labels_train))} classes")
    print(f"Val: {len(images_val)} images - {len(set(labels_val))} classes")
    print(f"Test: {len(images_test)} images - {len(set(labels_test))} classes")

    y_train = to_categorical(labels_train)
    y_test = to_categorical(labels_test)

    return images_train, images_val, images_test, y_train, y_test


def loadDataset_CIFAR10():
    """
    Loads CIFAR-10 dataset and prepares it for training.
    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, DatasetConfig.nLabels_2D)
    y_test = to_categorical(y_test, DatasetConfig.nLabels_2D)

    # Split training into train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    print(f"CIFAR-10 loaded: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    return x_train, x_val, x_test, y_train, y_val, y_test


def preproc_dataset_1D(df, label_column='cluster', train_size=1500, test_size=50):
    df_train, df_test = pd.DataFrame(), pd.DataFrame()
    for k in range(DatasetConfig.nLabels_1D):
        class_data = df[df[label_column] == k]
        df_train = pd.concat([df_train, class_data.iloc[:train_size]])
        df_test = pd.concat([df_test, class_data.iloc[train_size:train_size + test_size]])
    return shuffle(df_train), shuffle(df_test)


def loadDataset_1D(root_path, nLabels, samples):
    """
    Loads a 1D dataset from subfolders with CSV files for each class.
    """
    all_data = []
    subdirs = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])

    for label_idx, subdir in enumerate(subdirs):
        subdir_path = os.path.join(root_path, subdir)
        csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))

        for f in csv_files:
            df = pd.read_csv(f).astype(int)
            df["cluster"] = label_idx
            all_data.append(df)

    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    df_train, df_test = preproc_dataset_1D(df_combined)

    x_train = df_train.drop('cluster', axis=1)
    y_train = to_categorical(df_train['cluster'], nLabels)

    x_test = df_test.drop('cluster', axis=1)
    y_test = to_categorical(df_test['cluster'], nLabels)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=0
    )

    print("1D dataset loaded from subfolders:", subdirs)
    return x_train, x_val, x_test, y_train, y_val, y_test
