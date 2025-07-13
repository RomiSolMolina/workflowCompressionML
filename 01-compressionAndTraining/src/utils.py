import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical


def normalizationPix(train_images, test_images):
    """
    Normalize image pixel values to the range [0, 1].
    """
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    return train_images, test_images


def confusionMatrixPlot(model, x_test, y_test, class_names=None, normalize=True):
    """
    Plot the confusion matrix for a trained model.
    """
    # Predict class labels
    y_pred = model.predict(x_test)
    if y_pred.shape[1] > 1:  # One-hot encoded output
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
    else:  # Already label encoded
        y_pred_labels = (y_pred > 0.5).astype("int32")
        y_true_labels = y_test

    # Compute confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", cbar=False)
    
    if class_names:
        plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=45)
        plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
