
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

# Plot confusion matrix
def confusionMatrixPlot(model, xTest, yTest):

    plt.rcParams["figure.figsize"] = (7,3)

    # Input is tranformed to a list 
    xTest_List = list()
    xTest_List.append(xTest)

    # Get predictions
    yPrediction = np.argmax(model.predict(xTest_List), axis=-1)

    yTest_List = np.asarray(yTest)
    yTest_List = yTest_List.argmax(1)

    data = {'y_Actual':  yTest_List, 
            'y_Predicted': yPrediction
            }

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:,np.newaxis]

    print (confusion_matrix)

    sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 12}, cmap="BuPu", fmt = '.2%') # font size

    plt.show()
    