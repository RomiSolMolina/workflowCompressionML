

from src.topology.modelTeacher_2D import *

def teacherTrainingAfterBPO_2D(bestHP, xTrain, xTest, yTrain, yTest, lr):

    # Topology for the teacher model to be trained after hyperparameters optimization - 1D signal
    model = topologyTeacher_2D(bestHP)


    adam = Adam(lr)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    
  
    history  = model.fit(x=xTrain, y=yTrain,
                  validation_data=(xTest, yTest), 
                  batch_size = 128,
                  epochs=32,
                  verbose=1
                  )



    return model