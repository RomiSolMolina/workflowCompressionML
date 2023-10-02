

from src.modelTeacher import *
from src.topology import *

def teacherTrainingAfterBPO_1D(bestHP, xTrain, xTest, yTrain, yTest, lr):

    model = modelTeacherDefinition_1D(bestHP)

    adam = Adam(lr)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    
  
    history  = model.fit(x=xTrain, y=yTrain,
                  validation_data=(xTest, yTest), 
                  batch_size = 128,
                  epochs=32,
                  verbose=1
                  )



    return model