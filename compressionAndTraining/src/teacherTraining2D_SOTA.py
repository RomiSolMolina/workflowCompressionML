
from src.modelTeacher2D_SOTA import *

def teacherTrainingAfterBPO_SOTA(bestHP, xTrain, xTest, yTrain, yTest, lr):
    print("Tacher SOTA")

    bestHP = [64, 64, 64, 64, 64, 96, 32, 64, 20, 50, 20]
    model = modelTeacherDefinition_2D_SOTA(bestHP)

    adam = Adam(lr)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    
   
    history  = model.fit(x=xTrain, y=yTrain,
                  validation_data=(xTest, yTest), 
                  batch_size = 128,
                  epochs=32,
                  verbose=1
                  )



    return model