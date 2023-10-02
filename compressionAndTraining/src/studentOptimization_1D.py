

from src.topology.studentOptimization_1D import *


def studentBO_1D(xTrain, xTest, yTrain, yTest, teacher_baseline, N_ITERATIONS_STUDENT):
    callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, verbose=1),
            ]  
    callbacks.append(pruning_callbacks.UpdatePruningStep())

    OUTPUT_PATH = "tuner"

    if (os.path.exists(OUTPUT_PATH) == 'True'):
        shutil.rmtree(OUTPUT_PATH, ignore_errors = True)

    studentCNN_ = Distiller(student=build_model_QK_student_1D, teacher=teacher_baseline)
        
    tuner = kt.BayesianOptimization(
        studentCNN_.student,
        objective = "val_accuracy",
        max_trials = N_ITERATIONS_STUDENT,
        seed = 49,
        directory = OUTPUT_PATH
    )

    tuner.search(

        x=xTrain, y=yTrain,
        validation_data = (xTest, yTest),
        batch_size = 32,
        callbacks = [callbacks],
        epochs = 32
    )


    tuner.get_best_hyperparameters(num_trials=1)[0] 
   
    bestHP = tuner.get_best_hyperparameters()[0]
    


    return bestHP
