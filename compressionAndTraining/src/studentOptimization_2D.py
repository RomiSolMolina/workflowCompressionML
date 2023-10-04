
from compressionAndTraining.src.topology.student_HPO_2D import *
from src.config import *

def studentBO_2D(images_train, y_train, images_test, y_test, teacher_baseline, N_ITERATIONS_STUDENT):
    callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, verbose=1),
            ]  
    callbacks.append(pruning_callbacks.UpdatePruningStep())

    

    if (os.path.exists(OUTPUT_PATH_STUDENT) == 'True'):
        shutil.rmtree(OUTPUT_PATH_STUDENT, ignore_errors = True)

    studentCNN_ = Distiller(student=build_model_QK_student, teacher=teacher_baseline)
        
    tuner = kt.BayesianOptimization(
        studentCNN_.student,
        objective = "val_accuracy",
        max_trials = N_ITERATIONS_STUDENT,
        seed = 49,
        directory = OUTPUT_PATH_STUDENT
    )

    tuner.search(

        x=images_train, y=y_train,
        validation_data=(images_test, y_test),
        batch_size= BATCH_STUDENT,
        callbacks=[callbacks],
        epochs= EPOCHS_STUDENT
    )


    tuner.get_best_hyperparameters(num_trials=1)[0] 
    
    bestHP = tuner.get_best_hyperparameters()[0]


    return bestHP