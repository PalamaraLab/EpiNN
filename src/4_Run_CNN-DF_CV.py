import os
# reduce number of threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
import matplotlib.pyplot as plt
import PIL
import numpy as np
import math
import csv
import sklearn.metrics as metrics
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import keras
from keras import layers
from keras import Model
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
import sys
from os import walk
import pandas as pd
from sklearn.model_selection import KFold


prjname=sys.argv[1]
print (prjname)
max_number_cv=int(sys.argv[2])
print (max_number_cv)

#Color - optimal: 100k
param_grid = {'image_size': [1000],
              'dropout_rate' : [0.1],
              'learning_rate': [0.001],
              'number_of_epochs': [10],
              'size_of_batch': [64]}
grid = ParameterGrid(param_grid)


def makedata(dfNeg, dfPos):
    #Train - Val - Test: 70 - 15 - 15

    #Test:
    dfTestNeg = dfNeg.sample(frac=0.1, replace=False, random_state=42)
    dfTestPos = dfPos.sample(frac=0.1, replace=False, random_state=42)

    #Neg:
    dfNeg=dfNeg[~dfNeg.index.isin(dfTestNeg.index)] #subtracting val from df
    dfPos=dfPos[~dfPos.index.isin(dfTestPos.index)] 

    dfTrainVal= pd.concat([dfNeg, dfPos])



#    dfTrain = pd.concat([dfTrainNeg, dfTrainPos])
    dfTest = pd.concat([dfTestNeg, dfTestPos])
#    dfVal = pd-concat(dfValNeg, dfVaPos)
    return(dfTrainVal, dfTest)


#Workflow
def run_CNN(train_dir, validation_dir, image_size, dropout_rate, learning_rate, number_of_epochs, size_of_batch):
    model = build_CNN(dropout_rate, learning_rate, image_size)
    history = preprocess_data_train(image_size, size_of_batch, number_of_epochs, train_dir, validation_dir, model)
    report, train_accuracy, test_accuracy, confusion_matrix, precisionCurve, recallCurve, precision, recall, prauc, bce =evaluate_CNN(history, model, image_size, dropout_rate, learning_rate, size_of_batch, number_of_epochs, train_dir, validation_dir)
    return (test_accuracy, precisionCurve, recallCurve, precision, recall, prauc, bce)


#Build CNN
def build_CNN(dropout_rate, learning_rate, image_size):
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(image_size, 111, 3))

    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(16, 3, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout rate
    x = layers.Dropout(dropout_rate)(x)

    # Create output layer with a single node and sigmoid activation
    output = layers.Dense(1, activation='sigmoid')(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully 
    # connected layer + sigmoid output layer
    model = Model(img_input, output)
    
    #model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])
    
    return model


#Preprocess Data:
def preprocess_data_train(image_size, size_of_batch, number_of_epochs, train_dir, validation_dir, model):
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1./255)
    #test_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=dfTrain,
        directory=".",
        x_col="id",
        y_col="label",
        target_size=(image_size, 111),  # All images will be resized to 150x150
        batch_size=size_of_batch,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=dfVal,
        directory=".",
        x_col="id",
        y_col="label",
        target_size=(image_size, 111),
        batch_size=size_of_batch,
        class_mode='binary')

    #test_generator = test_datagen.flow_from_dataframe(
    #    dataframe=dfTest,
    #    directory=".",
    #    x_col="id",
    #    y_col="label",
    #    target_size=(image_size, 111),
    #    batch_size=size_of_batch,
    #    class_mode='binary')
 
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.floor(len(dfTrain)/size_of_batch),  # 2000 images = batch_size * steps
        epochs=number_of_epochs,
        validation_data=validation_generator,
        validation_steps=math.floor(len(dfVal)/size_of_batch), # 1000 images = batch_size * steps
        verbose=0)
    model.save(prjname + '/output/' + prjname + '.h5')
    return (history)


#Evaluate CNN:
def evaluate_CNN(history, model, image_size, dropout_rate, learning_rate, size_of_batch, number_of_epochs, train_dir, validation_dir):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_accuracy = acc[-1]
    test_accuracy = val_acc[-1]

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    
    predict_validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=dfVal,
        directory=".",
        x_col="id",
        y_col="label",
        target_size=(image_size, 111),
        batch_size=1,
        shuffle=False,
        class_mode='binary')    
    
    
    predictions = model.predict_generator(predict_validation_generator, steps=math.floor(len(dfVal)))

    predicted_classes = (predictions > 0.5).astype(np.int)
    true_classes = predict_validation_generator.classes
    class_labels = list(predict_validation_generator.class_indices.keys())
    
    
    report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
    
    precisionCurve, recallCurve, thresholods= precision_recall_curve(true_classes, predictions)
    
    precision = average_precision_score(true_classes, predicted_classes)
    recall= recall_score(true_classes, predicted_classes, average='macro')
    
    fpr, tpr, thresholds = metrics.roc_curve(true_classes, predicted_classes, pos_label=1)
    prauc = metrics.auc(fpr, tpr)

    bce=log_loss(true_classes, predictions)
#    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#    bce = bce(np.ravel(true_classes), np.ravel(predicted_classes)).numpy()
    print (bce)

    with open(prjname + '/cv/' + str(max_number_cv) +  '_output.csv', 'a')as output:
        writer = csv.writer(output, delimiter='\t', lineterminator='\n',  quotechar='"')
        row = [image_size, dropout_rate, learning_rate, number_of_epochs, size_of_batch, test_accuracy, prauc, precision, recall, precisionCurve, recallCurve, thresholds, bce]
        writer.writerow(row)
    
    return (report, train_accuracy, test_accuracy, confusion_matrix, precisionCurve, recallCurve, precision, recall, prauc, bce)

#RUN CNN:
Negative = []
for (dirpath, dirnames, filenames) in walk(prjname + '/' + prjname + '_nochunks/Negative'):
    Negative.extend(filenames)
    break
Negative = [prjname + '/' + prjname + '_nochunks/Negative/' + s for s in Negative]
NegVal= ["0" for s in Negative]
dfNeg = pd.DataFrame(list(zip(Negative, NegVal)),
               columns =['id', 'label'])

Positive = []
for (dirpath, dirnames, filenames) in walk(prjname + '/' + prjname + '_nochunks/Positive'):
    Positive.extend(filenames)
    break
Positive = [prjname + '/' + prjname + '_nochunks/Positive/' + s for s in Positive]
PosVal= ["1" for s in Positive]
dfPos = pd.DataFrame(list(zip(Positive, PosVal)),
               columns =['id', 'label'])

for params in grid:
    kfold = KFold(n_splits=max_number_cv, shuffle=True, random_state=42)
    dfTrainVal, dfTest = makedata(dfNeg, dfPos)
    for train_index, test_index in kfold.split(dfTrainVal):
        dfTrain = dfTrainVal.iloc[train_index]
        dfVal =  dfTrainVal.iloc[test_index]
        test_accuracy, precisionCurve, recallCurve, precision, recall, prauc, bce = run_CNN(dfTrain, dfVal, params['image_size'], params['dropout_rate'], params['learning_rate'], params['number_of_epochs'], params['size_of_batch'])
