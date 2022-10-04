# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 01:20:44 2022

@author: irfan
"""
from keras.regularizers import l2
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import keras.backend as K
from keras.optimizers import SGD, RMSprop, Adam
from PIL import Image
import pandas as pd
import numpy as np
from os import listdir
import os
from os.path import isfile, join
import h5py
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.xception import Xception


print("Loading metadata...")


 
def getDataframe(path_of_dir,get_class_from_file=False):
    
    class_to_ix = {}
    ix_to_class = {}
    dataframe = pd.DataFrame(columns = ["path"]) 
    if get_class_from_file:
        root_dir = "D:/PhD_CourseWork/deep_learning/food101";
        class_file_name = root_dir + '/meta/classes.txt'
        with open(class_file_name, 'r') as txt:
            classes = [l.strip() for l in txt.readlines()]
            class_to_ix = dict(zip(classes, range(len(classes))))
            ix_to_class = dict(zip(range(len(classes)), classes))
            class_to_ix = {v: k for k, v in ix_to_class.items()}
            print(classes)
    else:
        
        classes = [ f.path.replace('\\','/').split('/')[-1] for f in os.scandir(path_of_dir) if f.is_dir() ]
        
        for cls in classes:
            mypath = path_of_dir +'/'  + cls
            onlyfiles = list( [ mypath + '/'+f for f in listdir(mypath) if isfile(join(mypath, f))])
            dataframe = dataframe.append(onlyfiles,ignore_index=True)
            
        
        class_to_ix = dict(zip(classes, range(len(classes))))
        ix_to_class = dict(zip(range(len(classes)), classes))
        class_to_ix = {v: k for k, v in ix_to_class.items()}
        print(classes)
        del dataframe['path']
        dataframe.rename(columns = {0:'path'}, inplace = True)
    return dataframe,class_to_ix,ix_to_class


def getXceptionModel(output_feat = 21):
    Xception_net = Xception(weights='imagenet', include_top=False,input_tensor=Input(shape=(299, 299, 3)))
    x = Xception_net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256,activation='relu')(x)
    # x = Dropout(0.4)(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(output_feat, activation='softmax')(x)
    
    model = Model(inputs=Xception_net.input, outputs=predictions)
    
    return model, 'Xception'

def getInception3Model(n_classes=21):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(n_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model, 'inception3'


def main(train_mode,epochs=10,batch_size=16, base = 'xception', input_size = (299,299)):
    
    tf.config.list_physical_devices('GPU') 
    tf.debugging.set_log_device_placement(True)
    
    data_set_root = "D:/PhD_CourseWork/deep_learning/food101/working"

    train_data_dir = data_set_root + '/train_21'
    test_data_dir = data_set_root + '/test_21'
    
    if base == 'xception':
        input_size = (299,299)
    else:
        input_size = (299,299)
    
    
    train_df,class_to_ix,ix_to_class = getDataframe(train_data_dir)
    test_df,_, _ = getDataframe(test_data_dir)
    
    train_df['label'] = train_df['path'].map(lambda x: spliter(data = x, class_or_id = 'Class')) 
    train_df['idx'] = train_df['path'].map(lambda x: spliter(x)) 
    
    test_df['label'] = test_df['path'].map(lambda x: spliter(x, 'class')) 
    test_df['idx'] = test_df['path'].map(lambda x: spliter(x)) 
    
    
    datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_gen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_dataframe(dataframe = train_df, directory=None, x_col='path', y_col='label',
    weight_col=None, target_size=input_size, color_mode='rgb',
    classes=None, class_mode='categorical', batch_size=batch_size, shuffle=True)
    
    test_gen = test_gen.flow_from_dataframe(dataframe = test_df, directory=None, x_col='path', y_col='label',
    weight_col=None, target_size=input_size, color_mode='rgb',
    classes=None, class_mode='categorical', batch_size=batch_size, shuffle=True)
    
    
    if base == 'xception':
        model,name_base = getXceptionModel()
    else:
        model,name_base = getInception3Model()
            
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit_generator(train_gen,
                        steps_per_epoch = len(train_gen) // batch_size,
                        epochs=epochs,
                        verbose=1)
    
    model_name = 'models/{}_tf_model_{}'.format(name_base,str(max(history.history['accuracy'])).split('.')[-1][:4])
    model_mobile = 'models/mobile/{}__tf_model_{}'.format(name_base,str(max(history.history['accuracy'])).split('.')[-1][:4])
    
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'{}'.format(model_name))
    mobile_directory = os.path.join(current_directory, r'{}'.format(model_mobile))
    
    final_directory = final_directory.replace('\\', '/')
    mobile_directory = mobile_directory.replace('\\', '/')
    
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
       
    if not os.path.exists(mobile_directory):
       os.makedirs(mobile_directory)
    
    model.save(final_directory)
    
    # saving mobile version
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(mobile_directory + "/model.tflite", 'wb') as f:
        f.write(tflite_model)
    
    
    model = tf.keras.models.load_model(final_directory)
    
    results = model.evaluate(test_gen, batch_size=batch_size)
    print(results)
    return train_df,test_df,model,history,final_directory

def spliter(data, class_or_id='id'):
    if class_or_id.upper() == 'CLASS':
        output = data.split('/')[-2]
        
    else:
        output = data.split('/')[-1]
    return output





if __name__ == '__main__':
    
    train_mode = False
    train_df,test_df,model,history,model_directory = main(train_mode,base='xception' ,epochs=100)
    print('Model is saved at: '+model_directory)
    
    
    


