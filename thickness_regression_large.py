import cv2
import sys
import datetime
from os import *
import numpy as np
from os.path import *
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, GlobalAveragePooling2D

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import data_generator


def create_opdir(dir_name):
    if not isdir(dir_name):
        makedirs(dir_name)
    return dir_name


def train_model(models_dir, hist_dir, eval_dir, model_name, 
                 traindata, thickness_estims=None):
    
    fncall="(weights='imagenet', include_top=False)"
    base_model = eval(model_name+fncall)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(max_labels, activation='relu')(x)
    model = Model(inputs=base_model.input, outputs=[predictions,predictions])
    
    ### possible alternatives ###
    # x1 = Dense(1024, activation='relu')(x)
    # x2 = Dense(1024, activation='relu')(x)
    # predictions1 = Dense(max_labels, activation='relu')(x1)
    # predictions2 = Dense(max_labels, activation='relu')(x2)
    # model = Model(inputs=base_model.input, outputs=[predictions1,predictions2])
    
    model_chkpnt = ModelCheckpoint(
    filepath=join(models_dir,model_name+'.h5'),
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)
    
    model.compile(optimizer=Adam(lr=0.0001), loss=['mae','mae'], loss_weights=[1-lPhy,lPhy], metrics=['mae'])
    
    hist = model.fit(traindata, verbose=1, epochs=epochs, callbacks=[model_chkpnt])

    print(" -- " + model_name + ' trained -- ')
    np.save(join(hist_dir,model_name+np_ext),hist.history)
    np.savetxt(join(eval_dir,model_name+txt_ext), model.evaluate(traindata, thickness_estims, verbose=1))
    return hist

def plot_model(hist, model_name, folder_out):
    print(hist.history.keys())
    plt.plot(hist.history["loss"])
    plt.title(model_name)
    plt.ylabel("MAE Loss")
    plt.xlabel("Epoch")
    plt.yscale("linear")
    plt.savefig(join(folder_out,model_name+img_ext))
    plt.close()

if __name__ == "__main__":
    epochs = 10
    max_labels = 34 ### number of classes = number of layers
    # lPhy = float(sys.argv[1]) # lambda for phy loss
    lPhy = 0.1 # lambda for phy loss
    print("### exp lambda = " + str(lPhy) + " ###")
    img_ext = '.png'
    txt_ext = '.txt'
    np_ext = '.npy'
    timestamp = datetime.datetime.now()
    dt_format = "%m-%d-%y_%H-%M"
    out = join('output', timestamp.strftime(dt_format) + "_lambda"+str(lPhy))
    
    op_dirs = ['models','plots','evaluation','history']
    models_dir, plots_dir, eval_dir, hist_dir = [create_opdir(join(out,op_dir)) for op_dir in op_dirs]


    # traindata = data_loader.traindata
    # traindata = data_loader_large2.train_ds
    # # traindata = data_loader_large.traindata
    
    # thick_estims = data_loader.train_thick
    
    # trainX = data_loader_large.traindata
    # trainY = data_loader_large.train_thick
    # datagen = ImageDataGenerator()
    # train_generator = datagen.flow(trainX, trainY, batch_size=32)
    
    train_generator = data_generator.batch_generator
    
    
    # model_names = ['InceptionV3','DenseNet121','ResNet50','Xception','MobileNetV2']
    model_names = ['DenseNet121','ResNet50','Xception','MobileNetV2']
    histories=[]
    for model_name in model_names:
        print(" -- training " + model_name + ' -- ')
        model_history = train_model(models_dir, hist_dir, eval_dir, model_name, train_generator)
        plot_model(model_history, model_name, plots_dir)
        histories.append(model_history)
        
        
    """ plot all model losses together """
    for hist in histories: plt.plot(hist.history["loss"])
    plt.title("Model Loss Curves")
    plt.ylabel("MAE Loss")
    plt.xlabel("Epoch")
    plt.yscale("linear")
    plt.legend(model_names)
    plt.savefig(join(plots_dir,'all_loss'+img_ext))
    plt.close()   
        
    
    print('done')