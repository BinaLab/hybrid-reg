import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
from PIL import Image
from os import *
from os.path import *
import cv2
import tensorflow as tf
import datetime
import data_loader

from keras import backend as K
import sys

epochs = 10
max_labels = 34 ### number of classes = number of layers
lPhy = float(sys.argv[1]) # lambda for phy loss
# lPhy = 0.1
print("### exp lambda = " + str(lPhy) + " ###")
img_ext = '.png'
txt_ext = '.txt'
np_ext = '.npy'
timestamp = datetime.datetime.now()
dt_format = "%m-%d-%y_%H-%M"
out = join('output', timestamp.strftime(dt_format) + "_lMan"+str(1-lPhy) + "_lPhy"+str(lPhy))
models = join(out,'models')
plots = join(out,'plots')
evaluation = join(out,'evaluation')
history = join(out,'history')

if not isdir(models):
    makedirs(models)
if not isdir(plots):
    makedirs(plots)
if not isdir(evaluation):
    makedirs(evaluation)
if not isdir(history):
    makedirs(history)

mae = MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)

def hybrid_loss(y_true, y_pred):
    ### shape = batch_size x label type (2) x 34
    # y_man = y_true[:,0,:]
    # y_phy = y_true[:,1,:]
    
    ### shape = batch_size x 34
    # y_man = y_true[:,:max_labels]
    # y_phy = y_true[:,max_labels:]
    
    ### shape = batch_size x 34 x label type(2)
    # y_man = y_true[:,:,0]
    # y_phy = y_true[:,:,1]

    y_man = y_true[:,0]
    y_phy = y_true[:,1]

    loss = mae(y_man, y_pred) + 0.5*mae(y_phy, y_pred)

    # loss_man = mae(y_man, y_pred)
    # loss_phy = 0.5*mae(y_phy, y_pred)
    
    # loss_man = K.mean(K.abs(y_man-y_pred), axis=-1)
    # loss_phy = 0.5*K.mean(K.abs(y_phy-y_pred), axis=-1)
    # loss = loss_man + loss_phy
    return loss

def loss1(y_true, y_pred):
    loss = mae(y_true, y_pred)
    return loss


def loss2(y_true, y_pred):


    loss = mae(y_true, y_pred)

    return loss

def train_model(base_model, model_name, traindata, thickness_estims):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    # predictions = Dense(max_labels, activation='relu')(x)
    # model = Model(inputs=base_model.input, outputs=[predictions,predictions])
    x1 = Dense(1024, activation='relu')(x)
    x2 = Dense(1024, activation='relu')(x)
    predictions1 = Dense(max_labels, activation='relu')(x1)
    predictions2 = Dense(max_labels, activation='relu')(x2)
    model = Model(inputs=base_model.input, outputs=[predictions1,predictions2])
    
    model_chkpnt = ModelCheckpoint(
    filepath=join(models,model_name+'.h5'),
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)
    
    # model.compile(optimizer=Adam(lr=0.0001), loss='mae', metrics=['accuracy']) # old
    # model.compile(optimizer=Adam(lr=0.0001), loss='mae', metrics=['mae'])
    # model.compile(optimizer=Adam(lr=0.0001), loss=hybrid_loss, metrics=['mae'])
    model.compile(optimizer=Adam(lr=0.0001), loss=['mae','mae'], loss_weights=[1-lPhy,lPhy], metrics=['mae'])
    # model.compile(optimizer=Adam(lr=0.0001), loss=[loss1,loss2], loss_weights=[1,0.5], metrics=['mae'])
    
    # model.summary()
    # hist = model.fit(traindata, [np.asarray(thickness_estims[0]), np.asarray(thickness_estims[1])], verbose=1, epochs=epochs, callbacks=[model_chkpnt])
    hist = model.fit(traindata, thickness_estims, verbose=1, epochs=epochs, callbacks=[model_chkpnt])

    print("-- " + model_name + ' trained -- ')
    np.save(join(history,model_name+np_ext),hist.history)
    np.savetxt(join(evaluation,model_name+txt_ext), model.evaluate(traindata, thickness_estims, verbose=1))
    return hist

def plot_model(hist, model_name, folder_out=plots):
    print(hist.history.keys())
    plt.plot(hist.history["loss"])
    plt.title(model_name)
    plt.ylabel("MAE Loss")
    plt.xlabel("Epoch")
    plt.yscale("linear")
    plt.savefig(join(folder_out,model_name+img_ext))
    plt.close()


traindata = data_loader.traindata


thick_estims = data_loader.train_thick

### model definition ###

model_name1 = 'InceptionV3'
print('training ' + model_name1)
inception = InceptionV3(weights='imagenet', include_top=False)
# inception = InceptionV3(weights=None, include_top=False)
hist1 = train_model(inception, model_name1, traindata, thick_estims)
plot_model(hist1, model_name1)


model_name2 = 'DenseNet121'
print('training ' + model_name2)
densenet = DenseNet121(weights='imagenet', include_top=False)
# densenet = DenseNet121(weights=None, include_top=False)
hist2 = train_model(densenet, model_name2, traindata, thick_estims)
plot_model(hist2, model_name2)


model_name3 = 'ResNet50'
print('training ' + model_name3)
resnet = ResNet50(weights='imagenet', include_top=False)
# resnet = ResNet50(weights=None, include_top=False)
hist3 = train_model(resnet, model_name3, traindata, thick_estims)
plot_model(hist3, model_name3)


model_name4 = 'Xception'
print('training ' + model_name4)
xception = Xception(weights='imagenet', include_top=False)
# xception = Xception(weights=None, include_top=False)
hist4 = train_model(xception, model_name4, traindata, thick_estims)
plot_model(hist4, model_name4)


model_name5 = 'MobileNetV2'
print('training ' + model_name5)
mobilenet = MobileNetV2(weights='imagenet', include_top=False)
# mobilenet = MobileNetV2(weights=None, include_top=False)
hist5 = train_model(mobilenet, model_name5, traindata, thick_estims)
plot_model(hist5, model_name5)


## plot all
plt.plot(hist1.history["loss"])
plt.plot(hist2.history["loss"])
plt.plot(hist3.history["loss"])
plt.plot(hist4.history["loss"])
plt.plot(hist5.history["loss"])
plt.title("Model Loss Curves")
plt.ylabel("MAE Loss")
plt.xlabel("Epoch")
plt.yscale("linear")
plt.legend([model_name1,model_name2,model_name3,model_name4,model_name5])
plt.savefig(join(plots,'all_loss'+img_ext))
plt.close()

print('done')