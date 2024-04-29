"""
data_loader for larger dataset
"""
import os
import tensorflow as tf
# import bytes
import numpy as np
# from mat73 import loadmat
from scipy.io import loadmat

train_path = 'G:/My Drive/Research/Dataset/SR_Dataset_v1/train_data/img*.png'
train_path2 = 'G:\My Drive\Research\Dataset\SR_Dataset_v1\L1\AttUNet\*.mat'

# root = 'G:/My Drive/Research/Dataset/SR_Dataset_v1/train_data/'
# img_paths = [os.path.join(root,e) for e in os.listdir(root) if '.mat' in e]



# def read_mat(filepath):
#     def _read_mat(filepath):
#         filepath = bytes.decode(filepath.numpy())     
#         mat_file = loadmat(filepath)
#         echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)

#         echo = tf.expand_dims(echo, axis=-1)       

#         # if config['img_channels'] > 1:
#         #     echo = tf.image.grayscale_to_rgb(echo)

#         # layer = tf.cast(mat_file['raster'], dtype=tf.float64)     
#         layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)

#         layer = tf.expand_dims(layer, axis=-1)
#         # layer = tf.keras.utils.to_categorical(layer, config['num_classes'] )
#         shape0 = echo.shape  
#         # return echo,layer,np.asarray(shape0) 
#         return 'xyz'

    

#     output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
#     shape = output[2]
#     data0 = tf.reshape(output[0], shape)
#     data0.set_shape([500,256,3])

   

#     data1 = output[1]  

#     data1.set_shape([500,256,1]) #,30  

#     return data0,data1



def _read_png(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img)
    img = img/255 # This is range [0,1] could also try [-1 1]
    returnList = np.random.rand(1,34)
    return img, [returnList[0].astype('float64'), returnList[0].astype('float64')]
 

train_ds = tf.data.Dataset.list_files(train_path,shuffle=True)
train_ds = train_ds.map(_read_png,num_parallel_calls=8)
train_ds = train_ds.batch(32,drop_remainder=True)

 

def read_mat(filepath):
    def _read_mat(filepath):       

        filepath = bytes.decode(filepath.numpy())  
        mat_file = loadmat(filepath)
        echo = tf.cast(mat_file['echo_tmp'], dtype=tf.float64)

        echo = tf.expand_dims(echo, axis=-1)     

        layer = tf.cast( tf.cast(mat_file['raster'], dtype=tf.bool), dtype=tf.float64)

        layer = tf.expand_dims(layer, axis=-1)
        shape0 = echo.shape  

        return echo,layer,np.asarray(shape0)    

    output = tf.py_function(_read_mat,[filepath],[tf.double,tf.double, tf.int64])
    shape = output[2]
    data0 = tf.reshape(output[0], shape)

    data0.set_shape([1664,256,1])
    data1 = output[1]  

    data1.set_shape([1664,256,1]) #,30  

    return data0,data1

 

# train_ds = tf.data.Dataset.list_files(train_path2,shuffle=True) #'*.mat'

# train_ds = train_ds.map(read_mat,num_parallel_calls=8)

# train_ds = train_ds.batch(4,drop_remainder=True)








# print("######## in data_loader_large #######")
# train_ds = tf.data.Dataset.list_files(train_path,shuffle=True) #'*.mat'
# # train_ds = tf.data.Dataset.from_tensor_slices(img_paths)
# train_ds = train_ds.map(read_mat,num_parallel_calls=8)
# train_ds = train_ds.batch(64,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) # 64 = batch size