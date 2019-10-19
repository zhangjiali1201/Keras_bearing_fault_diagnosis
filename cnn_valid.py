#coding:utf-8
"""
Created on Tue Mar 26 20:37:58 2019

@author: jiali zhang

for choosing the best model

"""

from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten
from keras.models import Sequential
from keras.utils import plot_model
from keras.regularizers import l2
import preprocess
from keras.callbacks import TensorBoard
import numpy as np
import time

# 训练参数
batch_size = 128
epochs = 12
num_classes = 10
length = 2048
BatchNorm = True # 是否批量归一化
number = 1000 # 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例
date=time.strftime("%Y%m%d", time.localtime())

path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=rate,
                                                                  enc=True, enc_step=28)

x_train, x_valid, x_test = x_train[:,:,np.newaxis], x_valid[:,:,np.newaxis], x_test[:,:,np.newaxis]

input_shape =x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

# train multiple models
conv_layers = [1, 2, 3]
layer_sizes = [32, 64, 128]
dense_layers = [0, 1, 2]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            mark=time.strftime("%Y%m%d_%H%M", time.localtime())#在模型末尾标注日期时间
            model_name = "{}-conv-{}-filters-{}-dense-{}".format(conv_layer, layer_size, dense_layer, mark)
            tb_cb = TensorBoard(log_dir='logs/{}_logs/{}'.format(date, model_name))
            print(model_name)
            model = Sequential()
         
            model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))

            for l in range(conv_layer-1):
                model.add(Conv1D(layer_size,kernel_size=3, strides=1, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

            #从卷积到全连接展平
            model.add(Flatten()) 
            
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))
            
            #编译模型
            model.compile(loss="categorical_crossentropy",
                         optimizer="adam",
                         metrics=["accuracy"])
            
            #训练模型
            model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                      verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
                      callbacks=[tb_cb])
            
            plot_model(model=model, to_file="images/{}-conv-{}-filters-{}-dense.png".format(conv_layer, layer_size, dense_layer), show_shapes=True)

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("loss：", score[0])
print("accuracy：", score[1])
