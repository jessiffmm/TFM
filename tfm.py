from keras.models import Model
from keras.layers import MaxPooling3D, BatchNormalization,Conv3D,Input
from keras.layers import concatenate
from keras.utils import plot_model
from keras.optimizers import Adam
import numpy as np
from keras import backend as K
from generate_data import *
from get_patches import  *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import SimpleITK as sitk
from keras.utils import multi_gpu_model
import tensorflow as tf


###############################################################################
### TERCERA ETAPA  ############################################################
###############################################################################
# Se añaden 6 capas convolucionales:
# 1º- 60 units, dimensiones 3x3x3 y normalización batch.
# 2º- 70 units, dimensiones 3x3x3 y normalización batch.
# 3º- 75 units, dimensiones 3x3x3 y normalización batch.
# 4º- 150 units, dimensiones 3x3x3 y normalización batch.
# 5ª- 150 units, dimensiones 1x1x1 y normalización batch.
# 6ª- 3 units, dimensiones 1x1x1 y normalización batch.
def tercera_etapa(input):

   model_etapa = Conv3D(60, (3, 3, 3), activation='relu', name='1_conv60')(input)
   model_etapa = BatchNormalization(axis=1)(model_etapa)

   model_etapa = Conv3D(70, (3, 3, 3), activation='relu',name='1_conv70')(model_etapa)
   model_etapa = BatchNormalization(axis=1)(model_etapa)

   model_etapa = Conv3D(75, (3, 3, 3), activation='relu',name='1_conv75')(model_etapa)
   model_etapa = BatchNormalization(axis=1)(model_etapa)

   model_etapa = Conv3D(150, (3, 3, 3), activation='relu', name='1_conv150')(model_etapa)
   model_etapa = BatchNormalization(axis=1)(model_etapa)

   model_etapa = Conv3D(150, (1, 1, 1), activation='relu', name='2_conv150')(model_etapa)
   model_etapa = BatchNormalization(axis=1)(model_etapa)

   model_etapa = Conv3D(3, (1, 1, 1), activation='relu', name='1_conv3')(model_etapa)
   model_etapa = BatchNormalization(axis=1)(model_etapa)

   return model_etapa

###############################################################################
### SEGUNDA ETAPA  IMÁGENES 29 x 29 x 29  #####################################
###############################################################################
# Las imágenes de tamaño 29x29x29 pasan por dos capas convolucionales. Una de 25
# units(dimensionalidad del espacio de salida) y otra de 30 , de dimensiones
# 3x3x3 y normalización batch. Posteriormente se realiza max.pooling para reducir
# el tamaño.
def segunda_etapa_29(input):

   model_capa_29 =Conv3D(25, (3, 3, 3), activation='relu', name='1_conv25_29') (input)
   model_capa_29 = BatchNormalization(axis=1)(model_capa_29)

   model_capa_29 = Conv3D(30, (3, 3, 3), activation='relu',name='1_conv30_29') (model_capa_29)
   model_capa_29 = BatchNormalization(axis=1)(model_capa_29)

   model_capa_29 = MaxPooling3D(pool_size=(2, 2, 2), strides=None,
                                        padding='valid', data_format=None, name='1_maxpooling_29')(model_capa_29)

   return model_capa_29

###############################################################################
### SEGUNDA ETAPA  IMÁGENES 27 x 27 x 27  #####################################
###############################################################################
# Las imágenes de tamaño 27x27x27 pasan por tres capas convolucionales de 25
# units(dimensionalidad del espacio de salida), dimensiones 3x3x3 y
# normalización batch. Posteriormente pasan por tres capas convolucionales de 30
# units(dimensionalidad del espacio de salida), dimensiones 3x3x3 y
# normalización batch.
def segunda_etapa_27(input):

   model_capa_27 = Conv3D(25, (3, 3, 3), activation='relu', name='1_conv25_27')(input)
   model_capa_27 = BatchNormalization(axis=1)(model_capa_27)

   model_capa_27 = Conv3D(25, (3, 3, 3), activation='relu',name='2_conv25_27')(model_capa_27)
   model_capa_27 = BatchNormalization(axis=1)(model_capa_27)

   model_capa_27 = Conv3D(25, (3, 3, 3), activation='relu',name='3_conv25_27')(model_capa_27)
   model_capa_27 = BatchNormalization(axis=1)(model_capa_27)

   model_capa_27 = Conv3D(30, (3, 3, 3), activation='relu', name='1_conv30_27')(model_capa_27)
   model_capa_27 = BatchNormalization(axis=1)(model_capa_27)

   model_capa_27 = Conv3D(30, (3, 3, 3), activation='relu', name='2_conv30_27')(model_capa_27)
   model_capa_27 = BatchNormalization(axis=1)(model_capa_27)

   model_capa_27 = Conv3D(30, (3, 3, 3), activation='relu', name='3_conv30_27')(model_capa_27)
   model_capa_27 = BatchNormalization(axis=1)(model_capa_27)

   return model_capa_27

###############################################################################
### PRIMERA ETAPA  IMÁGENES 29 x 29 x 29  #####################################
###############################################################################
# Las imágenes de tamaño 29x29x29 pasan por tres capas convolucionales con 16
# units(dimensionalidad del espacio de salida) , de dimensiones 3x3x3 y
# normalización batch.

def primera_etapa_29(input,name_type):

   model_29 = Conv3D(16, (3, 3, 3), activation='relu',name='1_conv16_'+name_type+'29')(input)
   model_29 = BatchNormalization(axis=1,name='batch_'+name_type+'29')(model_29)

   model_29 = Conv3D(16, (3, 3, 3), activation='relu',name='2_conv16_'+name_type+'29')(model_29)
   model_29 = BatchNormalization(axis=1)(model_29)

   model_29 = Conv3D(16, (3, 3, 3), activation='relu',name='3_conv16_'+name_type+'29')(model_29)
   model_29 = BatchNormalization(axis=1)(model_29)

   return model_29

###############################################################################
### PRIMERA ETAPA  IMÁGENES 27 x 27 x 27  #####################################
###############################################################################
# Las imágenes de tamaño 27x27x27 pasan por tres capas convolucionales con 16
# units(dimensionalidad del espacio de salida) , de dimensiones 3x3x3 y
# normalización batch.

def primera_etapa_27(input,name_type):

   model_27 = Conv3D(16, (3, 3, 3), activation='relu', name='1_conv16_'+name_type+'27')(input)
   model_27 = BatchNormalization(axis=1, name='batch_'+name_type+'27')(model_27)

   model_27 = Conv3D(16, (3, 3, 3), activation='relu',name='2_conv16_'+name_type+'27')(model_27)
   model_27 = BatchNormalization(axis=1)(model_27)

   model_27 = Conv3D(16, (3, 3, 3), activation='relu',name='3_conv16_'+name_type+'27')(model_27)
   model_27 = BatchNormalization(axis=1)(model_27)

   return model_27

# Main principal
if __name__ == '__main__':
   img = sitk.ReadImage('brain.nii')
   data_img = sitk.GetArrayFromImage(img)
   #reset the buffers
   K.clear_session()
   #K.get_session()

   #x_29 = K.placeholder(shape=(None, 29, 29, 29, 2), dtype='float32', name="x")
   #x_27 = K.placeholder(shape=(None, 27, 27, 27, 2), dtype='float32', name="x")
   x_29_fixed = Input(shape=(29, 29, 29, 1),dtype='float32',name="xfixed29")
   x_29_moved = Input(shape=(29, 29, 29, 1), dtype='float32', name="xmoved29")
   x_27_fixed = Input(shape=(27, 27, 27, 1), dtype='float32', name="xfixed27")
   x_27_moved = Input(shape=(27, 27, 27, 1), dtype='float32', name="xmoved27")
   #y = K.placeholder(shape=(None, 1, 1, 1, 3), dtype='float32', name="label")

   #FIXED 29
   fixed_29_conv_1 = primera_etapa_29(x_29_fixed,'fixed')
   # MOVED 29
   moved_29_conv_1 = primera_etapa_29(x_29_moved,'moved')

   # FIXED 27
   fixed_27_conv_1 = primera_etapa_27(x_27_fixed,'fixed')
   # MOVED 27
   moved_27_conv_1 = primera_etapa_27(x_27_moved,'moved')

   # Concatenamos las capas de las imágenes fijas de 29x29x29 con las de las imágenes en movimiento
   primera_capa_29 = concatenate([fixed_29_conv_1, moved_29_conv_1])
   # Concatenamos las capas de las imágenes fijas de 27x27x27 con las de las imágenes en movimiento
   primera_capa_27 = concatenate([fixed_27_conv_1, moved_27_conv_1])

   # Pasamos a la segunda etapa de capas por la que pasan las imágenes de dimensiones 29x29x29
   salida_29_segunda_etapa = segunda_etapa_29(primera_capa_29)
   # Pasamos a la segunda etapa de capas por la que pasan las imágenes de dimensiones 27x27x27
   salida_27_segunda_etapa = segunda_etapa_27(primera_capa_27)

   # Concatenamos la salida de la segunda etapa de las imágenes de 29x29x29 con las de 27x27x27
   entrada_tercera_etapa = concatenate([salida_29_segunda_etapa, salida_27_segunda_etapa])
   # Añadimos las últimas capas a la red
   salida = tercera_etapa(entrada_tercera_etapa)

   #with tf.device('/cpu:0'):
   # Creamos el modelo con las entradas y la salida total, con sus correspondientes capas
   model = Model(inputs=[x_29_fixed,x_29_moved,x_27_fixed,x_27_moved], outputs=salida)
   # Imprimimos un resumen de la red
   print(model.summary())
   # Se muestra gráficamente la red
   plot_model(model, to_file='multiple_inputs.png')

   # Se crea el optimizador Adam
   adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1.25)
   parallel_model = multi_gpu_model(model,gpus=2)
   #Compilamos el modelo
   parallel_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mae','accuracy'])
   
   print(model.output_shape)
   # model.fit([X_fija_29,X_mov_29,X_fija_27,X_mov_27],Y_train, batch_size=100, epochs=20,
   #           verbose=2, validation_data=(X_test, Y_test))

   #for iteraciones in range(0,1000):
   DVF = generate_data()
   DVF.generate_dvf()
   for i in range(0,20):

      patches_train = get_patches(sitk.ReadImage("brain.nii"),100)
      [X_27_moving_train, X_29_moving_train, X_27_fixed_train, X_29_fixed_train, Y_train] = patches_train.patches()
      # print('maximo',np.max(X_27_moving_train))
      # print('maximo', np.max(X_29_moving_train))
      # print('maximo', np.max(X_27_fixed_train))
      # print('maximo', np.max(X_29_fixed_train))

      #print((patches.Y.shape))
      patches_test = get_patches(sitk.ReadImage("brain.nii"), 25)
      [X_27_moving_test, X_29_moving_test, X_27_fixed_test, X_29_fixed_test, Y_test] = patches_test.patches()
      # print('maximo', np.max(X_27_moving_test))
      # print('maximo', np.max(X_29_moving_test))
      # print('maximo', np.max(X_27_fixed_test))
      # print('maximo', np.max(X_29_fixed_test))
      #X_27 = np.concatenate((patches.X_27_fixed, patches.X_27_moving), axis=4).astype(np.float32)
      #X_29 = np.concatenate((patches.X_29_fixed, patches.X_29_moving), axis=4).astype(np.float32)
      # X_29_fixed_train = X_29_fixed_train.astype('float32')
      # X_29_moving_train = X_29_moving_train.astype('float32')
      # X_27_fixed_train = X_27_fixed_train.astype('float32')
      # X_27_moving_train = X_27_moving_train.astype('float32')
      #
      # X_29_fixed_test = X_29_fixed_test.astype('float32')
      # X_29_moving_test = X_29_moving_test.astype('float32')
      # X_27_fixed_test = X_27_fixed_test.astype('float32')
      # X_27_moving_test = X_27_moving_test.astype('float32')

      X_29_fixed_train = (X_29_fixed_train + 2000)/np.max(data_img)
      X_29_moving_train = (X_29_moving_train + 2000)/np.max(data_img)
      X_27_fixed_train = (X_27_fixed_train + 2000)/np.max(data_img)
      X_27_moving_train = (X_27_moving_train + 2000)/np.max(data_img)

      X_29_fixed_test = (X_29_fixed_test + 2000)/np.max(data_img)
      X_29_moving_test = (X_29_moving_test + 2000)/np.max(data_img)
      X_27_fixed_test = (X_27_fixed_test + 2000)/np.max(data_img)
      X_27_moving_test = (X_27_moving_test + 2000)/np.max(data_img)
      #print(X_27.shape)
      #print(X_29.shape)
      #print('x 29 traing',X_29_fixed_train.shape)
      #print('Y traing', Y_train.shape)
      history = parallel_model.fit([X_29_fixed_train,X_29_moving_train,X_27_fixed_train,X_27_moving_train], Y_train,
                          batch_size=32, epochs=20, verbose=2,
                          validation_data=([X_29_fixed_test,X_29_moving_test,X_27_fixed_test,X_27_moving_test], Y_test))

      score = model.evaluate([X_29_fixed_test,X_29_moving_test,X_27_fixed_test,X_27_moving_test], Y_test, verbose=0)
      print ('Score',score)
      print('history',history.history)
      print('Test score:', score[0])
      print('Test mean absolute error:', score[1])
      print('Test accuracy:', score[2])


      plt.figure()
      plt.plot(history.history['acc'])
      plt.plot(history.history['val_acc'])
      plt.title('Model Accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.savefig('result/acurracy' + str(i) + '.jpg')

      plt.savefig('Accuracy_cnn.jpg')
      # summarize history for loss
      plt.figure()
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('Model Loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.savefig('result/Loss_cnn'+str(i)+'.jpg')
      #plt.show()


   # preds = model.predict([X_29_fixed_test,X_29_moving_test,X_27_fixed_test,X_27_moving_test])
   # cm = confusion_matrix(np.argmax(Y_test, axis=1), preds)
   # plt.matshow(cm)
   # plt.title('Confusion matrix')
   # plt.colorbar()
   # plt.ylabel('True label')
   # plt.xlabel('Predicted label')
   # plt.savefig('Confusion_cnn.jpg')
   # plt.show()

   model_json = model.to_json()
   open('model_cnn.json', 'w').write(model_json)
   model.save_weights('model_cnn.h5', overwrite=True)

