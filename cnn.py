#Import libs and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initialise CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPool2D(pool_size = (2, 2)))

#Adding 2nd convolution layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2, 2)))


#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(p= 0.1))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting CNN 
#get the code from keras documentation -> image augmentaton
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=5,
        validation_data=test_set,
        validation_steps=2000 )


#single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
