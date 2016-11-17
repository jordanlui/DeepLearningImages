'''Code for fine-tuning Inception V3 for a new task.

Start with Inception V3 network, not including last fully connected layers.

Train a simple fully connected layer on top of these.


'''

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
import inception_v3 as inception

N_CLASSES = 3
IMSIZE = (299, 299)

# TO DO:: Replace these with paths to the downloaded data.
# Training directory
train_dir = 'animals/train'
# Testing directory
test_dir = 'animals/test'


# Start with an Inception V3 model, not including the final softmax layer.
base_model = inception.InceptionV3(weights='imagenet')
print 'Loaded Inception model'

# Turn off training on base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add on new fully connected layers for the output classes.
x = Dense(32, activation='relu')(base_model.get_layer('flatten').output)
x = Dropout(0.5)(x)
predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

model = Model(input=base_model.input, output=predictions)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# Show some debug output
print (model.summary())

print 'Trainable weights'
print model.trainable_weights


# Data generators for feeding training/testing images to the model.
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=32,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,  # this is the target directory
        target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=32,
        nb_epoch=5,
        validation_data=test_generator,
        verbose=2,
        nb_val_samples=80)
model.save_weights('animals_pretrain.h5')  # always save your weights after training or during training


# Test our model against a bird
img_path = 'animals/test/bird/n01503061_172.JPEG'
img = image.load_img(img_path, target_size=IMSIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inception.preprocess_input(x)
preds = model.predict(x)
print('Model prediction on a bird test image', preds)

# Test our model against a fish
img_path = 'animals/test/fish/n02512053_323.JPEG'
img = image.load_img(img_path, target_size=IMSIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inception.preprocess_input(x)
preds = model.predict(x)
print('Model prediction on a fish test image', preds)


# Test our model against an insect
img_path = 'animals/test/insect/n02159955_249.JPEG'
img = image.load_img(img_path, target_size=IMSIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inception.preprocess_input(x)
preds = model.predict(x)
print('Model prediction on an insect test image', preds)