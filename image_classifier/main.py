import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFile


file_path = os.path.join('data', 'happy')

# avoid OOM(Out Of Memory) error, so we set GPU consumption growth in case system uses a gpu
gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)

# remove dodgy images
import cv2
import imghdr


data_dir = 'data'
# print(os.listdir(os.path.join(data_dir, 'happy')))

image_exts = ['jpeg', 'jpg', 'png','bmp', 'gif']



# removing images that are not of image format and which tensorflow cannot process
for image_class in os.listdir(data_dir) :
    for image in os.listdir(os.path.join(data_dir, image_class)) :
        image_path = os.path.join(data_dir, image_class, image)
        if image.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):   
            try :
                # Check if the image extension is valid
                img_type = imghdr.what(image_path)
                if img_type not in image_exts:
                    print(f"Image not in allowed extensions list, removing: {image_path}")
                    os.remove(image_path)
                    continue

                # Attempt to open the image with OpenCV and Pillow to catch any corrupted files
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Corrupted image detected and removed: {image_path}")
                    os.remove(image_path)
                    continue

                with Image.open(image_path) as img:
                    img.verify()  # Verify that it is indeed an image
                 # Check if TensorFlow can decode the image
                raw_img = tf.io.read_file(image_path)
                _ = tf.image.decode_jpeg(raw_img)
            except Exception as e:
                print(f"Removing corrupted or unreadable image: {image} - Error: {e}")
                os.remove(image_path) 


# Load dataset
tf.data.Dataset
data = tf.keras.utils.image_dataset_from_directory('data')
# print(data)
        
data_iterator = data.as_numpy_iterator()
# print(data_iterator)
batch = data_iterator.next()

# images represented as numpy array batch[1] represents the label
print(batch[0].shape)

# class 0 happy class 1 sad

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]) :
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# Pre-processing in the pipeline
scaled_data = data.map(lambda x,y : (x/255, y))
scaled_iter = scaled_data.as_numpy_iterator()
print(scaled_iter.next()[0].max())

# SPLIT DATA
print(len(scaled_data))

train_size = int(len(scaled_data)*.7)
validate_size = int(len(scaled_data)*.2)+1
test_size = int(len(scaled_data)*.1)+1

print(train_size + validate_size + test_size)

train = scaled_data.take(train_size)
validate = scaled_data.skip(train_size).take(validate_size)
test = scaled_data.skip(train_size + validate_size).take(test_size)

# Deep model
# following code in interactive window

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers  import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential([
    Conv2D(16,(3,3),1, activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(),
    Conv2D(32,(3,3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(16,(3,3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
    ])

# compile
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# train
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs = 20, validation_data = validate, callbacks = [tensorboard_callback])

# Plot performance

# loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='label')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='label')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Evaluate Performance

# evaluate
from tensorflow.python.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

# Test

img = cv2.imread("test_img_1.jpg")
plt.imshow(img, cv2.COLOR_BGR2RGB)
plt.show()

resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))







