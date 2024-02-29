import os

import boto3
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from io import BytesIO
from PIL import Image

data_image_directory = 'dataset/'
img_width, img_height = 512, 512
batch_size = 32
epochs = 12

bucket_name = 'tuber-ai-image'
prefix = 'tuberculosis/'

s3 = boto3.client('s3')

response = s3.list_objects(Bucket=bucket_name, Prefix=prefix)

images = []
labels = []

for obj in response.get('Contents', []):
    obj_data = s3.get_object(Bucket=bucket_name, Key=obj['Key'])['Body'].read()

    image = Image.open(BytesIO(obj_data))
    image = image.resize((img_width, img_height))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    images.append(image_array)
    labels.append(1 if 'tuber' in obj['Key'] else 0)


images = tf.convert_to_tensor(images)
dataset = tf.data.Dataset.from_tensor_slices(images)
dataset = dataset.shuffle(buffer_size=len(images)).batch(batch_size)


model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'], loss_weights=[0.5])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(dataset, epochs=epochs, callbacks=[early_stopping])

model.save('tuber-ai.h5')

s3_key = 'tuber-ai-' + str(pd.Timestamp.utcnow().value) + '.h5'

s3.upload_file('tuber-ai.h5', 'tuber-ai', s3_key)

print('Model uploaded to S3')

os.remove('tuber-ai.h5')

