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
bucket_name = 'tuber-ai-images'
prefix = 'Tuberculosis'


def load_images_from_s3(bucket, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects(Bucket=bucket, Prefix=prefix)

    images = []
    labels = []

    for obj in response.get('Contents', []):
        try:
            obj_data = s3.get_object(Bucket=bucket, Key=obj['Key'])['Body'].read()
            image = preprocess_image(obj_data)
            images.append(image)
            labels.append(1 if 'tuber' in obj['Key'] else 0)
        except Exception as e:
            print(f"Error processing object {obj['Key']}: {str(e)}")

    return images, labels


def preprocess_image(data):
    image = Image.open(BytesIO(data))
    image = image.resize((img_width, img_height))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image_array)


def create_model():
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

    return model


def train_model(model, dataset, epochs, callbacks):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'], loss_weights=[0.5])
    model.fit(dataset, epochs=epochs, callbacks=callbacks)


def save_model_to_s3(model, local_filename, s3_bucket, s3_key_prefix):
    model.save(local_filename)
    s3_key = f'{s3_key_prefix}-{str(pd.Timestamp.utcnow().value)}.h5'
    boto3.client('s3').upload_file(local_filename, s3_bucket, s3_key)
    print('Model uploaded to S3')

    os.remove(local_filename)


def main():
    images, labels = load_images_from_s3(bucket_name, prefix)

    images_tensor = tf.convert_to_tensor(images)
    dataset = tf.data.Dataset.from_tensor_slices(images_tensor).shuffle(buffer_size=len(images)).batch(batch_size)

    model = create_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    train_model(model, dataset, epochs, callbacks=[early_stopping])

    save_model_to_s3(model, 'tuber-ai.h5', 'tuber-ai', 'tuber-ai')


if __name__ == "__main__":
    main()

