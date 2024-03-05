import os
import boto3
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

data_directory = 'dataset'
img_width, img_height = 512, 512
batch_size = 32
epochs = 12
bucket_name = 'tuber-ai-image'
prefix = 'Tuberculosis'

def load_images():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

def train_model(model, train_generator, validation_generator, epochs):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )

def save_model_to_s3(model, local_filename, s3_bucket, s3_key_prefix):
    model.save(local_filename)
    s3_key = f'{s3_key_prefix}-{str(pd.Timestamp.utcnow().value)}.h5'

    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key).upload_file(local_filename, s3_bucket, s3_key)
    print('Model uploaded to S3')

    os.remove(local_filename)

def main():
    train_generator, validation_generator = load_images()

    model = create_model()

    train_model(model, train_generator, validation_generator, epochs)

    save_model_to_s3(model, 'tuber-ai.h5', 'tuber-ai', 'tuber-ai')

if __name__ == "__main__":
    main()
