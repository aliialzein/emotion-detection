from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, test_dir, img_size=(96, 96), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        validation_split=0.2  # 20% for validation
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='rgb',  # 💥 Now using RGB
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
