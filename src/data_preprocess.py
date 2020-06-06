from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


def data_augmentation():
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(config.TRAIN_DIR,
                                                        target_size=config.TARGET_SIZE,
                                                        batch_size=config.BATCH_SIZE,
                                                        class_mode=config.CLASS_MODE)

    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(config.VALID_DIR,
                                                        target_size=config.TARGET_SIZE,
                                                        batch_size=config.BATCH_SIZE,
                                                        class_mode=config.CLASS_MODE)

    return train_generator, valid_generator


if __name__ == '__main__':
    data_augmentation()
