import tensorflow as tf
import numpy as np
from models.MobileNetV2 import MobileNetV2 

import tensorflow_datasets as tfds
import urllib3
urllib3.disable_warnings()

def automatic_gpu_usage() :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

automatic_gpu_usage()

BATCH_SIZE = 32
INPUT_SHAPE = (32, 32, 3)
EPOCH = 200

ds_train, ds_info = tfds.load(
    'cifar10',
    split='train[:80%]',
    as_supervised=True,
    with_info=True,
)
ds_test, ds_info = tfds.load(
    'cifar10',
    split='train[80%:]',
    as_supervised=True,
    with_info=True,
)


def normalize_and_resize_img(image, label):

    image = tf.image.resize(image, (32, 32))

    return tf.cast(image, tf.float32)/255., label

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(normalize_and_resize_img, num_parallel_calls=1)
    ds = ds.batch(batch_size)

    if not is_test:
        ds=ds.repeat()
        ds=ds.shuffle(200)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

ds_train = apply_normalize_on_dataset(ds_train, batch_size=BATCH_SIZE)
ds_test = apply_normalize_on_dataset(ds_test, batch_size=BATCH_SIZE)

model = MobileNetV2(input_shape=INPUT_SHAPE, classes=10)

# print(model.summary())

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
    metrics=['accuracy'],
)

history_MBNetV2 = model.fit(
    ds_train, 
    steps_per_epoch=int(40000/BATCH_SIZE),
    validation_steps=int(10000/BATCH_SIZE),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)
 