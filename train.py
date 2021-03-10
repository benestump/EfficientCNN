import tensorflow as tf
import numpy as np
from models.MobileNetV2 import MobileNetV2 
import math

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

class Trainer(object):
    def __init__(self,
                 model,
                 epochs,
                 global_batch_size,
                 strategy,
                 initial_learning_rate,
                 start_epoch=1,
                 tensorboard_dir='./logs'):
        
        self.start_epoch = start_epoch
        self.model = model
        self.epochs = epochs
        self.strategy = strategy 
        self.global_batch_size = global_batch_size
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
        self.model = model
        self.current_learning_rate = initial_learning_rate
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.patience_count = 0
        self.max_patience = 10
        self.tensorboard_dir = tensorboard_dir
        self.best_model = None
        self.version = version 

    def lr_decay(self):
        if self.patience_count >= self.max_patience:
            self.current_learning_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0
        self.patience_count += 1

        self.optimizer.learning_rate = self.current_learning_rate

    def lr_decay_step(self, epoch):
        if epoch == 25 or epoch == 50 or epoch == 75:
            self.current_learning_rate /= 10.0
        self.optimizer.learning_rate = self.current_learning_rate

    def run(self, train_dist_dataset, val_dist_dataset):


    def save_model(self, epoch, loss):
        model_name = f'./models/MobileNetV2-epoch-{epoch}-loss-{loss:.4f}.h5'
        self.model.save_weights(model_name)
        self.best_model = model_name
        print(f'Model {model_name} saved.')

def create_dataset(tfrecords, batch_size, is_train):

    dataset = tfrecords.map(normalize_and_resize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(batch_size)
    
    dataset = dataset.batch(batch_size)
    


def train(dataset='cifar10',epochs, start_epoch, leargning_rate, tensorboard_dir, checkpoint, batch_size, train_tfrecords, val_tfrecords):

    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * batch_size
    
    tfdatasets, info = tfds.load(name=dataset, with_info=True, as_supervised=True)

    ds_train, ds_test = tfdatasets['train'], tfdatasets['test']

    train_dataset = create_dataset(ds_train, global_batch_size, is_train='True')
    val_dataset = create_dataset(ds_test, global_batch_size, is_train='False')

automatic_gpu_usage()




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
 