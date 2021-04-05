import argparse
import tensorflow as tf
import numpy as np
from models.MobileNetV2 import MobileNetV2 
import math
import os

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

def automatic_tpu_usage():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

def normalize_and_resize_img(image, label):

    image = tf.image.resize(image, (32, 32))

    return tf.cast(image, tf.float32)/255., label

def create_dataset(tfrecords, batch_size, is_train):

    dataset = tfrecords.map(normalize_and_resize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if is_train:
        dataset = dataset.shuffle(200)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset


def train(dataset,input_shape, epochs, leargning_rate, batch_size):

    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * batch_size
    
    tfdatasets, info = tfds.load(name=dataset, with_info=True, as_supervised=True)

    ds_train, ds_test = tfdatasets['train'], tfdatasets['test']

    train_dataset = create_dataset(ds_train, global_batch_size, is_train='True')
    val_dataset = create_dataset(ds_test, global_batch_size, is_train='False')

    if not os.path.exists(os.path.join('./models_save')):
        os.makedirs(os.path.join('./models_save/'))

    with strategy.scope():

        model = MobileNetV2(input_shape=input_shape, classes=10)
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(momentum=0.9),
            metrics=['accuracy'],
        )

    checkpoint_dir = './training_checkpoints'
    if not os.path.exists(os.path.join(checkpoint_dir)):
        os.makedirs(os.path.join(checkpoint_dir))

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    def decay(epoch):
        if epoch < 250:
            return leargning_rate
        elif epoch >= 250 and epoch < 280:
            return leargning_rate/10.
        else:
            return leargning_rate/10.
    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\n에포크 {}의 학습률은 {}입니다.'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))  
        
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                         save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]


    history_MBNetV2 = model.fit(
    train_dataset, 
    epochs=epochs,
    validation_data=val_dataset,
    verbose=1,
    use_multiprocessing=True,
    callbacks=callbacks
    )
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='Select device for training')
    parser.add_argument('--dataset', help='cifar10, imagenet')
    parser.add_argument('--epochs', type=int, default=200, help='Epoch')
    parser.add_argument('--model', help='choose model')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.1, help='Learning Rate')
    
    args = parser.parse_args()

    INPUT_SAHPE=(32, 32, 3)
    
    if args.device == 'GPU':
        automatic_gpu_usage()
    elif args.device == 'TPU':
        automatic_tpu_usage()

    train(dataset=args.dataset,input_shape=INPUT_SAHPE, epochs=args.epochs, leargning_rate=args.learning_rate,batch_size=args.batch_size)