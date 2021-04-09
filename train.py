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
    return resolver

def normalize_and_resize_img(image, label):

    image = tf.image.resize(image, (32, 32))

    return tf.cast(image, tf.float32)/255., label

def create_dataset(tfrecords, batch_size, is_train):

    dataset = tfrecords.map(normalize_and_resize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    '''
    Only shuffle and repeat the dataset in training. The advantage to have a
    infinite dataset for training is to avoid the potential last partial batch
    in each epoch, so users don't need to think about scaling the gradients
    based on the actual batch size.
    '''
    if is_train:
        dataset = dataset.shuffle(200)
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

class Trainer(object):
    def __init__(self, model, epochs, global_batch_size, strategy, initial_learning_rate, start_epoch=1, tensorboard_dir='./logs'):
        self.start_epoch = start_epoch
        self.model = model
        self.epochs = epochs
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
        self.model = model

        self.current_learning_rate = initial_learning_rate
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.patience_count = 0
        self.max_patience = 10
        # self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        # self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        # self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.tensorboard_dir = tensorboard_dir
        self.best_model = None

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

    def compute_loss(self, labels, predictions):
        per_example_loss == self.loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

    def train_step(sefl, inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.compute_loss(labels, predictions)

        grads = tape.gradient(target=loss, sources=self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # self.train_accuracy.update_state(labels, predictions)
        
        return loss

    def test_step(self, inputs):
        images, labels = inputs

        predictions = self.model(images, training=False)
        t_loss = self.compute_loss(labels, predictions)

        # self.test_loss.update_state(t_loss)
        # self.test_accuracy.update_state(labels, predictions)
        return t_loss

    def run(self, train_dist_dataset, val_dist_dataset):

        @tf.function
        def distribute_train_epoch(dataset):
            tf.print('Start distributed training...')
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in dataset:
                per_replica_loss = self.strategy.experimental_run_v2(
                    self.train_step, args=(one_batch)
                )
                batch_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                total_loss += batch_loss
                num_train_batches += 1
                tf.print('Trained batch', num_train_batches, 'batch loss', batch_loss, 'epoch total loss', total_loss/num_train_batches)
            return total_loss, num_train_batches
        
        @tf.function
        def distribute_test_epoch(dataset):
            total_loss = 0.0
            num_test_batches = 0.0
            for one_batch in dataset:
                per_replica_loss = self.strategy.experimental_run_v2(
                    self.test_step, args=(one_batch, )
                )
                num_test_batches += 1
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
                )
                tf.print('Validated batch', num_test_batches, 'batch loss', batch_loss)
                if not tf.math.is_nan(batch_loss):
                    total_loss += batch_loss
                else:
                    num_test_batches -= 1
                
            return total_loss, num_test_batches
        
        summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)
        summary_writer.set_as_default()

        

    
def train(dataset, device, epochs, leargning_rate, batch_size, resolver=None):
    if device == 'GPU':
        strategy = tf.distribute.MirroredStrategy()
    elif device == 'TPU':
        strategy = tf.distribute.TPUStrategy(resolver)
    
    
    tfdatasets, info = tfds.load(name=dataset, with_info=True, as_supervised=True)

    ds_train, ds_test = tfdatasets['train'], tfdatasets['test']
    ds = ds_train.take(1)
    for image, label in ds:
        input_shape = image.shape

    BATCH_SIZE_PER_REPLICA = batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    if not os.path.exists(os.path.join('./models_save')):
        os.makedirs(os.path.join('./models_save/'))
    
    '''
    Create the model, optimizer and metrics inside strategy scope, so that the
    variables can be mirrored on each device.
    '''
    with strategy.scope():

        model = MobileNetV2(input_shape=input_shape, classes=10)
        training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('training_accuracy', dtype=tf.float32)

        train_dataset = strategy.experimental_distribute_datasets_from_function(lambda _: create_dataset(ds_train, BATCH_SIZE_PER_REPLICA, is_train='True'))
        val_dataset = strategy.experimental_distribute_datasets_from_function(lambda _:create_dataset(ds_test, BATCH_SIZE_PER_REPLICA, is_train='False'))

    @tf.function
    def train_step(iterator):
        '''The step function for one training step'''
        
        def step_fn(inputs):
            '''The computation to run on each device.'''
            images, labels = inputs
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
                loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            training_loss.update_state(loss * strategy.num_replicas_in_sync)
            training_accuracy.update_state(labels, logits)
        
        strategy.run(step_fn, args=(next(iterator),))
    
    steps_per_epoch = 10000 // batch_size

    train_iterator = iter(train_dataset)
    for epoch in range(epochs):
        print(f'Epoch : {epoch}/{epochs}')

        for step in range(steps_per_epoch):
            train_step(train_iterator)

        print(f'Current step: {optimizer.iterations.numpy()}, training loss: {round(float(training_loss.result()), 4)}, accuracy: {round(float(training_accuracy.result()) * 100, 2)}%')
        training_loss.reset_states()
        training_accuracy.reset_states()

    checkpoint_dir = './training_checkpoints'
    if not os.path.exists(os.path.join(checkpoint_dir)):
        os.makedirs(os.path.join(checkpoint_dir))

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    def decay(epoch):
        if epoch < 150:
            return leargning_rate
        elif epoch >= 150 and epoch < 180:
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


    # history_MBNetV2 = model.fit(
    # train_dataset, 
    # epochs=epochs,
    # validation_data=val_dataset,
    # verbose=1,
    # use_multiprocessing=True,
    # callbacks=callbacks
    # )
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='Select device for training')
    parser.add_argument('--dataset', help='cifar10, imagenet')
    parser.add_argument('--epochs', type=int, default=200, help='Epoch')
    parser.add_argument('--model', help='choose model')
    parser.add_argument('--batch_size', type=int, help='batch size of multi device')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.1, help='Learning Rate')
    
    args = parser.parse_args()
    
    if args.device == 'GPU':
        automatic_gpu_usage()
    elif args.device == 'TPU':
        resolver = automatic_tpu_usage()
    
    if args.device == 'GPU':
        train(dataset=args.dataset, device=args.device, epochs=args.epochs, leargning_rate=args.learning_rate, batch_size=args.batch_size)
    elif args.device == 'TPU':
        train(dataset=args.dataset, device=args.device, epochs=args.epochs, leargning_rate=args.learning_rate, batch_size=args.batch_size, resolver=resolver)
    