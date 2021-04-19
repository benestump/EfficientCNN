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
    
    if is_train:
        dataset = dataset.shuffle(200)
    
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
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        # self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
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
        per_example_loss =self.loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

    def train_step(self, inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.compute_loss(labels, predictions)

        grads = tape.gradient(target=loss, sources=self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_accuracy.update_state(labels, predictions)
        
        return loss

    def test_step(self, inputs):
        images, labels = inputs

        predictions = self.model(images, training=False)
        loss = self.compute_loss(labels, predictions)

        self.test_loss.update_state(loss)
        self.test_accuracy.update_state(labels, predictions)

    def run(self, train_dist_dataset, val_dist_dataset):
        
        @tf.function
        def distribute_train_step(dataset):
            per_replica_loss = self.strategy.run(self.train_step, args=(dataset,))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        
        @tf.function
        def distribute_test_step(dataset):
            return self.strategy.run(self.test_step, args=(dataset, ))
        
        summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)
        summary_writer.set_as_default()

        for epoch in range(self.start_epoch, self.epochs + 1):
            tf.summary.experimental.set_step(epoch)
            total_loss = 0.0
            num_batches = 0
            
            self.lr_decay()
            tf.summary.scalar('epoch learnig rage', self.current_learning_rate)

            print(f'Start epoch {epoch} with learning rate {self.current_learning_rate}')
            for x in train_dist_dataset:  
                total_loss += distribute_train_step(x)
                num_batches += 1
                
            train_loss = total_loss / num_batches
            tf.summary.scalar('epoch train loss', train_loss)

            for x in val_dist_dataset:
                distribute_test_step(x)
            tf.summary.scalar('epoch val loss', self.test_loss.result())

            print(f'epoch {epoch}, train_loss: {train_loss}, train_accuracy: {self.train_accuracy.result()*100}, test_loss: {self.test_loss.result()}, test_accuracy: {self.test_accuracy.result()*100}')

            self.test_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()

            if self.test_loss.result() < self.lowest_val_loss:
                self.save_model(epoch, self.test_loss.result())
                self.lowest_val_loss = self.test_loss.result()
            self.last_val_loss = self.test_loss.result()

        return self.best_model
    
    def save_model(self, epoch, loss):
        model_name = f'./models_save/model-epoch-{epoch}-loss-{loss:.4f}.h5'
        self.model.save_weights(model_name)
        self.best_model = model_name
        print(f'Model {model_name} saved')
    
def train(model, dataset, device, epochs, leargning_rate, batch_size, resolver=None):
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

        if model == 'MobileNetV2':
            model = MobileNetV2(input_shape=input_shape, classes=10)

        train_dist_dataset = strategy.experimental_distribute_datasets_from_function(lambda _: create_dataset(ds_train, BATCH_SIZE_PER_REPLICA, is_train='True'))
        val_dist_dataset = strategy.experimental_distribute_datasets_from_function(lambda _:create_dataset(ds_test, BATCH_SIZE_PER_REPLICA, is_train='False'))

        trainer = Trainer(
            model,
            epochs,
            GLOBAL_BATCH_SIZE,
            strategy,
            initial_learning_rate=leargning_rate,         
        )

    print('Start training...')
    return trainer.run(train_dist_dataset, val_dist_dataset)
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='Select device for training')
    parser.add_argument('--dataset', help='cifar10, imagenet')
    parser.add_argument('--epochs', type=int, default=200, help='Epoch')
    parser.add_argument('--model', help='MobileNetV2')
    parser.add_argument('--batch_size', type=int, help='batch size of multi device')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.1, help='Learning Rate')
    
    args = parser.parse_args()
    
    if args.device == 'GPU':
        automatic_gpu_usage()
    elif args.device == 'TPU':
        resolver = automatic_tpu_usage()
    
    if args.device == 'GPU':
        train(model=args.model, dataset=args.dataset, device=args.device, epochs=args.epochs, leargning_rate=args.learning_rate, batch_size=args.batch_size)
    elif args.device == 'TPU':
        train(model=args.model, dataset=args.dataset, device=args.device, epochs=args.epochs, leargning_rate=args.learning_rate, batch_size=args.batch_size, resolver=resolver)
    