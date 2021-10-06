#!/usr/bin/env python 
import argparse,logging,time,sys,os,json
sys.path.append('..')
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger(__name__)

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precisionss

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from tools.CalcMean import CalcMean

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_IMAGE_SIZE = 72
DEFAULT_PATCH_SIZE = 6
DEFAULT_PROJECTION_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_TRANSFORMER_LAYERS = 8
DEFUALT_OUTPUT = __file__.replace('.py','.json')
DEFAULT_INTEROP = int(os.cpu_count() / 4)
DEFAULT_INTRAOP = int(os.cpu_count() / 4)

def main():
   ''' horovod enabled implimentation of Vision Transformer DL training using MNIST data and the Tensorflow framework. '''
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-i','--input',help='path to mnist dataset on disk. Use "wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz" if you need to download it.',required=True)
   parser.add_argument('-e','--epochs',help='number of epochs to train [DEFAULT=%d]' % DEFAULT_EPOCHS,default=DEFAULT_EPOCHS,type=int)
   parser.add_argument('-b','--batch-size',help='batch size for training [DEFAULT=%d]' % DEFAULT_BATCH_SIZE,default=DEFAULT_BATCH_SIZE,type=int)
   parser.add_argument('-lr','--learning-rate',help='learning rate for training [DEFAULT=%d]' % DEFAULT_LEARNING_RATE,default=DEFAULT_LEARNING_RATE,type=int)
   parser.add_argument('-im','--image-size',help='Dimension image will be resized to for training [DEFAULT=%d]' % DEFAULT_IMAGE_SIZE,default=DEFAULT_IMAGE_SIZE,type=int)
   parser.add_argument('-p','--patch-size',help='Size of image patches [DEFAULT=%d]' % DEFAULT_PATCH_SIZE,default=DEFAULT_PATCH_SIZE,type=int)
   parser.add_argument('-pr','--projection-dim',help='Dimension that image patches will be projected into [DEFAULT=%d]' % DEFAULT_PROJECTION_DIM,default=DEFAULT_PROJECTION_DIM,type=int)
   parser.add_argument('-h','--num-heads',help='Number of attention heads in model [DEFAULT=%d]' % DEFAULT_NUM_HEADS,default=DEFAULT_NUM_HEADS,type=int)
   parser.add_argument('-tl','--transformer-layers',help='Number of transformer layers in model [DEFAULT=%d]' % DEFAULT_TRANSFORMER_LAYERS,default=DEFAULT_TRANSFORMER_LAYERS,type=int)
   parser.add_argument('-o','--output',help='output json filename where metrics will be stored[DEFAULT=%s]' % DEFUALT_OUTPUT,default=DEFUALT_OUTPUT)

   parser.add_argument('--interop',type=int,help='set Tensorflow "inter_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTEROP,default=DEFAULT_INTEROP)
   parser.add_argument('--intraop',type=int,help='set Tensorflow "intra_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTRAOP,default=DEFAULT_INTRAOP)
   parser.add_argument('--horovod', default=False, action='store_true', help="Use MPI with horovod")

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   hvd = None
   rank = 0
   nranks = 1
   logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
   if args.horovod:
      import horovod
      import horovod.tensorflow as hvd
      hvd.init()
      logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:' + (
                 '%05d' % hvd.rank()) + ':%(name)s:%(message)s'
      rank = hvd.rank()
      nranks = hvd.size()
      if rank > 0:
         logging_level = logging.WARNING
      # Horovod: pin GPU to be used to process local rank (one GPU per process)
      if gpus:
         print("hvd.local_rank: ", hvd.local_rank()) 
         tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
   
      # Setup Logging
   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
      os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)
   
   # report rank makeup
   if hvd:
      logging.warning('rank: %5d   size: %5d  local rank: %5d  local size: %5d', hvd.rank(), hvd.size(),
                      hvd.local_rank(), hvd.local_size())

   tf.config.threading.set_inter_op_parallelism_threads(args.interop)
   tf.config.threading.set_intra_op_parallelism_threads(args.intraop)

   logger.info('number of gpus:              %s',len(gpus))
   logger.info('input:                       %s',args.input)
   logger.info('epochs:                      %s',args.epochs)
   logger.info('batch size:                  %s',args.batch_size)
   logger.info('output:                      %s',args.output)
   logger.info('interop:                     %s',args.interop)
   logger.info('intraop:                     %s',args.intraop)
   logger.info('using tensorflow version:    %s (%s)',tf.__version__,tf.__git_version__)
   logger.info('using tensorflow from:       %s',tf.__file__)
   if hvd:
      logger.info('using horovod version:       %s',horovod.__version__)
      logger.info('using horovod from:          %s',horovod.__file__)

   output_data = {
      'input': args.input,
      'epochs': args.epochs,
      'batch_size': args.batch_size,
      'interop': args.interop,
      'intraop': args.intraop,
      'horovod': args.horovod,
      'tf_version': tf.__version__,
      'tf_path': tf.__file__,
      'nranks': nranks,
   }
   if hvd:
      output_data['hvd_version'] = horovod.__version__
      output_data['hvd_path'] = horovod.__file__


   # can use wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(args.input)
   x_train, x_test = x_train / 255.0, x_test / 255.0

   # Add a channels dimension
   x_train = x_train[..., tf.newaxis].astype("float32")
   x_test = x_test[..., tf.newaxis].astype("float32")

   train_ds = tf.data.Dataset.from_tensor_slices(
       (x_train, y_train)).shuffle(10000).batch(args.batch_size)

   test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)

   # Create an instance of the model
   model = create_vit_classifier()

   loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

   optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning-rate)

   # Add Horovod Distributed Optimizer
   if hvd:
      optimizer = hvd.DistributedOptimizer(optimizer)

   train_loss = tf.keras.metrics.Mean(name='train_loss')
   train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

   test_loss = tf.keras.metrics.Mean(name='test_loss')
   test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
   
   output_data['epoch_data'] = []
   first_batch = True
   for epoch in range(args.epochs):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

      train_img_per_sec = CalcMean()
      test_img_per_sec = CalcMean()
      
      batch_counter = 0
      for images, labels in train_ds:
         start = time.time()
         train_step(model,loss_object,optimizer,train_loss,train_accuracy,
                    images, labels, hvd, first_batch)
         duration = time.time() - start
         current_img_rate = args.batch_size / duration
         if hvd:
            current_img_rate *= hvd.size()
         # first few batches are slow due to compile time, so exclude them from the average
         if batch_counter > 10:
            train_img_per_sec.add_value(current_img_rate)
         first_batch = False
         batch_counter += 1

      batch_counter = 0
      for test_images, test_labels in test_ds:
         start = time.time()
         test_step(model,loss_object,test_loss,test_accuracy,test_images,test_labels)
         duration = time.time() - start
         current_img_rate = args.batch_size / duration
         if hvd:
            current_img_rate *= hvd.size()
         # first few batches are slow due to compile time, so exclude them from the average
         if batch_counter > 1:
            test_img_per_sec.add_value(current_img_rate)
         batch_counter += 1

      logger.info('[Epoch %02d] Train Loss: %10f Acc: %10f ImgRate: %10f =  Test Loss: %10f Acc: %10f ImgRate: %10f',
         epoch, train_loss.result(), train_accuracy.result(), train_img_per_sec.mean(),
         test_loss.result(), test_accuracy.result(), test_img_per_sec.mean())

      output_data['epoch_data'].append({
         'epoch': epoch,
         'train_loss': float(train_loss.result()),
         'train_acc': float(train_accuracy.result()),
         'train_img_per_sec_mean': train_img_per_sec.mean(),
         'train_img_per_sec_sigma': train_img_per_sec.sigma(),
         'test_loss': float(test_loss.result()),
         'test_acc': float(test_accuracy.result()),
         'test_img_per_sec_mean': test_img_per_sec.mean(),
         'test_img_per_sec_sigma': test_img_per_sec.sigma(),
         })

   if rank == 0:
      json.dump(output_data,open(args.output,'w'), sort_keys=True, indent=2)
   

##############
num_classes = 10
input_shape = (28,28,1)

#weight_decay = 0.0001
#batch_size = 1
#num_epochs = 100
image_size = args.image-size  # We'll resize input images to this size
patch_size = args.patch-size  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = args.projection-dim
num_heads = args.num_heads
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = args.transformer-layers

mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    resized = layers.Resizing(image_size, image_size)(inputs)

    #TODO:
    # Augment data.
    #augmented = data_augmentation(inputs)

    # Create patches.
    #patches = Patches(patch_size)(augmented)
    patches = Patches(patch_size)(resized)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print("encoded patches: ", encoded_patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
##############


@tf.function
def train_step(model,loss_object,optimizer,train_loss,train_accuracy,
               images, labels, hvd, first_batch):
   with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = model(images, training=True)
      loss = loss_object(labels, predictions)

   # Horovod: add Horovod Distributed GradientTape.
   if hvd:
      tape = hvd.DistributedGradientTape(tape)

   gradients = tape.gradient(loss, model.trainable_variables)
   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

   train_loss(loss)
   train_accuracy(labels, predictions)

   # Horovod: broadcast initial variable states from rank 0 to all other processes.
   # This is necessary to ensure consistent initialization of all workers when
   # training is started with random weights or restored from a checkpoint.
   #
   # Note: broadcast should be done after the first gradient step to ensure optimizer
   # initialization.
   if first_batch and hvd:
      hvd.broadcast_variables(model.variables, root_rank=0)
      hvd.broadcast_variables(optimizer.variables(), root_rank=0)


@tf.function
def test_step(model,loss_object,test_loss,test_accuracy,images, labels):
   # training=False is only needed if there are layers with different
   # behavior during training versus inference (e.g. Dropout).
   predictions = model(images, training=False)
   t_loss = loss_object(labels, predictions)

   test_loss(t_loss)
   test_accuracy(labels, predictions)


if __name__ == "__main__":
   main()
