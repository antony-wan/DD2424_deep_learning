from build_model import Layers
import tensorflow as tf

IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
NUM_CLASSES = 10

def model():

    model = Layers()

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS], name='Input')
    y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='Output')
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS], name='images')

    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    # CONV LAYER 1
    network = model.conv_layer(layer=x_image, kernel_size=3, input_depth=3, output_depth=96, stride_size=2)
    network = model.activation_layer(network)
    # MAX POOLING 1
    network = model.pooling_layer(layer=network, kernel_size=3, stride_size=2)
    print(network)

    # CONV LAYER 2
    network = model.conv_layer(layer=network, kernel_size=3, input_depth=96, output_depth=256, stride_size=1)
    network = model.activation_layer(network)
    # MAX POOLING 2
    network = model.pooling_layer(network, kernel_size=3, stride_size=2)
    print(network)

    # CONV LAYER 3
    network = model.conv_layer(layer=network, kernel_size=3, input_depth=256, output_depth=384, stride_size=1)
    network = model.activation_layer(network)
    print(network)

    # CONV LAYER 4
    network = model.conv_layer(layer=network, kernel_size=3, input_depth=384, output_depth=384, stride_size=1)
    network = model.activation_layer(network)
    print(network)

    # CONV LAYER 5
    network = model.conv_layer(layer=network, kernel_size=3, input_depth=384, output_depth=256, stride_size=1)
    network = model.activation_layer(network)
    # MAX POOLING 5
    network = model.pooling_layer(layer=network, kernel_size=3, stride_size=2)
    print(network)

    # flattening layer
    network, features = model.flattening_layer(network)
    print(network)

    tf.nn.dropout(network, keep_prob=0.3)

    # fully connected layer
    network = model.fully_connected_layer(network, features, 4096)
    network = model.activation_layer(network)
    print(network)

    tf.nn.dropout(network, keep_prob=0.3)

    # fully connected layer
    network = model.fully_connected_layer(network, 4096, 4096)
    network = model.activation_layer(network)
    print(network)

    # output layer
    network = model.fully_connected_layer(network, 4096, NUM_CLASSES)
    print(network)

    y_pred = tf.argmax(network, axis=1)

    return x, y, network, y_pred, global_step