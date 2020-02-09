from build_model import model_tools
import tensorflow as tf
model=model_tools()

def generate_model(images_ph,number_of_classes):
        #generate_model(self, layer, kernel, input_depth, output_depth, stride_size)
        network=model.conv_layer(images_ph,11,3,96,4)
        #55x55x96
        network=model.pooling_layer(network,3,2)
        #27x27x96
        network=model.activation_layer(network)
        print(network)

        #level 2 convolution
        network=model.conv_layer(network,5,96,256,1)
        network=model.pooling_layer(network,3,2)
        #13x13x256
        network=model.activation_layer(network)
        print(network)

        #level 3 convolution
        network=model.conv_layer(network,3,256,384,1)
        network=model.activation_layer(network)
        print(network)

        #level 4 convolution
        network=model.conv_layer(network,3,384,384,1)
        network=model.activation_layer(network)
        print(network)

        #level 5 convolution
        network=model.conv_layer(network,3,384,256,1)
        #13x13x256
        network=model.pooling_layer(network,3,2)
        #6x6x256
        network=model.activation_layer(network)
        print(network)

        #flattening layer
        network,features=model.flattening_layer(network)
        print(network)

        #fully connected layer
        network=model.fully_connected_layer(network,features,4096)
        #network.size = 4096
        network=model.activation_layer(network)
        print(network)

        #fully connected layer
        network=model.fully_connected_layer(network,4096,4096)
        #network.size = 4096
        network=model.activation_layer(network)
        print(network)

        #output layer
        network=model.fully_connected_layer(network,4096,1000)
        network=tf.nn.softmax(network)
        print(network)
