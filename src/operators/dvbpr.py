# Libraries
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Concatenate, Dot, ZeroPadding2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Flatten, Dense


# Model definition - Convolutional Neural Network
class CNN:

    @staticmethod
    def build(width,
              height,
              depth,
              latent_dim,
              w_init="RandomNormal",
              cnn_w_regularizer=None,
              fc_w_regularizer=None,
              b_init="RandomNormal"):
        """
        Build the CNN.

            :param width (int): Image width in pixels.
            :param height (int): The image height in pixels.
            :param depth (int): The number of channels for the image.
            :param latent_dim (int): Dimesion of the latent space - embedding of the image.
            :param w_init="he_normal" (str): The kernel initializer.
            :param cnn_w_regularizer=None (str): Regularization method.
            :param fc_w_regularizer=None (str): Regularization method.
        """

        # Initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential(name='cnn')
        input_shape = (height, width, depth)
        chan_dim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1

        # conv1
        #
        # Our first CONV layer will learn a total of 64 filters, each
        # of which are 11x11 -- we'll then apply 4x4 strides to reduce
        # the spatial dimensions of the volume
        # Moreover, a max-pooling layer is added
        model.add(Conv2D(64, (11, 11),
                         strides=(4, 4),
                         padding="valid",
                         kernel_initializer=w_init,
                         kernel_regularizer=cnn_w_regularizer,
                         bias_initializer=b_init,
                         input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               padding="same"))

        # conv2
        #
        # Here we stack one more CONV layer on top,
        # each layer will learn a total of 256 (5x5) filters
        # A max-pooling layer is added
        model.add(ZeroPadding2D(padding=(2, 2)))
        model.add(Conv2D(256, (5, 5),
                         strides=(1, 1),
                         kernel_initializer=w_init,
                         kernel_regularizer=cnn_w_regularizer,
                         bias_initializer=b_init))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               padding="same"))

        # conv3
        #
        # Stack one more CONV layer, keeping 256 total learned filters
        # but decreasing the the size of each filter to 3x3
        model.add(ZeroPadding2D(padding=(1, 1)))
        model.add(Conv2D(256, (3, 3),
                         strides=(1, 1),
                         kernel_initializer=w_init,
                         kernel_regularizer=cnn_w_regularizer,
                         bias_initializer=b_init))
        model.add(Activation("relu"))

        # Two more CONV layers, same filter size and number
        #
        # conv4
        model.add(ZeroPadding2D(padding=(1, 1)))
        model.add(Conv2D(256, (3, 3),
                         strides=(1, 1),
                         kernel_initializer=w_init,
                         kernel_regularizer=cnn_w_regularizer,
                         bias_initializer=b_init))
        model.add(Activation("relu"))

        # conv5
        model.add(ZeroPadding2D(padding=(1, 1)))
        model.add(Conv2D(256, (3, 3),
                         strides=(1, 1),
                         kernel_initializer=w_init,
                         kernel_regularizer=cnn_w_regularizer,
                         bias_initializer=b_init))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               padding="same"))

        # Two fully-connected layers on top of each other
        #
        # full1
        model.add(Flatten())
        model.add(Dense(4096,
                        kernel_initializer=w_init,
                        kernel_regularizer=fc_w_regularizer,
                        bias_initializer=b_init))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # full2
        model.add(Dense(4096,
                        kernel_initializer=w_init,
                        kernel_regularizer=fc_w_regularizer,
                        bias_initializer=b_init))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # full3
        model.add(Dense(latent_dim,
                        kernel_initializer=w_init,
                        kernel_regularizer=fc_w_regularizer,
                        bias_initializer=b_init))

        # Any classifier layer (e.g. see softmax below) is added
        # since getting an embedding model is the goal here but solving a prediction task
        # model.add(Dense(classes))
        # model.add(Activation("softmax"))

        # Return the constructed network architecture
        return model


# Model definition - Convolutional Siamese Network
class ConvSiameseNet:

    @staticmethod
    def build(users_dim,
              width,
              height,
              depth,
              latent_dim,
              w_init="RandomNormal",
              cnn_w_regularizer=None,
              fc_w_regularizer=None,
              u_w_regularizer=None,
              b_init="RandomNormal"):

        # Define the input
        #   Unlike the Sequential model, you must create and define
        #   a standalone "Input" layer that specifies the shape of input
        #   data. The input layer takes a "shape" argument, which is a
        #   tuple that indicates the dimensionality of the input data.
        user_input = Input((1,))
        user_E = Input((users_dim * latent_dim, latent_dim))

        image_shape = (width, height, depth)
        left_input = Input(image_shape)
        right_input = Input(image_shape)

        # Build convnet to use in each siamese 'leg'
        conv_net = CNN.build(width,
                             height,
                             depth,
                             latent_dim,
                             w_init,
                             cnn_w_regularizer,
                             fc_w_regularizer,
                             b_init)

        # Connecting layers
        #   The layers in the model are connected pairwise.
        #   This is done by specifying where the input comes from when
        #   defining each new layer. A bracket notation is used, such that
        #   after the layer is created, the layer from which the input to
        #   the current layer comes from is specified.
        #
        # merge the two encoded inputs through the L1 distance
        L1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))

        # user's preferences theta_u
        theta_user = []
        for u in range(users_dim):
            theta_user.append(Dense(latent_dim,
                                    kernel_initializer=w_init,
                                    kernel_regularizer=u_w_regularizer,
                                    bias_initializer=b_init))

        # concatenate all users' preferences vectors to get the
        # users' preferences matrix Theta
        concatenate = Concatenate(axis=-1)

        # single user's preferences theta_u
        user_preference = Dot(axes=1)

        # preference layer
        preference_relationship = Dot(axes=1)

        # Apply the pipeline to the inputs
        #
        # call the convnet Sequential model on each of the input tensors
        # so params will be shared
        encoded_l = conv_net(left_input)
        encoded_r = conv_net(right_input)

        # merge the two encoded inputs through the L1 distance
        L1_dist = L1_distance([encoded_l, encoded_r])

        # concatenate user's preferences theta_u to get the preferences
        # matrix Theta
        theta_urs = []
        for u in range(users_dim):
            theta_urs.append(theta_user[u](user_input))
        theta = concatenate(theta_urs)

        # retrieve the single user preference
        theta_ur = user_preference([user_E, theta])

        # get the preference score
        prediction = preference_relationship([theta_ur, L1_dist])

        # Create the model
        #   After creating all of your model layers and connecting them
        #   together, you must then define the model.
        #   As with the Sequential API, the model is the thing that you can
        #   summarize, fit, evaluate, and use to make predictions.
        #   Keras provides a "Model" class that you can use to create a model
        #   from your created layers. It requires that you only specify the
        #   input and output layers.
        model = Model(inputs=[user_input, user_E, left_input, right_input],
                      outputs=prediction)
        return model
