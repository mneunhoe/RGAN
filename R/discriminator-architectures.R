#' @title Discriminator
#'
#' @description Provides a torch::nn_module with a simple fully connected neural
#'   net, for use as the default architecture for tabular data in RGAN.
#'
#' @param data_dim The number of columns in the data set
#' @param hidden_units A list of the number of neurons per layer, the length of the list determines the number of hidden layers
#' @param dropout_rate The dropout rate for each hidden layer
#' @param sigmoid Switch between a sigmoid and linear output layer (the sigmoid is needed for the original GAN value function)
#'
#' @return A torch::nn_module for the Discriminator
#' @export
Discriminator <- torch::nn_module(
  initialize = function(data_dim,
                        hidden_units = list(128, 128),
                        dropout_rate = 0.5,
                        sigmoid = FALSE) {
    # Initialize an empty nn_sequential module
    self$seq <- torch::nn_sequential()

    # For the hidden layers we need to keep track of our input and output dimensions. The first input will be our noise vector, therefore, it will be noise_dim
    dim <- data_dim

    # i will be a simple counter to keep track of our network depth
    i <- 1

    # Now we loop over the list of hidden units and add the hidden layers to the nn_sequential module
    for (neurons in hidden_units) {
      # We start with a fully connected linear layer
      self$seq$add_module(module = torch::nn_linear(dim, neurons),
                          name = paste0("Linear_", i))
      # Add a leakyReLU activation
      self$seq$add_module(module = torch::nn_relu(),
                          name = paste0("Activation_", i))
      # And a Dropout layer
      self$seq$add_module(module = torch::nn_dropout(dropout_rate),
                          name = paste0("Dropout_", i))
      # Update the input dimension to the next layer
      dim <- neurons
      # Update the counter
      i <- i + 1
    }
    # Add an output layer to the net. Since it will be one score for each example we only need a dimension of 1.
    self$seq$add_module(module = torch::nn_linear(dim, 1),
                        name = "Output")
    if (sigmoid) {
      self$seq$add_module(module = torch::nn_sigmoid(),
                          name = "Sigmoid_Output")
    }

  },
  forward = function(input) {
    data <- self$seq(input)
    data
  }
)


#' @title DCGAN Discriminator
#'
#' @description Provides a torch::nn_module with a simple deep convolutional neural
#'   net architecture, for use as the default architecture for image data in RGAN.
#'   Architecture inspired by: [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
#'
#' @param number_channels The number of channels in the image (RGB is 3 channels)
#' @param ndf The number of feature maps in discriminator
#' @param dropout_rate The dropout rate for each hidden layer
#' @param sigmoid Switch between a sigmoid and linear output layer (the sigmoid is needed for the original GAN value function)
#'
#' @return A torch::nn_module for the DCGAN Discriminator
#' @export
DCGAN_Discriminator <- torch::nn_module(
  initialize = function(number_channels = 3,
                        # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        ndf = 64,
                        dropout_rate = 0.5,
                        sigmoid = FALSE) {
    # Initialize an empty nn_sequential module
    self$seq <- torch::nn_sequential()

    # First, we add a ResidualBlock of the respective size.
    self$seq$add_module(
      module =  torch::nn_conv2d(number_channels, ndf, 4, 2, 1, bias = FALSE),
      name = paste0("Conv", 1)
    )

    self$seq$add_module(
      module =  torch::nn_leaky_relu(0.2, inplace = TRUE),
      name = paste0("LeakyReLU", 1)
    )

    self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                        name = paste0("Dropout", 1))


    self$seq$add_module(
      module =  torch::nn_conv2d(ndf, ndf * 2, 4, 2, 1, bias = FALSE),
      name = paste0("Conv", 2)
    )

    self$seq$add_module(
      module =  torch::nn_leaky_relu(0.2, inplace = TRUE),
      name = paste0("LeakyReLU", 2)
    )

    self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                        name = paste0("Dropout", 2))

    self$seq$add_module(module =  torch::nn_batch_norm2d(ndf * 2),
                        name = paste0("BatchNorm", 2))






    self$seq$add_module(
      module =  torch::nn_conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = FALSE),
      name = paste0("Conv", 3)
    )



    self$seq$add_module(
      module =  torch::nn_leaky_relu(0.2, inplace = TRUE),
      name = paste0("LeakyReLU", 3)
    )

    self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                        name = paste0("Dropout", 3))

    self$seq$add_module(module =  torch::nn_batch_norm2d(ndf * 4),
                        name = paste0("BatchNorm", 3))

    self$seq$add_module(
      module =  torch::nn_conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = FALSE),
      name = paste0("Conv", 4)
    )



    self$seq$add_module(
      module =  torch::nn_leaky_relu(0.2, inplace = TRUE),
      name = paste0("LeakyReLU", 4)
    )

    self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                        name = paste0("Dropout", 4))


    self$seq$add_module(module =  torch::nn_batch_norm2d(ndf * 8),
                        name = paste0("BatchNorm", 4))

    self$seq$add_module(
      module =  torch::nn_conv2d(ndf * 8, 1, 4, 1, 0, bias = FALSE),
      name = paste0("Conv", 5)
    )


    if (sigmoid) {
      self$seq$add_module(module =  torch::nn_sigmoid(),
                          name = paste0("Output"))

    }

  },
  forward = function(input) {
    data <- self$seq(input)
    data
  }
)
