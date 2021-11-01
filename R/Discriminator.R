#' @title Discriminator
#'
#' @description Provides a class to transform data for RGAN.
#'   Method `$new()` initializes a new transformer, method `$fit(data)` learns
#'   the parameters for the transformation from data (e.g. means and sds).
#'   Methods `$transform()` and `$inverse_transform()` can be used to transform
#'   and back transform a data set based on the learned parameters.
#'   Currently, DataTransformer supports z-transformation (a.k.a. normalization)
#'   for numerical features/variables and one hot encoding for categorical
#'   features/variables. In your call to fit you just need to indicate which
#'   columns contain discrete features.
#'
#' @return A neural net for the Discriminator
#' @export
Discriminator <- torch::nn_module(
  initialize = function(data_dim, # The number of columns in our data
                        hidden_units = list(128, 128), # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        dropout_rate = 0.5, # The dropout probability
                        sigmoid = FALSE
  ) {

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
    if(sigmoid){
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
#' @description Provides a class to transform data for RGAN.
#'   Method `$new()` initializes a new transformer, method `$fit(data)` learns
#'   the parameters for the transformation from data (e.g. means and sds).
#'   Methods `$transform()` and `$inverse_transform()` can be used to transform
#'   and back transform a data set based on the learned parameters.
#'   Currently, DataTransformer supports z-transformation (a.k.a. normalization)
#'   for numerical features/variables and one hot encoding for categorical
#'   features/variables. In your call to fit you just need to indicate which
#'   columns contain discrete features.
#'
#' @return A neural net for the DCGAN Discriminator
#' @export
DCGAN_Discriminator <- torch::nn_module(
  initialize = function(image_size = 64, # The number of columns in our data
                        number_channels = 3, # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        ndf = 64,
                        dropout_rate = 0.5
  ) {

    # Initialize an empty nn_sequential module
    self$seq <- torch::nn_sequential()

    # First, we add a ResidualBlock of the respective size.
    self$seq$add_module(
      module =  torch::nn_conv2d(number_channels, ndf, 4, 2, 1, bias = FALSE),
      name = paste0("Conv", 1)
    )

    self$seq$add_module(module =  torch::nn_leaky_relu(0.2, inplace = TRUE),
                        name = paste0("LeakyReLU", 1))

    self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                        name = paste0("Dropout", 1))


    self$seq$add_module(
      module =  torch::nn_conv2d(ndf, ndf * 2, 4, 2, 1, bias = FALSE),
      name = paste0("Conv", 2)
    )

    self$seq$add_module(module =  torch::nn_leaky_relu(0.2, inplace = TRUE),
                        name = paste0("LeakyReLU", 2))

    self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                        name = paste0("Dropout", 2))

    self$seq$add_module(module =  torch::nn_batch_norm2d(ndf * 2),
                        name = paste0("BatchNorm", 2))






    self$seq$add_module(
      module =  torch::nn_conv2d(ndf*2, ndf * 4, 4, 2, 1, bias = FALSE),
      name = paste0("Conv", 3)
    )



    self$seq$add_module(module =  torch::nn_leaky_relu(0.2, inplace = TRUE),
                        name = paste0("LeakyReLU", 3))

    self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                        name = paste0("Dropout", 3))

    self$seq$add_module(module =  torch::nn_batch_norm2d(ndf * 4),
                        name = paste0("BatchNorm", 3))

    self$seq$add_module(
      module =  torch::nn_conv2d(ndf*4, ndf * 8, 4, 2, 1, bias = FALSE),
      name = paste0("Conv", 4)
    )



    self$seq$add_module(module =  torch::nn_leaky_relu(0.2, inplace = TRUE),
                        name = paste0("LeakyReLU", 4))

    self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                        name = paste0("Dropout", 4))


    self$seq$add_module(module =  torch::nn_batch_norm2d(ndf * 8),
                        name = paste0("BatchNorm", 4))

    self$seq$add_module(
      module =  torch::nn_conv2d(ndf*8, 1, 4, 1, 0, bias = FALSE),
      name = paste0("Conv", 5)
    )

    # self$seq$add_module(
    #   module =  torch::nn_sigmoid(),
    #   name = paste0("Output")
    # )



  },
  forward = function(input) {
    data <- self$seq(input)
    data
  }
)


