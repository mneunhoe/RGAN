#' @title Generator
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
#' @param noise_dim a
#' @param data_dim b
#' @param hidden_units list of number of neurons per layer the length of the list determines the number of hidden layers
#' @param dropout_rate dropout for each hidden layer
#'
#' @return A neural net class for the Generator
#' @export
Generator <- torch::nn_module(
  initialize = function(noise_dim, # The length of our noise vector per example
                        data_dim, # The number of columns in our data
                        hidden_units = list(128, 128), # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        dropout_rate = 0.5 # The dropout probability
  ) {
    # Initialize an empty nn_sequential module
    self$seq <- torch::nn_sequential()

    # For the hidden layers we need to keep track of our input and output dimensions. The first input will be our noise vector, therefore, it will be noise_dim
    dim <- noise_dim

    # i will be a simple counter to keep track of our network depth
    i <- 1

    # Now we loop over the list of hidden units and add the hidden layers to the nn_sequential module
    for (neurons in hidden_units) {
      # First, we add a ResidualBlock of the respective size.
      self$seq$add_module(module =  torch::nn_linear(dim, neurons),
                          name = paste0("Linear_", i))

      # Add a leakyReLU activation
      self$seq$add_module(module = torch::nn_relu(),
                          name = paste0("Activation_", i))
      # And then a Dropout layer.
      self$seq$add_module(module = torch::nn_dropout(dropout_rate),
                          name = paste0("Dropout_", i))
      # Now we update our dim for the next hidden layer.
      # Since it will be another ResidualBlock the input dimension will be dim+neurons
      dim <- neurons
      # Update the counter
      i <- i + 1
    }
    # Finally, we add the output layer. The output dimension must be the same as our data dimension (data_dim).
    self$seq$add_module(module = torch::nn_linear(dim, data_dim),
                        name = "Output")
  },
  forward = function(input) {
    input <- self$seq(input)
    input
  }
)
