#' @title Generator
#'
#' @description Provides a torch::nn_module with a simple fully connected neural
#'   net, for use as the default architecture for tabular data in RGAN.
#'
#' @param noise_dim The length of the noise vector per example
#' @param data_dim The number of columns in the data set
#' @param hidden_units A list of the number of neurons per layer, the length of the list determines the number of hidden layers
#' @param dropout_rate The dropout rate for each hidden layer
#'
#' @return A torch::nn_module for the Generator
#' @export
Generator <- torch::nn_module(
  initialize = function(noise_dim,
                        data_dim,
                        hidden_units = list(128, 128),
                        dropout_rate = 0.5
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

                          # Add a LeakyReLU activation for better gradient flow
                          self$seq$add_module(module = torch::nn_leaky_relu(0.2),
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



#' @title DCGAN Generator
#'
#' @description Provides a torch::nn_module with a simple deep convolutional neural
#'   net architecture, for use as the default architecture for image data in RGAN.
#'   Architecture inspired by: [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
#'
#' @param noise_dim The length of the noise vector per example
#' @param number_channels The number of channels in the image (RGB is 3 channels)
#' @param ngf The number of feature maps in generator
#' @param dropout_rate The dropout rate for each hidden layer
#'
#' @return A torch::nn_module for the DCGAN Generator
#' @export
DCGAN_Generator <- torch::nn_module(
  initialize = function(noise_dim = 100,
                        number_channels = 3,
                        ngf = 64, # Number of feature maps in Generator
                        dropout_rate = 0.5
                        ) {
                        # Initialize an empty nn_sequential module
                        self$seq <- torch::nn_sequential()

                        self$seq$add_module(
                          module =  torch::nn_conv_transpose2d(noise_dim, ngf * 8, 4, 1, 0, bias = FALSE),
                          name = paste0("Conv", 1)
                        )



                        self$seq$add_module(module =  torch::nn_relu(TRUE),
                                            name = paste0("ReLU", 1))

                        self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                                            name = paste0("Dropout", 1))

                        self$seq$add_module(module =  torch::nn_batch_norm2d(ngf * 8),
                                            name = paste0("BatchNorm", 1))

                        self$seq$add_module(
                          module =  torch::nn_conv_transpose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = FALSE),
                          name = paste0("Conv", 2)
                        )



                        self$seq$add_module(module =  torch::nn_relu(TRUE),
                                            name = paste0("ReLU", 2))

                        self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                                            name = paste0("Dropout", 2))

                        self$seq$add_module(module =  torch::nn_batch_norm2d(ngf * 4),
                                            name = paste0("BatchNorm", 2))


                        self$seq$add_module(
                          module =  torch::nn_conv_transpose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = FALSE),
                          name = paste0("Conv", 3)
                        )



                        self$seq$add_module(module =  torch::nn_relu(TRUE),
                                            name = paste0("ReLU", 3))

                        self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                                            name = paste0("Dropout", 3))

                        self$seq$add_module(module =  torch::nn_batch_norm2d(ngf * 2),
                                            name = paste0("BatchNorm", 3))

                        self$seq$add_module(
                          module =  torch::nn_conv_transpose2d(ngf * 2, ngf, 4, 2, 1, bias = FALSE),
                          name = paste0("Conv", 4)
                        )



                        self$seq$add_module(module =  torch::nn_relu(TRUE),
                                            name = paste0("ReLU", 4))

                        self$seq$add_module(module =  torch::nn_dropout2d(p = dropout_rate),
                                            name = paste0("Dropout", 4))

                        self$seq$add_module(module =  torch::nn_batch_norm2d(ngf),
                                            name = paste0("BatchNorm", 4))

                        self$seq$add_module(
                          module =  torch::nn_conv_transpose2d(ngf, number_channels, 4, 2, 1, bias = FALSE),
                          name = paste0("Conv", 5)
                        )

                        self$seq$add_module(module =  torch::nn_tanh(),
                                            name = paste0("Output"))

                        },
forward = function(input) {
  input <- self$seq(input)
  input
}
  )


#' @title Tabular Generator with Gumbel-Softmax
#'
#' @description Provides a torch::nn_module Generator for tabular data that applies
#'   Gumbel-Softmax to categorical outputs for differentiable sampling. This improves
#'   gradient flow for discrete variables compared to standard softmax.
#'
#' @param noise_dim The length of the noise vector per example
#' @param output_info A list describing the output structure from data_transformer$output_info.
#'   Each element is a list with (dimension, type) where type is "linear", "mode_specific", or "softmax".
#' @param hidden_units A list of the number of neurons per layer
#' @param dropout_rate The dropout rate for each hidden layer
#' @param tau Temperature for Gumbel-Softmax. Lower values produce more discrete outputs. Defaults to 0.2.
#'
#' @return A torch::nn_module for the Tabular Generator
#' @export
TabularGenerator <- torch::nn_module(
  initialize = function(noise_dim,
                        output_info,
                        hidden_units = list(256, 256),
                        dropout_rate = 0.5,
                        tau = 0.2) {

    # Store output_info and tau for forward pass
    self$output_info <- output_info
    self$tau <- tau
    self$training_mode <- TRUE

    # Calculate total output dimension
    data_dim <- sum(sapply(output_info, function(x) x[[1]]))

    # Build the network
    self$seq <- torch::nn_sequential()

    dim <- noise_dim
    i <- 1

    for (neurons in hidden_units) {
      self$seq$add_module(
        module = torch::nn_linear(dim, neurons),
        name = paste0("Linear_", i)
      )
      self$seq$add_module(
        module = torch::nn_leaky_relu(0.2),
        name = paste0("Activation_", i)
      )
      self$seq$add_module(
        module = torch::nn_dropout(dropout_rate),
        name = paste0("Dropout_", i)
      )
      dim <- neurons
      i <- i + 1
    }

    # Output layer - no activation, will apply specific activations in forward
    self$seq$add_module(
      module = torch::nn_linear(dim, data_dim),
      name = "Output"
    )
  },

  forward = function(input, hard = NULL) {
    # Get raw output from network
    data <- self$seq(input)

    # Apply appropriate activation to each output block
    outputs <- list()
    start_idx <- 1

    for (info in self$output_info) {
      dim <- info[[1]]
      col_type <- info[[2]]
      end_idx <- start_idx + dim - 1

      # Slice the relevant columns (R uses 1-based indexing, torch uses 0-based)
      col_data <- data[, start_idx:end_idx]

      if (col_type == "softmax") {
        # Apply Gumbel-Softmax for categorical columns
        # Use hard=TRUE during inference (eval mode), soft during training
        use_hard <- if (!is.null(hard)) hard else !self$training
        col_data <- gumbel_softmax(col_data, tau = self$tau, hard = use_hard, dim = 2)
      } else if (col_type == "linear") {
        # For standard continuous columns, apply tanh to bound output
        col_data <- torch::torch_tanh(col_data)
      } else if (col_type == "mode_specific") {
        # For mode-specific: first n-1 columns are mode indicators (softmax),
        # last column is normalized value (tanh)
        if (dim > 1) {
          mode_logits <- col_data[, 1:(dim - 1)]
          value <- col_data[, dim, drop = FALSE]

          # Apply Gumbel-Softmax to mode indicators
          use_hard <- if (!is.null(hard)) hard else !self$training
          mode_probs <- gumbel_softmax(mode_logits, tau = self$tau, hard = use_hard, dim = 2)

          # Apply tanh to value
          value <- torch::torch_tanh(value)

          col_data <- torch::torch_cat(list(mode_probs, value), dim = 2)
        } else {
          # Single column mode_specific (shouldn't happen, but handle gracefully)
          col_data <- torch::torch_tanh(col_data)
        }
      }

      outputs[[length(outputs) + 1]] <- col_data
      start_idx <- end_idx + 1
    }

    # Concatenate all outputs
    return(torch::torch_cat(outputs, dim = 2))
  }
)
