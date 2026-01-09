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


#' @title Self-Attention Layer for Tabular Data
#'
#' @description Multi-head self-attention layer that captures relationships between
#'   features in the hidden representation. This allows the generator to learn
#'   dependencies between different columns (e.g., age correlates with income).
#'
#' @param embed_dim The dimension of the input embeddings (hidden layer size)
#' @param num_heads Number of attention heads. Must divide embed_dim evenly. Defaults to 4.
#' @param dropout Dropout rate for attention weights. Defaults to 0.1.
#'
#' @return A torch::nn_module for self-attention
#' @keywords internal
SelfAttention <- torch::nn_module(
  initialize = function(embed_dim, num_heads = 4, dropout = 0.1) {
    # Ensure embed_dim is divisible by num_heads
    if (embed_dim %% num_heads != 0) {
      # Adjust num_heads to be a divisor of embed_dim
      possible_heads <- c(1, 2, 4, 8, 16)
      num_heads <- max(possible_heads[embed_dim %% possible_heads == 0])
    }

    self$embed_dim <- embed_dim
    self$num_heads <- num_heads
    self$head_dim <- embed_dim %/% num_heads

    # Query, Key, Value projections
    self$q_proj <- torch::nn_linear(embed_dim, embed_dim)
    self$k_proj <- torch::nn_linear(embed_dim, embed_dim)
    self$v_proj <- torch::nn_linear(embed_dim, embed_dim)

    # Output projection
    self$out_proj <- torch::nn_linear(embed_dim, embed_dim)

    # Dropout
    self$dropout <- torch::nn_dropout(dropout)

    # Layer normalization for residual connection
    self$layer_norm <- torch::nn_layer_norm(embed_dim)
  },

  forward = function(x) {
    # x shape: (batch_size, embed_dim)
    # For tabular data, we treat the hidden representation as a single "token"
    # and apply self-attention across the feature dimension

    batch_size <- x$shape[1]

    # Store residual
    residual <- x

    # Reshape for multi-head attention: (batch, 1, embed_dim) -> treat as sequence of 1
    # Actually for tabular, we can reshape features into groups
    # Let's reshape (batch, embed_dim) -> (batch, num_heads, head_dim)
    # and apply attention across head dimension

    # Project to Q, K, V
    q <- self$q_proj(x)$view(c(batch_size, self$num_heads, self$head_dim))
    k <- self$k_proj(x)$view(c(batch_size, self$num_heads, self$head_dim))
    v <- self$v_proj(x)$view(c(batch_size, self$num_heads, self$head_dim))

    # Compute attention scores: (batch, num_heads, head_dim) @ (batch, head_dim, num_heads)
    # -> (batch, num_heads, num_heads)
    scale <- sqrt(self$head_dim)
    attn_weights <- torch::torch_bmm(q, k$transpose(2, 3)) / scale
    attn_weights <- torch::nnf_softmax(attn_weights, dim = -1)
    attn_weights <- self$dropout(attn_weights)

    # Apply attention to values
    # (batch, num_heads, num_heads) @ (batch, num_heads, head_dim) -> (batch, num_heads, head_dim)
    attn_output <- torch::torch_bmm(attn_weights, v)

    # Reshape back: (batch, num_heads, head_dim) -> (batch, embed_dim)
    attn_output <- attn_output$view(c(batch_size, self$embed_dim))

    # Output projection
    attn_output <- self$out_proj(attn_output)

    # Residual connection and layer norm
    output <- self$layer_norm(residual + attn_output)

    return(output)
  }
)


#' @title Residual Block for Generator
#'
#' @description A residual block with configurable normalization and activation.
#'   Used internally by TabularGenerator for CTGAN-style architecture.
#'
#' @param input_dim Input dimension
#' @param output_dim Output dimension
#' @param normalization Type of normalization: "batch", "layer", or "none"
#' @param activation Activation function: "relu", "leaky_relu", "gelu", or "silu"
#' @param dropout_rate Dropout rate (only used when normalization is "none")
#'
#' @return A torch::nn_module for the residual block
#' @keywords internal
ResidualBlock <- torch::nn_module(
  initialize = function(input_dim,
                        output_dim,
                        normalization = "batch",
                        activation = "relu",
                        dropout_rate = 0.0) {

    self$input_dim <- input_dim
    self$output_dim <- output_dim
    self$use_residual <- (input_dim == output_dim)

    # Linear layer
    self$linear <- torch::nn_linear(input_dim, output_dim)

    # Normalization
    self$normalization <- normalization
    if (normalization == "batch") {
      self$norm <- torch::nn_batch_norm1d(output_dim)
    } else if (normalization == "layer") {
      self$norm <- torch::nn_layer_norm(output_dim)
    } else {
      self$norm <- NULL
      # Use dropout when no normalization
      if (dropout_rate > 0) {
        self$dropout <- torch::nn_dropout(dropout_rate)
      } else {
        self$dropout <- NULL
      }
    }

    # Activation
    self$activation_type <- activation
    self$act <- switch(
      activation,
      "relu" = torch::nn_relu(),
      "leaky_relu" = torch::nn_leaky_relu(0.2),
      "gelu" = torch::nn_gelu(),
      "silu" = torch::nn_silu(),
      torch::nn_relu()  # default
    )
  },

  forward = function(x) {
    out <- self$linear(x)

    if (!is.null(self$norm)) {
      out <- self$norm(out)
    }

    out <- self$act(out)

    if (!is.null(self$dropout)) {
      out <- self$dropout(out)
    }

    # Residual connection (only if dimensions match)
    if (self$use_residual) {
      out <- out + x
    }

    return(out)
  }
)


#' @title Tabular Generator with Gumbel-Softmax
#'
#' @description Provides a torch::nn_module Generator for tabular data that applies
#'   Gumbel-Softmax to categorical outputs for differentiable sampling. This improves
#'   gradient flow for discrete variables compared to standard softmax.
#'
#'   Supports state-of-the-art architectural choices from CTGAN and other modern
#'   tabular GAN architectures:
#'   \itemize{
#'     \item \strong{Residual connections:} Skip connections that improve gradient flow
#'       in deeper networks (enabled by default when consecutive layers have same width)
#'     \item \strong{Batch Normalization:} Stabilizes training (CTGAN default)
#'     \item \strong{Layer Normalization:} Alternative that works better with small batches
#'     \item \strong{Multiple activation functions:} ReLU, LeakyReLU, GELU, SiLU
#'     \item \strong{Weight initialization:} Xavier or Kaiming initialization
#'     \item \strong{Self-Attention:} Captures relationships between features
#'     \item \strong{Progressive Training:} Gradually increase network capacity
#'   }
#'
#' @param noise_dim The length of the noise vector per example
#' @param output_info A list describing the output structure from data_transformer$output_info.
#'   Each element is a list with (dimension, type) where type is "linear", "mode_specific", or "softmax".
#' @param hidden_units A list of the number of neurons per layer. Defaults to list(256, 256)
#'   as used in CTGAN.
#' @param dropout_rate The dropout rate for each hidden layer. Only used when
#'   normalization is "none". Defaults to 0.0.
#' @param tau Temperature for Gumbel-Softmax. Lower values produce more discrete outputs.
#'   Defaults to 0.2.
#' @param normalization Type of normalization to use: "batch" (default, as in CTGAN),
#'   "layer", or "none". Batch normalization is generally preferred for GANs.
#' @param activation Activation function: "relu" (default, as in CTGAN), "leaky_relu",
#'   "gelu", or "silu". GELU and SiLU are modern alternatives that can improve performance.
#' @param init_method Weight initialization method: "xavier_uniform" (default),
#'   "xavier_normal", "kaiming_uniform", or "kaiming_normal". Xavier is generally
#'   preferred for networks with tanh/sigmoid outputs.
#' @param residual Enable residual connections between layers of the same width.
#'   Defaults to TRUE.
#' @param attention Enable self-attention layers after residual blocks. Can be TRUE (add
#'   attention after each block), FALSE (no attention), or a vector of layer indices
#'   where attention should be added (e.g., c(2, 4) adds attention after blocks 2 and 4).
#'   Defaults to FALSE.
#' @param attention_heads Number of attention heads. Must divide hidden layer size evenly.
#'   Defaults to 4.
#' @param attention_dropout Dropout rate for attention weights. Defaults to 0.1.
#'
#' @return A torch::nn_module for the Tabular Generator
#' @export
#'
#' @examples
#' \dontrun{
#' # Basic usage with CTGAN-style defaults
#' output_info <- list(list(1, "linear"), list(3, "softmax"))
#' gen <- TabularGenerator(noise_dim = 128, output_info = output_info)
#'
#' # With self-attention for capturing feature relationships
#' gen <- TabularGenerator(
#'   noise_dim = 128,
#'   output_info = output_info,
#'   hidden_units = list(256, 256, 256),
#'   attention = TRUE,
#'   attention_heads = 8
#' )
#'
#' # Custom architecture with layer normalization and GELU
#' gen <- TabularGenerator(
#'   noise_dim = 128,
#'   output_info = output_info,
#'   hidden_units = list(256, 256, 256),
#'   normalization = "layer",
#'   activation = "gelu",
#'   init_method = "kaiming_uniform"
#' )
#' }
TabularGenerator <- torch::nn_module(
  initialize = function(noise_dim,
                        output_info,
                        hidden_units = list(256, 256),
                        dropout_rate = 0.0,
                        tau = 0.2,
                        normalization = "batch",
                        activation = "relu",
                        init_method = "xavier_uniform",
                        residual = TRUE,
                        attention = FALSE,
                        attention_heads = 4,
                        attention_dropout = 0.1) {

    # Store parameters for forward pass
    self$output_info <- output_info
    self$tau <- tau
    self$training_mode <- TRUE
    self$normalization <- normalization
    self$activation <- activation
    self$residual <- residual
    self$use_attention <- !isFALSE(attention)

    # For progressive training: track which blocks are active
    self$num_blocks <- length(hidden_units)
    self$active_blocks <- self$num_blocks  # All blocks active by default

    # Determine which layers get attention
    if (isTRUE(attention)) {
      self$attention_layers <- seq_along(hidden_units)
    } else if (is.numeric(attention)) {
      self$attention_layers <- attention
    } else {
      self$attention_layers <- integer(0)
    }

    # Validate parameters
    valid_norms <- c("batch", "layer", "none")
    if (!normalization %in% valid_norms) {
      stop(sprintf("normalization must be one of: %s", paste(valid_norms, collapse = ", ")))
    }
    valid_acts <- c("relu", "leaky_relu", "gelu", "silu")
    if (!activation %in% valid_acts) {
      stop(sprintf("activation must be one of: %s", paste(valid_acts, collapse = ", ")))
    }
    valid_inits <- c("xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal")
    if (!init_method %in% valid_inits) {
      stop(sprintf("init_method must be one of: %s", paste(valid_inits, collapse = ", ")))
    }

    # Calculate total output dimension
    data_dim <- sum(sapply(output_info, function(x) x[[1]]))
    self$data_dim <- data_dim

    # Build the network with residual blocks
    self$blocks <- torch::nn_module_list()
    self$attention_blocks <- torch::nn_module_list()
    self$attention_block_indices <- list()  # Maps block index to attention block index

    dim <- noise_dim
    attn_idx <- 1
    for (i in seq_along(hidden_units)) {
      neurons <- hidden_units[[i]]

      # Create residual block
      block <- ResidualBlock(
        input_dim = dim,
        output_dim = neurons,
        normalization = normalization,
        activation = activation,
        dropout_rate = if (normalization == "none") dropout_rate else 0.0
      )

      # Disable residual if requested or dimensions don't match
      if (!residual) {
        block$use_residual <- FALSE
      }

      self$blocks$append(block)

      # Add attention layer if specified for this block
      if (i %in% self$attention_layers) {
        attn <- SelfAttention(
          embed_dim = neurons,
          num_heads = attention_heads,
          dropout = attention_dropout
        )
        self$attention_blocks$append(attn)
        self$attention_block_indices[[as.character(i)]] <- attn_idx
        attn_idx <- attn_idx + 1
      }

      dim <- neurons
    }

    # Store final hidden dimension for progressive training
    self$final_hidden_dim <- dim

    # For progressive training, we need output layers for each possible stage
    # Each output layer maps from that block's output dimension to data_dim
    self$output_layers <- torch::nn_module_list()
    self$block_output_dims <- c()

    block_dim <- noise_dim
    for (i in seq_along(hidden_units)) {
      block_dim <- hidden_units[[i]]
      self$block_output_dims <- c(self$block_output_dims, block_dim)
      out_layer <- torch::nn_linear(block_dim, data_dim)
      self$output_layers$append(out_layer)
    }

    # Legacy: keep output_layer pointing to the final one for backwards compatibility
    self$output_layer <- self$output_layers[[length(hidden_units)]]

    # Apply weight initialization
    self$apply(function(module) {
      if (inherits(module, "nn_linear")) {
        switch(
          init_method,
          "xavier_uniform" = torch::nn_init_xavier_uniform_(module$weight),
          "xavier_normal" = torch::nn_init_xavier_normal_(module$weight),
          "kaiming_uniform" = torch::nn_init_kaiming_uniform_(module$weight, a = 0.2),
          "kaiming_normal" = torch::nn_init_kaiming_normal_(module$weight, a = 0.2)
        )
        if (!is.null(module$bias)) {
          torch::nn_init_zeros_(module$bias)
        }
      }
    })
  },

  forward = function(input, hard = NULL) {
    # Pass through residual blocks (with progressive training support)
    data <- input
    num_active <- min(self$active_blocks, length(self$blocks))

    for (i in seq_len(num_active)) {
      # Apply residual block
      data <- self$blocks[[i]](data)

      # Apply attention if defined for this block
      attn_idx <- self$attention_block_indices[[as.character(i)]]
      if (!is.null(attn_idx)) {
        data <- self$attention_blocks[[attn_idx]](data)
      }
    }

    # Use the output layer corresponding to the last active block
    # This ensures correct dimensions during progressive training
    data <- self$output_layers[[num_active]](data)

    # Apply appropriate activation to each output block
    outputs <- list()
    start_idx <- 1

    for (info in self$output_info) {
      dim <- info[[1]]
      col_type <- info[[2]]
      end_idx <- start_idx + dim - 1

      # Slice the relevant columns
      col_data <- data[, start_idx:end_idx]

      if (col_type == "softmax") {
        # Apply Gumbel-Softmax for categorical columns
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

          use_hard <- if (!is.null(hard)) hard else !self$training
          mode_probs <- gumbel_softmax(mode_logits, tau = self$tau, hard = use_hard, dim = 2)
          value <- torch::torch_tanh(value)

          col_data <- torch::torch_cat(list(mode_probs, value), dim = 2)
        } else {
          col_data <- torch::torch_tanh(col_data)
        }
      }

      outputs[[length(outputs) + 1]] <- col_data
      start_idx <- end_idx + 1
    }

    # Concatenate all outputs
    return(torch::torch_cat(outputs, dim = 2))
  },

  #' Set number of active blocks for progressive training
  #' @param n Number of blocks to activate (1 to num_blocks)
  set_active_blocks = function(n) {
    if (n < 1 || n > self$num_blocks) {
      stop(sprintf("n must be between 1 and %d", self$num_blocks))
    }
    self$active_blocks <- n
    invisible(self)
  },

  #' Get current number of active blocks
  get_active_blocks = function() {
    return(self$active_blocks)
  }
)
