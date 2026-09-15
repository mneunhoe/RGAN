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
      self$seq$add_module(module = torch::nn_leaky_relu(0.2),
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


# ---------------------------------------------------------------------------
# Column-token transformer internals (Experiment 4 of the 2026 plan)
#
# These two modules are the shared building blocks of `TokenDiscriminator`
# (below) and `TokenGenerator` (in R/generator-architectures.R). They are
# deliberately written with plain torch ops rather than
# `torch::nn_multihead_attention`, for two reasons:
#
#   * the ViTGAN L2 attention variant is not expressible through the built-in
#     module, and
#   * every operation has to be free of in-place writes so that the R1/R2
#     penalties, which call `autograd_grad(..., create_graph = TRUE)` on the
#     discriminator input, can be differentiated a second time.
# ---------------------------------------------------------------------------

#' @title Multi-head attention over a token sequence
#'
#' @description Internal multi-head self-attention over a `(batch, tokens, d_model)`
#'   tensor, with two variants of the attention logits.
#'
#'   * `attention = "dot"` is the standard scaled dot-product,
#'     \eqn{q_i^\top k_j / \sqrt{d_h}}, with separate query and key projections.
#'   * `attention = "l2"` is the ViTGAN variant (Lee et al., ICLR 2022,
#'     arXiv:2107.04589): \eqn{-\|q_i - k_j\|^2 / \sqrt{d_h}} with the query and
#'     key projection **tied** to a single weight matrix. Dot-product attention
#'     is not Lipschitz, and the ViTGAN authors show that spectral norm or a
#'     gradient penalty alone do not stabilise a transformer discriminator; the
#'     tied-weight L2 form is.
#'
#'   The squared distance is computed as \eqn{\|q\|^2 + \|k\|^2 - 2 q^\top k}
#'   rather than with `torch_cdist()`. `torch_cdist()` is available in torch
#'   0.17, but it differentiates a square root whose argument is exactly zero on
#'   the diagonal whenever the query and key projections are tied, which yields
#'   non-finite second derivatives. The expansion has no such problem.
#'
#' @param d_model Token width.
#' @param n_heads Number of attention heads; must divide `d_model`.
#' @param attention Either `"l2"` (default) or `"dot"`.
#' @param dropout Dropout rate applied to the attention weights and the output.
#'
#' @return A torch::nn_module.
#' @keywords internal
TokenAttention <- torch::nn_module(
  classname = "TokenAttention",
  initialize = function(d_model, n_heads = 4, attention = c("l2", "dot"),
                        dropout = 0.0) {
    attention <- match.arg(attention)
    if (d_model %% n_heads != 0) {
      stop(sprintf("d_model (%d) must be divisible by n_heads (%d)",
                   d_model, n_heads))
    }
    self$d_model <- d_model
    self$n_heads <- n_heads
    self$head_dim <- d_model %/% n_heads
    self$attention <- attention

    # Tied query/key projection for the L2 variant: one linear layer is used
    # for both q and k, which is what makes the logits a genuine (negated)
    # squared distance and the map Lipschitz.
    self$q_proj <- torch::nn_linear(d_model, d_model, bias = FALSE)
    self$k_proj <- if (attention == "l2") NULL else
      torch::nn_linear(d_model, d_model, bias = FALSE)
    self$v_proj <- torch::nn_linear(d_model, d_model)
    self$out_proj <- torch::nn_linear(d_model, d_model)
    self$attn_dropout <- torch::nn_dropout(dropout)
    self$out_dropout <- torch::nn_dropout(dropout)
  },
  # Split (batch, tokens, d_model) into (batch, heads, tokens, head_dim).
  split_heads = function(x) {
    shp <- x$shape
    x$reshape(c(shp[1], shp[2], self$n_heads, self$head_dim))$permute(c(1, 3, 2, 4))
  },
  forward = function(x) {
    shp <- x$shape
    batch <- shp[1]
    n_tok <- shp[2]

    q <- self$split_heads(self$q_proj(x))
    k <- if (identical(self$attention, "l2")) q else
      self$split_heads(self$k_proj(x))
    v <- self$split_heads(self$v_proj(x))

    scale <- sqrt(self$head_dim)
    qk <- torch::torch_matmul(q, k$transpose(3, 4))

    if (identical(self$attention, "l2")) {
      q_sq <- torch::torch_sum(q^2, dim = -1, keepdim = TRUE)          # (b,h,t,1)
      k_sq <- torch::torch_sum(k^2, dim = -1, keepdim = TRUE)$transpose(3, 4)
      logits <- -(q_sq + k_sq - 2 * qk) / scale
    } else {
      logits <- qk / scale
    }

    weights <- torch::nnf_softmax(logits, dim = -1)
    weights <- self$attn_dropout(weights)

    out <- torch::torch_matmul(weights, v)                             # (b,h,t,dh)
    out <- out$permute(c(1, 3, 2, 4))$reshape(c(batch, n_tok, self$d_model))
    self$out_dropout(self$out_proj(out))
  }
)


#' @title Pre-LayerNorm transformer block over tokens
#'
#' @description Internal transformer block: pre-LayerNorm, multi-head
#'   [TokenAttention()], residual, pre-LayerNorm, GELU MLP of width
#'   `4 * d_model`, residual. All operations are out-of-place so that the block
#'   is twice differentiable (needed for R1/R2 penalties).
#'
#' @param d_model Token width.
#' @param n_heads Number of attention heads.
#' @param attention Either `"l2"` (default) or `"dot"`, see [TokenAttention()].
#' @param dropout Dropout rate.
#'
#' @return A torch::nn_module.
#' @keywords internal
TokenTransformerBlock <- torch::nn_module(
  classname = "TokenTransformerBlock",
  initialize = function(d_model, n_heads = 4, attention = c("l2", "dot"),
                        dropout = 0.0) {
    attention <- match.arg(attention)
    self$norm1 <- torch::nn_layer_norm(d_model)
    self$attn <- TokenAttention(d_model, n_heads, attention, dropout)
    self$norm2 <- torch::nn_layer_norm(d_model)
    self$fc1 <- torch::nn_linear(d_model, 4 * d_model)
    self$fc2 <- torch::nn_linear(4 * d_model, d_model)
    self$mlp_dropout <- torch::nn_dropout(dropout)
  },
  forward = function(x) {
    x <- x + self$attn(self$norm1(x))
    h <- self$norm2(x)
    h <- self$fc2(self$mlp_dropout(torch::nnf_gelu(self$fc1(h))))
    x + self$mlp_dropout(h)
  }
)


#' @title Column-Token Discriminator with Attention
#'
#' @description A discriminator that turns every column block of an
#'   `output_info` structure into a token, adds a learned column-ID embedding,
#'   and runs transformer blocks over the resulting token sequence, pooling to a
#'   single score through a learned `[CLS]` token. This is the FT-Transformer
#'   tokenisation (Gorishniy et al. 2021) applied on the critic side, as done by
#'   FCT-GAN and TabTransGAN, with two additions:
#'
#'   \itemize{
#'     \item **ViTGAN L2 attention** (default). Dot-product attention is not
#'       Lipschitz and destabilises transformer discriminators; the L2 variant
#'       with tied query/key weights is the ViTGAN fix (Lee et al., ICLR 2022,
#'       arXiv:2107.04589). Use `attention = "dot"` for the standard variant.
#'     \item **Optional SAINT-style row attention** (Somepalli et al. 2021,
#'       arXiv:2106.01342). After every column-attention block the token tensor
#'       is transposed so that the *batch* becomes the sequence dimension and a
#'       second attention block mixes information across the samples in the
#'       minibatch, per token. This is a learned, differentiable generalisation
#'       of PacGAN and minibatch discrimination: the critic can see that a batch
#'       lacks diversity. Cost is O(batch^2) attention per token, which is
#'       comfortable for batches up to a few hundred rows.
#'   }
#'
#'   Because there is no positional encoding beyond the learned column-ID
#'   embedding, the architecture is permutation equivariant in the columns: a
#'   consistent permutation of `output_info`, of the input blocks and of the
#'   column-ID embeddings leaves the score unchanged.
#'
#'   The module is written without in-place operations, so it is twice
#'   differentiable and works with the zero-centred R1/R2 penalties
#'   (`regularization = list(type = "r1r2", gamma = ...)` in [gan_trainer()]).
#'
#' @param output_info A list describing the input structure, exactly the
#'   `output_info` of a fitted [data_transformer()]. Each element is a list
#'   `list(dimension, type)`; only the dimensions are used here, so every type
#'   produced by the transformer (`"linear"`, `"mode_specific"`, `"softmax"`,
#'   `"quantile"`, `"cossin"`, `"index"`) is accepted.
#' @param d_model Token width. Defaults to 64.
#' @param n_heads Number of attention heads; must divide `d_model`. Defaults to 4.
#' @param n_layers Number of transformer blocks. Defaults to 2.
#' @param attention `"l2"` (default, ViTGAN, recommended) or `"dot"`.
#' @param row_attention Add a SAINT-style inter-sample attention block after each
#'   column-attention block. Defaults to `FALSE`.
#' @param pac PacGAN packing factor. Must match the `pac` given to
#'   [gan_trainer()], because the trainer hands the discriminator a
#'   `(batch / pac, pac * width)` tensor. The `pac` samples of a pack become
#'   `pac * n_cols` tokens, the column-ID embedding is shared across pack slots
#'   and a learned pack-slot embedding is added, so the critic can compare the
#'   samples inside a pack. Defaults to 1.
#' @param extra_inputs Number of extra features appended to each sample's
#'   feature vector (for example the noise level sigma of instance-noise
#'   training, or a condition vector). They are mapped to one additional token
#'   by their own linear layer. Defaults to 0.
#' @param dropout Dropout rate inside the transformer blocks. Defaults to 0.
#' @param sigmoid Switch between a sigmoid and a linear output (the sigmoid is
#'   needed for the original GAN value function). Defaults to `FALSE`.
#'
#' @details
#'   `forward()` takes the flat
#'   `(batch / pac, pac * (data_dim + extra_inputs))` tensor that [gan_trainer()]
#'   passes, where `data_dim = sum(dim_j)` over `output_info` and the `pac`
#'   samples of a pack are laid out consecutively (the layout produced by the
#'   internal `pack_samples()`). With `pac = 1` this is simply
#'   `(batch, data_dim + extra_inputs)`. The `extra_inputs` columns are the
#'   **last** `extra_inputs` columns of each sample's slice.
#'
#' @references
#'   Lee, K., Chang, H., Jiang, L., Zhang, H., Tu, Z. & Liu, C. (2022). ViTGAN:
#'   Training GANs with Vision Transformers. *ICLR 2022*. arXiv:2107.04589.
#'
#'   Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B. & Goldstein, T.
#'   (2021). SAINT: Improved Neural Networks for Tabular Data via Row Attention
#'   and Contrastive Pre-Training. arXiv:2106.01342.
#'
#'   Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. (2021). Revisiting
#'   Deep Learning Models for Tabular Data. *NeurIPS 2021*. arXiv:2106.11959.
#'
#' @return A torch::nn_module for the column-token discriminator.
#' @export
#'
#' @examples
#' \dontrun{
#' transformer <- data_transformer$new()
#' transformer$fit(data, discrete_columns = c("sex", "income"),
#'                 continuous_transform = "quantile",
#'                 categorical_encoding = "cossin")
#' transformed <- transformer$transform(data)
#'
#' d_net <- TokenDiscriminator(
#'   output_info = transformer$output_info,
#'   d_model = 64, n_heads = 4, n_layers = 2,
#'   attention = "l2", row_attention = TRUE
#' )
#'
#' trained <- gan_trainer(
#'   transformed,
#'   discriminator = d_net,
#'   output_info = transformer$output_info,
#'   value_function = "rpgan",
#'   regularization = list(type = "r1r2", gamma = 10),
#'   batch_size = 500
#' )
#'
#' # With PacGAN packing the discriminator must know the packing factor:
#' d_net <- TokenDiscriminator(transformer$output_info, pac = 10)
#' trained <- gan_trainer(transformed, discriminator = d_net,
#'                        output_info = transformer$output_info, pac = 10)
#' }
TokenDiscriminator <- torch::nn_module(
  classname = "TokenDiscriminator",
  initialize = function(output_info,
                        d_model = 64,
                        n_heads = 4,
                        n_layers = 2,
                        attention = c("l2", "dot"),
                        row_attention = FALSE,
                        pac = 1,
                        extra_inputs = 0,
                        dropout = 0.0,
                        sigmoid = FALSE) {
    attention <- match.arg(attention)

    if (!is.list(output_info) || length(output_info) == 0) {
      stop("output_info must be a non-empty list")
    }
    dims <- vapply(seq_along(output_info), function(i) {
      info <- output_info[[i]]
      if (!is.list(info) || length(info) < 2) {
        stop("Each element of output_info must be a list with at least 2 elements (dimension, type)")
      }
      d <- as.integer(info[[1]])
      if (is.na(d) || d < 1) {
        stop("Each output_info block must have a positive integer dimension")
      }
      d
    }, integer(1))

    if (pac < 1 || pac != as.integer(pac)) stop("pac must be a positive integer")
    if (extra_inputs < 0 || extra_inputs != as.integer(extra_inputs)) {
      stop("extra_inputs must be a non-negative integer")
    }
    if (n_layers < 1) stop("n_layers must be at least 1")

    pac <- as.integer(pac)
    extra_inputs <- as.integer(extra_inputs)

    self$output_info <- output_info
    self$block_dims <- dims
    self$n_cols <- length(dims)
    self$data_dim <- sum(dims)
    self$pac <- pac
    self$extra_inputs <- extra_inputs
    self$has_extra <- extra_inputs > 0
    self$sample_dim <- self$data_dim + extra_inputs
    self$d_model <- d_model
    self$use_row_attention <- isTRUE(row_attention)
    self$use_sigmoid <- isTRUE(sigmoid)

    # One token per column block, plus (optionally) one for the extra inputs.
    self$n_tokens <- self$n_cols + as.integer(self$has_extra)

    self$tokenisers <- torch::nn_module_list()
    for (d in dims) self$tokenisers$append(torch::nn_linear(d, d_model))
    if (self$has_extra) {
      self$extra_tokeniser <- torch::nn_linear(extra_inputs, d_model)
    }

    # Learned column-ID embedding (no positional encoding: the model is
    # permutation equivariant in the columns up to this embedding).
    self$col_embedding <- torch::nn_embedding(self$n_tokens, d_model)
    self$col_ids <- torch::nn_buffer(
      torch::torch_tensor(seq_len(self$n_tokens), dtype = torch::torch_long())
    )
    if (pac > 1) {
      self$pac_embedding <- torch::nn_embedding(pac, d_model)
      self$pac_ids <- torch::nn_buffer(
        torch::torch_tensor(seq_len(pac), dtype = torch::torch_long())
      )
    }

    # Learned [CLS] token.
    self$cls_token <- torch::nn_parameter(torch::torch_randn(c(1, 1, d_model)) * 0.02)

    self$blocks <- torch::nn_module_list()
    self$row_blocks <- torch::nn_module_list()
    for (i in seq_len(n_layers)) {
      self$blocks$append(TokenTransformerBlock(d_model, n_heads, attention, dropout))
      if (self$use_row_attention) {
        self$row_blocks$append(TokenTransformerBlock(d_model, n_heads, attention, dropout))
      }
    }
    self$n_layers <- as.integer(n_layers)

    self$head_norm <- torch::nn_layer_norm(d_model)
    self$head <- torch::nn_linear(d_model, 1)
  },

  forward = function(input) {
    batch <- input$shape[1]
    expected <- self$pac * self$sample_dim
    if (input$shape[2] != expected) {
      stop(sprintf(
        "TokenDiscriminator expects %d input columns (pac = %d, data_dim = %d, extra_inputs = %d) but got %d",
        expected, self$pac, self$data_dim, self$extra_inputs, input$shape[2]
      ))
    }

    # (batch, pac * sample_dim) -> (batch, pac, sample_dim)
    x <- input$reshape(c(batch, self$pac, self$sample_dim))

    tokens <- vector("list", self$n_tokens)
    start_idx <- 1
    for (j in seq_len(self$n_cols)) {
      end_idx <- start_idx + self$block_dims[j] - 1
      tokens[[j]] <- self$tokenisers[[j]](x[, , start_idx:end_idx])
      start_idx <- end_idx + 1
    }
    if (self$has_extra) {
      tokens[[self$n_tokens]] <-
        self$extra_tokeniser(x[, , start_idx:(start_idx + self$extra_inputs - 1)])
    }

    # (batch, pac, n_tokens, d_model)
    h <- torch::torch_stack(tokens, dim = 3)

    col_emb <- self$col_embedding(self$col_ids)$reshape(c(1, 1, self$n_tokens, self$d_model))
    h <- h + col_emb
    if (self$pac > 1) {
      pac_emb <- self$pac_embedding(self$pac_ids)$reshape(c(1, self$pac, 1, self$d_model))
      h <- h + pac_emb
    }

    # (batch, pac * n_tokens, d_model), then prepend the [CLS] token.
    h <- h$reshape(c(batch, self$pac * self$n_tokens, self$d_model))
    cls <- self$cls_token$expand(c(batch, 1, self$d_model))
    h <- torch::torch_cat(list(cls, h), dim = 2)

    for (i in seq_len(self$n_layers)) {
      h <- self$blocks[[i]](h)
      if (self$use_row_attention) {
        # SAINT-style inter-sample attention: make the batch the sequence
        # dimension, so every token attends across the rows of the minibatch.
        h <- self$row_blocks[[i]](h$transpose(1, 2)$contiguous())$transpose(1, 2)$contiguous()
      }
    }

    out <- self$head(self$head_norm(h[, 1, ]))
    if (self$use_sigmoid) out <- torch::torch_sigmoid(out)
    out
  }
)
