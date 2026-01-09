#' @title gan_trainer
#'
#' @description Provides a function to quickly train a GAN model.
#'
#' @param data Input a data set. Needs to be a matrix, array, torch::torch_tensor or torch::dataset.
#' @param noise_dim The dimensions of the GAN noise vector z. Defaults to 2.
#' @param noise_distribution The noise distribution. Expects a function that samples from a distribution and returns a torch_tensor. For convenience "normal" and "uniform" will automatically set a function. Defaults to "normal".
#' @param value_function The value function for GAN training. Expects a function that takes discriminator scores of real and fake data as input and returns a list with the discriminator loss and generator loss. For convenience four loss functions "original", "wasserstein", "wgan-gp", and "f-wgan" are already implemented. Defaults to "original".
#' @param gp_lambda The gradient penalty coefficient for WGAN-GP training. Only used when value_function is "wgan-gp". Defaults to 10.
#' @param data_type "tabular" or "image", controls the data type, defaults to "tabular".
#' @param generator The generator network. Expects a neural network provided as torch::nn_module. Default is NULL which will create a simple fully connected neural network.
#' @param generator_optimizer The optimizer for the generator network. Expects a torch::optim_xxx function, e.g. torch::optim_adam(). Default is NULL which will setup `torch::optim_adam(g_net$parameters, lr = base_lr)`.
#' @param discriminator The discriminator network. Expects a neural network provided as torch::nn_module. Default is NULL which will create a simple fully connected neural network.
#' @param discriminator_optimizer The optimizer for the generator network. Expects a torch::optim_xxx function, e.g. torch::optim_adam(). Default is NULL which will setup `torch::optim_adam(g_net$parameters, lr = base_lr * ttur_factor)`.
#' @param base_lr The base learning rate for the optimizers. Default is 0.0001. Only used if no optimizer is explicitly passed to the trainer.
#' @param ttur_factor A multiplier for the learning rate of the discriminator, to implement the two time scale update rule.
#' @param weight_clipper The wasserstein GAN puts some constraints on the weights of the discriminator, therefore weights are clipped during training.
#' @param batch_size The number of training samples selected into the mini batch for training. Defaults to 50.
#' @param epochs The number of training epochs. Defaults to 150.
#' @param plot_progress Monitor training progress with plots. Defaults to FALSE.
#' @param plot_interval Number of training steps between plots. Input number of steps or "epoch". Defaults to "epoch".
#' @param eval_dropout Should dropout be applied during the sampling of synthetic data? Defaults to FALSE.
#' @param synthetic_examples Number of synthetic examples that should be generated. Defaults to 500. For image data e.g. 16 would be more reasonable.
#' @param plot_dimensions If you monitor training progress with a plot which dimensions of the data do you want to look at? Defaults to c(1, 2), i.e. the first two columns of the tabular data.
#' @param track_loss Store the training losses as additional output. Defaults to FALSE.
#' @param plot_loss Monitor the losses during training with plots. Defaults to FALSE.
#' @param device Input on which device (e.g. "cpu", "cuda", or "mps") training should be done. Defaults to "cpu".
#' @param seed Optional seed for reproducibility. Sets both R's random seed and torch's random seed. Defaults to NULL (no seed).
#' @param validation_data Optional validation data for monitoring training. Should be in the same format as training data.
#' @param early_stopping Enable early stopping based on validation metrics. Defaults to FALSE.
#' @param patience Number of epochs without improvement before stopping. Only used if early_stopping is TRUE. Defaults to 10.
#' @param lr_schedule Learning rate schedule type. One of "constant" (default), "step", "exponential", or "cosine".
#'   "step" reduces LR by lr_decay_factor every lr_decay_steps epochs.
#'   "exponential" applies lr_decay_factor decay each epoch.
#'   "cosine" uses cosine annealing from base_lr to 0 over all epochs.
#' @param lr_decay_factor Multiplicative factor for learning rate decay. Used with "step" and "exponential" schedules. Defaults to 0.1.
#' @param lr_decay_steps Number of epochs between learning rate reductions for "step" schedule. Defaults to 50.
#' @param pac Number of samples to pack together for PacGAN (reduces mode collapse). The discriminator
#'   sees `pac` samples concatenated together, helping it detect lack of diversity. Must divide batch_size
#'   evenly. Defaults to 1 (standard GAN, no packing). Common values are 8 or 10.
#' @param output_info Optional output structure from data_transformer$output_info. When provided,
#'   enables Gumbel-Softmax for categorical columns, improving gradient flow for discrete variables.
#'   Each element should be a list with (dimension, type) where type is "linear", "mode_specific", or "softmax".
#' @param gumbel_tau Temperature for Gumbel-Softmax. Lower values (e.g., 0.2) produce more discrete
#'   outputs. Only used when output_info is provided. Defaults to 0.2.
#' @param generator_hidden_units List of hidden layer sizes for TabularGenerator. Defaults to
#'   list(256, 256) as used in CTGAN. Only used when output_info is provided.
#' @param generator_normalization Normalization type for TabularGenerator: "batch" (default, CTGAN-style),
#'   "layer", or "none". Only used when output_info is provided.
#' @param generator_activation Activation function for TabularGenerator: "relu" (default), "leaky_relu",
#'   "gelu", or "silu". Only used when output_info is provided.
#' @param generator_init Weight initialization for TabularGenerator: "xavier_uniform" (default),
#'   "xavier_normal", "kaiming_uniform", or "kaiming_normal". Only used when output_info is provided.
#' @param generator_residual Enable residual connections in TabularGenerator. Defaults to TRUE.
#'   Only used when output_info is provided.
#' @param checkpoint_epochs Interval for saving model checkpoints (in epochs). If NULL (default),
#'   no checkpoints are saved. For example, checkpoint_epochs = 10 saves checkpoints at epochs
#'   10, 20, 30, etc. Checkpoints enable post-GAN boosting for improved sample quality.
#' @param checkpoint_path Optional path for disk-based checkpoint persistence. If NULL (default),
#'   checkpoints are stored in memory only. If provided, checkpoints are saved to disk, enabling
#'   post-GAN boosting for large training runs with many checkpoints.
#'
#' @return gan_trainer trains the neural networks and returns an object of class trained_RGAN that contains the last generator, discriminator and the respective optimizers, as well as the settings.
#' @export
#'
#' @examples
#' \dontrun{
#' # Before running the first time the torch backend needs to be installed
#' torch::install_torch()
#' # Load data
#' data <- sample_toydata()
#' # Build new transformer
#' transformer <- data_transformer$new()
#' # Fit transformer to data
#' transformer$fit(data)
#' # Transform data and store as new object
#' transformed_data <-  transformer$transform(data)
#' # Train the default GAN
#' trained_gan <- gan_trainer(transformed_data)
#' # Sample synthetic data from the trained GAN
#' synthetic_data <- sample_synthetic_data(trained_gan, transformer)
#' # Plot the results
#' GAN_update_plot(data = data,
#' synth_data = synthetic_data,
#' main = "Real and Synthetic Data after Training")
#' }
gan_trainer <-
  function(data,
           noise_dim = 2,
           noise_distribution = "normal",
           value_function = "original",
           gp_lambda = 10,
           data_type = "tabular",
           generator = NULL,
           generator_optimizer = NULL,
           discriminator = NULL,
           discriminator_optimizer = NULL,
           base_lr = 0.0001,
           ttur_factor = 4,
           weight_clipper = NULL,
           batch_size = 50,
           epochs = 150,
           plot_progress = FALSE,
           plot_interval = "epoch",
           eval_dropout = FALSE,
           synthetic_examples = 500,
           plot_dimensions = c(1, 2),
           track_loss = FALSE,
           plot_loss = FALSE,
           device = "cpu",
           seed = NULL,
           validation_data = NULL,
           early_stopping = FALSE,
           patience = 10,
           lr_schedule = "constant",
           lr_decay_factor = 0.1,
           lr_decay_steps = 50,
           pac = 1,
           output_info = NULL,
           gumbel_tau = 0.2,
           generator_hidden_units = list(256, 256),
           generator_normalization = "batch",
           generator_activation = "relu",
           generator_init = "xavier_uniform",
           generator_residual = TRUE,
           checkpoint_epochs = NULL,
           checkpoint_path = NULL) {
# Set random seeds for reproducibility -----------------------------------------
    if (!is.null(seed)) {
      set.seed(seed)
      torch::torch_manual_seed(seed)
    }

# Input validation -------------------------------------------------------------
    if (batch_size <= 0) {
      stop("batch_size must be a positive integer")
    }
    if (epochs <= 0) {
      stop("epochs must be a positive integer")
    }
    if (base_lr <= 0) {
      stop("base_lr must be a positive number")
    }
    if (noise_dim <= 0) {
      stop("noise_dim must be a positive integer")
    }

    # Validate learning rate schedule parameters
    valid_schedules <- c("constant", "step", "exponential", "cosine")
    if (!lr_schedule %in% valid_schedules) {
      stop(sprintf(
        "lr_schedule must be one of: %s",
        paste(valid_schedules, collapse = ", ")
      ))
    }
    if (lr_decay_factor <= 0 || lr_decay_factor > 1) {
      stop("lr_decay_factor must be between 0 (exclusive) and 1 (inclusive)")
    }
    if (lr_decay_steps <= 0) {
      stop("lr_decay_steps must be a positive integer")
    }
    if (pac < 1 || pac != as.integer(pac)) {
      stop("pac must be a positive integer")
    }
    if (batch_size %% pac != 0) {
      stop(sprintf("batch_size (%d) must be divisible by pac (%d)", batch_size, pac))
    }
    if (gumbel_tau <= 0) {
      stop("gumbel_tau must be a positive number")
    }
    # Validate output_info structure if provided
    if (!is.null(output_info)) {
      if (!is.list(output_info)) {
        stop("output_info must be a list")
      }
      for (i in seq_along(output_info)) {
        info <- output_info[[i]]
        if (!is.list(info) || length(info) < 2) {
          stop("Each element of output_info must be a list with at least 2 elements (dimension, type)")
        }
        valid_types <- c("linear", "softmax", "mode_specific")
        if (!info[[2]] %in% valid_types) {
          stop(sprintf("output_info type must be one of: %s", paste(valid_types, collapse = ", ")))
        }
      }
    }

    # Validate checkpoint parameters
    if (!is.null(checkpoint_epochs)) {
      if (!is.numeric(checkpoint_epochs) || checkpoint_epochs <= 0 ||
          checkpoint_epochs != as.integer(checkpoint_epochs)) {
        stop("checkpoint_epochs must be a positive integer")
      }
    }
    if (!is.null(checkpoint_path) && is.null(checkpoint_epochs)) {
      warning("checkpoint_path provided but checkpoint_epochs is NULL. No checkpoints will be saved.")
    }

    # Validate device availability
    if (device == "cuda" && !torch::cuda_is_available()) {
      warning("CUDA device requested but not available. Falling back to CPU.")
      device <- "cpu"
    }
    if (device == "mps" && !torch::backends_mps_is_available()) {
      warning("MPS device requested but not available. Falling back to CPU.")
      device <- "cpu"
    }

# Check if data is in the correct format ---------------------------------------
    !(any(
      c("dataset", "matrix", "array", "torch_tensor") %in% class(data)
    ))

    if (!(any(
      c("dataset", "matrix", "array", "torch_tensor") %in% class(data)
    ))) {
      stop(
        "Data needs to be in correct format. \ntorch::dataset, matrix, array or torch::torch_tensor are permitted."
      )
    }

# Calculate the number of steps per epoch --------------------------------------
    if ((any(c("array", "matrix") %in% class(data)))) {
      # Check for empty data
      if (nrow(data) == 0) {
        stop("data cannot be empty")
      }
      if (ncol(data) == 0) {
        stop("data must have at least one column")
      }
      # Check for all-NA data
      if (all(is.na(data))) {
        stop("data cannot be all NA values")
      }

      data <- torch::torch_tensor(data)$to(device = "cpu")
      data_dim <- ncol(data)
      steps <- nrow(data) %/% batch_size

      # Warn if batch_size is larger than data
      if (batch_size > nrow(data)) {
        warning(sprintf(
          "batch_size (%d) is larger than number of observations (%d). Using full data as single batch.",
          batch_size, nrow(data)
        ))
        steps <- 1
      }

      # Validate plot_dimensions
      if (plot_progress) {
        if (length(plot_dimensions) != 2) {
          stop("plot_dimensions must be a vector of length 2")
        }
        if (any(plot_dimensions < 1) || any(plot_dimensions > data_dim)) {
          stop(sprintf(
            "plot_dimensions must be between 1 and %d (number of columns in data)",
            data_dim
          ))
        }
      }
    }

    if("image_folder" %in% class(data)) {
      steps <- length(data$imgs[[1]]) %/% batch_size
    }

# Prepare validation data if provided ------------------------------------------
    if (!is.null(validation_data)) {
      if ((any(c("array", "matrix") %in% class(validation_data)))) {
        validation_data <- torch::torch_tensor(validation_data)$to(device = "cpu")
      }
    }

# Initialize early stopping variables ------------------------------------------
    best_val_metric <- Inf
    epochs_without_improvement <- 0
    best_generator_state <- NULL
    best_discriminator_state <- NULL
    validation_metrics <- list()

# Initialize checkpoint storage ------------------------------------------------
    checkpoints <- NULL
    if (!is.null(checkpoint_epochs)) {
      checkpoints <- list(
        epochs = integer(0),
        discriminators = list(),
        generators = list(),
        on_disk = !is.null(checkpoint_path)
      )
      # Create checkpoint directory if needed
      if (!is.null(checkpoint_path) && !dir.exists(checkpoint_path)) {
        dir.create(checkpoint_path, recursive = TRUE)
      }
    }

# Set the plotting interval ----------------------------------------------------
    plot_interval <- ifelse(plot_interval == "epoch", steps, plot_interval)

# Set up the neural networks if none are provided ------------------------------
    if (is.null(generator)) {
      if (!is.null(output_info)) {
        # Use TabularGenerator with Gumbel-Softmax when output_info is provided
        g_net <-
          TabularGenerator(noise_dim = noise_dim,
                           output_info = output_info,
                           hidden_units = generator_hidden_units,
                           dropout_rate = 0.5,
                           tau = gumbel_tau,
                           normalization = generator_normalization,
                           activation = generator_activation,
                           init_method = generator_init,
                           residual = generator_residual)$to(device = device)
      } else {
        g_net <-
          Generator(noise_dim = noise_dim,
                    data_dim = data_dim,
                    dropout_rate = 0.5)$to(device = device)
      }
    } else {
      g_net <- generator
    }

    if (is.null(generator_optimizer)) {
      g_optim <- torch::optim_adam(g_net$parameters, lr = base_lr)
    } else {
      g_optim <- generator_optimizer
    }

    if (is.null(discriminator)) {
      # For PacGAN, discriminator input is data_dim * pac (packed samples)
      d_input_dim <- data_dim * pac
      if(value_function != "original") {
      d_net <-
        Discriminator(data_dim = d_input_dim, dropout_rate = 0.5)$to(device = device)
      } else {
        d_net <-
          Discriminator(data_dim = d_input_dim, dropout_rate = 0.5, sigmoid = TRUE)$to(device = device)
      }
    } else {
      d_net <- discriminator
    }

    if (is.null(discriminator_optimizer)) {
      d_optim <- torch::optim_adam(d_net$parameters, lr = base_lr * ttur_factor)
    } else {
      d_optim <- discriminator_optimizer
    }

    # Store initial learning rates for scheduling
    g_initial_lr <- g_optim$param_groups[[1]]$lr
    d_initial_lr <- d_optim$param_groups[[1]]$lr

# Define the noise distribution for the generator ------------------------------
    if (inherits(noise_distribution, "function")) {
      sample_noise <- noise_distribution
    } else {
      if (noise_distribution == "normal") {
        sample_noise <- torch::torch_randn
      }

      if (noise_distribution == "uniform") {
        sample_noise <- torch_rand_ab
      }
    }

# Define the value function ----------------------------------------------------
    if (inherits(value_function, "function")) {
      value_fct <- value_function
    } else {
      if (is.null(value_function) | value_function == "original") {
        value_fct <- GAN_value_fct

        weight_clipper <- function(d_net) {

        }
      }

      if (value_function == "wasserstein") {
        value_fct <- WGAN_value_fct

        weight_clipper <- WGAN_weight_clipper

      }
      if (value_function == "wgan-gp") {
        value_fct <- WGAN_value_fct

        # No weight clipping needed for WGAN-GP
        weight_clipper <- function(d_net) {

        }

      }
      if (value_function == "f-wgan") {
        value_fct <- KLWGAN_value_fct

        weight_clipper <- function(d_net) {

        }

      }


    }





# Sample a fixed noise vector to observe training progress ---------------------
    fixed_z <-
      sample_noise(c(synthetic_examples, noise_dim))$to(device = device)

# Initialize progress bar ------------------------------------------------------
    cli::cli_progress_bar("Training the GAN", total = epochs * steps)


    losses <- NULL


# Start GAN training loop ------------------------------------------------------
    for (i in 1:(epochs * steps)) {

      step_loss <- gan_update_step(
        data,
        batch_size,
        noise_dim,
        sample_noise,
        device,
        g_net,
        d_net,
        g_optim,
        d_optim,
        value_fct,
        weight_clipper,
        gp_lambda = if (value_function == "wgan-gp") gp_lambda else 0,
        track_loss,
        pac
      )

      if(track_loss & length(losses) == 0){
        losses <- step_loss
      }
      if(track_loss & length(losses) > 0){
        losses$g_loss <- c(losses$g_loss, step_loss$g_loss)
        losses$d_loss <- c(losses$d_loss, step_loss$d_loss)
      }

      cli::cli_progress_update()

        if (plot_progress & i %% plot_interval == 0) {
# Create synthetic data for our plot.
          synth_data <-
            expert_sample_synthetic_data(g_net, fixed_z, device, eval_dropout = eval_dropout)

          if (data_type == "tabular") {
            GAN_update_plot(
              data = data,
              dimensions = plot_dimensions,
              synth_data = synth_data,
              epoch = i %/% steps
            )
          }

          if (data_type == "image") {
            GAN_update_plot_image(synth_data = synth_data)
          }
        }

# Validation and early stopping at the end of each epoch -----------------------
        if (i %% steps == 0) {
          current_epoch <- i %/% steps

          if (!is.null(validation_data) || early_stopping) {
            # Compute validation metrics
            g_net$eval()
            d_net$eval()

            # Generate synthetic samples for validation
            # Ensure sample count is divisible by pac for PacGAN
            val_sample_count <- min(500, nrow(data))
            val_sample_count <- val_sample_count - (val_sample_count %% pac)
            if (val_sample_count < pac) val_sample_count <- pac

            val_noise <- sample_noise(c(val_sample_count, noise_dim))$to(device = device)
            val_synth <- torch::with_no_grad(g_net(val_noise))

            # Compute discriminator accuracy on validation data
            if (!is.null(validation_data)) {
              # Ensure batch size is divisible by pac for PacGAN
              val_size <- min(batch_size, nrow(validation_data))
              val_size <- val_size - (val_size %% pac)  # Make divisible by pac
              if (val_size < pac) val_size <- pac

              val_batch <- validation_data[sample(nrow(validation_data),
                                                   size = val_size)]$to(device = device)

              # Pack samples for PacGAN if needed
              if (pac > 1) {
                val_batch_packed <- pack_samples(val_batch, pac)
                val_synth_size <- min(val_size, nrow(val_synth))
                val_synth_size <- val_synth_size - (val_synth_size %% pac)
                if (val_synth_size < pac) val_synth_size <- pac
                val_synth_packed <- pack_samples(val_synth[1:val_synth_size], pac)
              } else {
                val_batch_packed <- val_batch
                val_synth_packed <- val_synth[1:min(batch_size, nrow(val_synth))]
              }

              val_real_scores <- torch::with_no_grad(d_net(val_batch_packed))
              val_fake_scores <- torch::with_no_grad(d_net(val_synth_packed))

              # Discriminator accuracy: real should be high, fake should be low
              if (value_function == "original") {
                d_acc_real <- (val_real_scores > 0.5)$float()$mean()$item()
                d_acc_fake <- (val_fake_scores < 0.5)$float()$mean()$item()
              } else {
                # For WGAN variants, use sign of scores
                d_acc_real <- (val_real_scores > 0)$float()$mean()$item()
                d_acc_fake <- (val_fake_scores < 0)$float()$mean()$item()
              }
              d_accuracy <- (d_acc_real + d_acc_fake) / 2

              # Compute generator diversity (average pairwise distance)
              val_synth_array <- torch::as_array(val_synth$cpu())
              if (nrow(val_synth_array) > 1) {
                sample_idx <- sample(nrow(val_synth_array), min(100, nrow(val_synth_array)))
                diversity <- mean(stats::dist(val_synth_array[sample_idx, , drop = FALSE]))
              } else {
                diversity <- 0
              }

              # Store validation metrics
              validation_metrics[[length(validation_metrics) + 1]] <- list(
                epoch = current_epoch,
                d_accuracy = d_accuracy,
                diversity = diversity
              )

              # Validation metric: we want balanced discriminator (accuracy ~0.5)
              # and high diversity. Lower is better.
              val_metric <- abs(d_accuracy - 0.5) - 0.01 * diversity
            } else {
              # Without validation data, use training loss variance as proxy
              if (track_loss && length(losses$d_loss) >= steps) {
                recent_d_loss <- tail(losses$d_loss, steps)
                val_metric <- stats::sd(recent_d_loss)
              } else {
                val_metric <- Inf
              }
            }

            # Early stopping check
            if (early_stopping) {
              if (val_metric < best_val_metric) {
                best_val_metric <- val_metric
                epochs_without_improvement <- 0
                # Save best model state
                best_generator_state <- g_net$state_dict()
                best_discriminator_state <- d_net$state_dict()
              } else {
                epochs_without_improvement <- epochs_without_improvement + 1
              }

              if (epochs_without_improvement >= patience) {
                cli::cli_alert_info(sprintf(
                  "Early stopping at epoch %d (no improvement for %d epochs)",
                  current_epoch, patience
                ))
                # Restore best model
                if (!is.null(best_generator_state)) {
                  g_net$load_state_dict(best_generator_state)
                  d_net$load_state_dict(best_discriminator_state)
                }
                break
              }
            }

            g_net$train()
            d_net$train()
          }

          # Adjust learning rates at epoch boundary
          if (lr_schedule != "constant") {
            adjust_learning_rate(
              g_optim, g_initial_lr, current_epoch, epochs,
              lr_schedule, lr_decay_factor, lr_decay_steps
            )
            adjust_learning_rate(
              d_optim, d_initial_lr, current_epoch, epochs,
              lr_schedule, lr_decay_factor, lr_decay_steps
            )
          }

          # Save checkpoints at specified intervals
          if (!is.null(checkpoint_epochs) && current_epoch %% checkpoint_epochs == 0) {
            if (!is.null(checkpoint_path)) {
              # Disk-based storage
              d_path <- file.path(checkpoint_path, sprintf("discriminator_epoch_%04d.pt", current_epoch))
              g_path <- file.path(checkpoint_path, sprintf("generator_epoch_%04d.pt", current_epoch))
              torch::torch_save(d_net$state_dict(), d_path)
              torch::torch_save(g_net$state_dict(), g_path)
              checkpoints$epochs <- c(checkpoints$epochs, current_epoch)
              checkpoints$discriminators[[length(checkpoints$discriminators) + 1]] <- d_path
              checkpoints$generators[[length(checkpoints$generators) + 1]] <- g_path
            } else {
              # In-memory storage (clone to avoid reference issues)
              checkpoints$epochs <- c(checkpoints$epochs, current_epoch)
              d_state <- lapply(d_net$state_dict(), function(x) x$clone())
              g_state <- lapply(g_net$state_dict(), function(x) x$clone())
              checkpoints$discriminators[[length(checkpoints$discriminators) + 1]] <- d_state
              checkpoints$generators[[length(checkpoints$generators) + 1]] <- g_state
            }
          }
        }

    }

    output <-  list(
      generator = g_net,
      discriminator = d_net,
      generator_optimizer = g_optim,
      discriminator_optimizer = d_optim,
      losses = losses,
      validation_metrics = if (length(validation_metrics) > 0) validation_metrics else NULL,
      checkpoints = checkpoints,
      settings = list(noise_dim = noise_dim,
                      noise_distribution = noise_distribution,
                      sample_noise = sample_noise,
                      value_function = value_function,
                      gp_lambda = gp_lambda,
                      data_type = data_type,
                      base_lr = base_lr,
                      ttur_factor = ttur_factor,
                      weight_clipper = weight_clipper,
                      batch_size = batch_size,
                      epochs = epochs,
                      plot_progress = plot_progress,
                      plot_interval = plot_interval,
                      eval_dropout = eval_dropout,
                      synthetic_examples = synthetic_examples,
                      plot_dimensions = plot_dimensions,
                      device = device,
                      early_stopping = early_stopping,
                      patience = patience,
                      lr_schedule = lr_schedule,
                      lr_decay_factor = lr_decay_factor,
                      lr_decay_steps = lr_decay_steps,
                      pac = pac,
                      output_info = output_info,
                      gumbel_tau = gumbel_tau,
                      generator_hidden_units = generator_hidden_units,
                      generator_normalization = generator_normalization,
                      generator_activation = generator_activation,
                      generator_init = generator_init,
                      generator_residual = generator_residual,
                      checkpoint_epochs = checkpoint_epochs,
                      checkpoint_path = checkpoint_path)
    )
    class(output) <- "trained_RGAN"
    return(
     output
    )

  }


#' @title gan_update_step
#'
#' @description Provides a function to perform a single GAN training update step,
#'   including discriminator and generator updates.
#'
#' @param data Input a data set. Needs to be a matrix, array, torch::torch_tensor or torch::dataset.
#' @param batch_size The number of training samples selected into the mini batch for training. Defaults to 50.
#' @param noise_dim The dimensions of the GAN noise vector z. Defaults to 2.
#' @param sample_noise A function to sample noise to a torch::tensor
#' @param device Input on which device (e.g. "cpu" or "cuda") training should be done. Defaults to "cpu".
#' @param g_net The generator network. Expects a neural network provided as torch::nn_module. Default is NULL which will create a simple fully connected neural network.
#' @param g_optim The optimizer for the generator network. Expects a torch::optim_xxx function, e.g. torch::optim_adam(). Default is NULL which will setup `torch::optim_adam(g_net$parameters, lr = base_lr)`.
#' @param d_net The discriminator network. Expects a neural network provided as torch::nn_module. Default is NULL which will create a simple fully connected neural network.
#' @param d_optim The optimizer for the generator network. Expects a torch::optim_xxx function, e.g. torch::optim_adam(). Default is NULL which will setup `torch::optim_adam(g_net$parameters, lr = base_lr * ttur_factor)`.
#' @param value_function The value function for GAN training. Expects a function that takes discriminator scores of real and fake data as input and returns a list with the discriminator loss and generator loss. For convenience four loss functions "original", "wasserstein", "wgan-gp", and "f-wgan" are already implemented. Defaults to "original".
#' @param weight_clipper The wasserstein GAN puts some constraints on the weights of the discriminator, therefore weights are clipped during training.
#' @param gp_lambda The gradient penalty coefficient for WGAN-GP. Set to 0 to disable. Defaults to 0.
#' @param track_loss Store the training losses as additional output. Defaults to FALSE.
#' @param pac Number of samples to pack together for PacGAN. Defaults to 1 (no packing).
#' @return A list with generator and discriminator losses if track_loss is TRUE, otherwise NULL
#' @export
gan_update_step <-
  function(data,
           batch_size,
           noise_dim,
           sample_noise,
           device = "cpu",
           g_net,
           d_net,
           g_optim,
           d_optim,
           value_function,
           weight_clipper,
           gp_lambda = 0,
           track_loss = FALSE,
           pac = 1) {
    # Get a fresh batch of data ------------------------------------------------
    real_data <- get_batch(data, batch_size, device)

    # Get a fresh noise sample -------------------------------------------------
    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)
    # Produce fake data from noise ---------------------------------------------
    fake_data <- torch::with_no_grad(g_net(input = z))

    # Pack samples for PacGAN --------------------------------------------------
    if (pac > 1) {
      real_data_packed <- pack_samples(real_data, pac)
      fake_data_packed <- pack_samples(fake_data, pac)
    } else {
      real_data_packed <- real_data
      fake_data_packed <- fake_data
    }

    # Compute the discriminator scores on real and fake data -------------------
    dis_real <- d_net(real_data_packed)
    dis_fake <- d_net(fake_data_packed)
    # Calculate the discriminator loss
    d_loss <- value_function(dis_real, dis_fake)[["d_loss"]]

    # Add gradient penalty for WGAN-GP -----------------------------------------
    # Note: GP is computed on unpacked samples for proper gradient computation
    if (gp_lambda > 0) {
      gp <- gradient_penalty(d_net, real_data_packed, fake_data_packed$detach(), device)
      d_loss <- d_loss + gp_lambda * gp
    }

    # Clip weights according to weight_clipper ---------------------------------
    weight_clipper(d_net)
    # What follows is one update step for the discriminator net-----------------
    # First set all previous gradients to zero
    d_optim$zero_grad()

    # Pass the loss backward through the net
    d_loss$backward()

    # Take one step of the optimizer
    d_optim$step()

    # Update the generator -----------------------------------------------------
    # Get a fresh noise sample -------------------------------------------------
    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)
    # Produce fake data --------------------------------------------------------
    fake_data <- g_net(z)

    # Pack samples for PacGAN --------------------------------------------------
    if (pac > 1) {
      fake_data_packed <- pack_samples(fake_data, pac)
    } else {
      fake_data_packed <- fake_data
    }

    # Calculate discriminator score for fake data ------------------------------
    dis_fake <- d_net(fake_data_packed)
    # Get generator loss based on scores ---------------------------------------
    g_loss <- value_function(dis_real, dis_fake)[["g_loss"]]
    # What follows is one update step for the generator net --------------------
    # First set all previous gradients to zero
    g_optim$zero_grad()

    # Pass the loss backward through the net
    g_loss$backward()

    # Take one step of the optimizer
    g_optim$step()

    if(track_loss){

      return(list(
        g_loss = torch::as_array(g_loss$detach()$cpu()),
        d_loss = torch::as_array(d_loss$detach()$cpu())
      ))


    }
  }


#' @title GAN_update_plot
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param data Real data to be plotted
#' @param dimensions Which columns of the data should be plotted
#' @param synth_data The synthetic data to be plotted
#' @param epoch The epoch during training for the plot title
#' @param main An optional plot title
#'
#' @return A function
#' @export
#' @examples
#' \dontrun{
#' # Before running the first time the torch backend needs to be installed
#' torch::install_torch()
#' # Load data
#' data <- sample_toydata()
#' # Build new transformer
#' transformer <- data_transformer$new()
#' # Fit transformer to data
#' transformer$fit(data)
#' # Transform data and store as new object
#' transformed_data <-  transformer$transform(data)
#' # Train the default GAN
#' trained_gan <- gan_trainer(transformed_data)
#' # Sample synthetic data from the trained GAN
#' synthetic_data <- sample_synthetic_data(trained_gan, transformer)
#' # Plot the results
#' GAN_update_plot(data = data,
#' synth_data = synthetic_data,
#' main = "Real and Synthetic Data after Training")
#' }
GAN_update_plot <-
  function(data,
           dimensions = c(1, 2),
           synth_data,
           epoch,
           main = NULL) {

    if("torch_tensor" %in% class(data)) {
      data <- torch::as_array(data$cpu())
    }
    # Now we plot the training data.
    plot(
      data[, dimensions],
      bty = "n",
      col = viridis::viridis(2, alpha = 0.7)[1],
      #xlim = c(-50, 50),
      pch = 19,
      xlab = ifelse(
        !is.null(colnames(data)),
        colnames(data)[dimensions[1]],
        paste0("Var ", dimensions[1])
      ),
      ylab = ifelse(
        !is.null(colnames(data)),
        colnames(data)[dimensions[2]],
        paste0("Var ", dimensions[2])
      ),
      main = ifelse(is.null(main), paste0("Epoch: ", epoch), main),
      las = 1
    )
    # And we add the synthetic data on top.
    graphics::points(
      synth_data[, dimensions],
      bty = "n",
      col = viridis::viridis(2, alpha = 0.7)[2],
      pch = 19
    )
    # Finally a legend to understand the plot.
    graphics::legend(
      "topleft",
      bty = "n",
      pch = 19,
      col = viridis::viridis(2),
      legend = c("Real", "Synthetic")
    )
  }


#' @title GAN_update_plot_image
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param mfrow The dimensions of the grid of images to be plotted
#' @param synth_data The synthetic data (images) to be plotted
#'
#' @return A function
#' @export
GAN_update_plot_image <-
  function(mfrow = c(4, 4),
           synth_data) {

    synth_data <- (synth_data + 1) / 2
    synth_data <- aperm(synth_data, c(1,3,4,2))
    oldpar <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(oldpar))
    graphics::par(mfrow = mfrow, mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
    lapply(lapply(lapply(1:(dim(synth_data)[1]), function(x) synth_data[x,,,]), grDevices::as.raster), plot)

  }

get_batch <- function(dataset, batch_size, device = "cpu") {
  if ("dataset" %in% class(dataset)) {
    dataloader <-
      torch::dataloader(dataset,
                        batch_size,
                        shuffle = TRUE,
                        num_workers = 0)
    torch::dataloader_next(torch::dataloader_make_iter(dataloader))$x$to(device = device)
  } else {
    n_rows <- nrow(dataset)
    # Use replacement if batch_size > n_rows
    use_replace <- batch_size > n_rows
    dataset[sample(n_rows, size = batch_size, replace = use_replace)]$to(device = device)
  }
}


#' @title Pack Samples for PacGAN
#'
#' @description Reshapes a batch of samples for PacGAN by concatenating `pac`
#'   consecutive samples along the feature dimension. This allows the discriminator
#'   to see multiple samples at once, helping it detect mode collapse.
#'
#' @param data A torch tensor of shape (batch_size, data_dim)
#' @param pac Number of samples to pack together. batch_size must be divisible by pac.
#'
#' @return A torch tensor of shape (batch_size/pac, data_dim*pac)
#' @keywords internal
pack_samples <- function(data, pac) {
  if (pac == 1) {
    return(data)
  }

  batch_size <- data$shape[1]
  data_dim <- data$shape[2]

  # Reshape from (batch_size, data_dim) to (batch_size/pac, pac*data_dim)
  # First reshape to (batch_size/pac, pac, data_dim)
  # Then flatten the last two dimensions
  packed <- data$view(c(batch_size %/% pac, pac * data_dim))

  return(packed)
}


#' @title Adjust Learning Rate
#'
#' @description Internal helper function to adjust optimizer learning rates
#'   according to the specified schedule.
#'
#' @param optimizer A torch optimizer object
#' @param initial_lr The initial learning rate
#' @param current_epoch The current epoch number (1-indexed)
#' @param total_epochs Total number of training epochs
#' @param lr_schedule The learning rate schedule type
#' @param lr_decay_factor Decay factor for step/exponential schedules
#' @param lr_decay_steps Epochs between decays for step schedule
#'
#' @return The new learning rate (invisibly). Modifies optimizer in place.
#' @keywords internal
adjust_learning_rate <- function(optimizer,
                                  initial_lr,
                                  current_epoch,
                                  total_epochs,
                                  lr_schedule,
                                  lr_decay_factor,
                                  lr_decay_steps) {
  new_lr <- switch(
    lr_schedule,
    "constant" = initial_lr,
    "step" = {
      # Reduce LR by decay_factor every decay_steps epochs
      num_decays <- current_epoch %/% lr_decay_steps
      initial_lr * (lr_decay_factor ^ num_decays)
    },
    "exponential" = {
      # Exponential decay each epoch
      initial_lr * (lr_decay_factor ^ (current_epoch - 1))
    },
    "cosine" = {
      # Cosine annealing from initial_lr to 0
      initial_lr * (1 + cos(pi * current_epoch / total_epochs)) / 2
    },
    initial_lr  # fallback
  )

  # Update learning rate in optimizer
  # In R torch, we need to modify param_groups in place using index access
  for (i in seq_along(optimizer$param_groups)) {
    optimizer$param_groups[[i]]$lr <- new_lr
  }

  invisible(new_lr)
}
