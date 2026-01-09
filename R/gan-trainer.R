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
           patience = 10) {
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

# Set the plotting interval ----------------------------------------------------
    plot_interval <- ifelse(plot_interval == "epoch", steps, plot_interval)

# Set up the neural networks if none are provided ------------------------------
    if (is.null(generator)) {
      g_net <-
        Generator(noise_dim = noise_dim,
                  data_dim = data_dim,
                  dropout_rate = 0.5)$to(device = device)
    } else {
      g_net <- generator
    }

    if (is.null(generator_optimizer)) {
      g_optim <- torch::optim_adam(g_net$parameters, lr = base_lr)
    } else {
      g_optim <- generator_optimizer
    }

    if (is.null(discriminator)) {
      if(value_function != "original") {
      d_net <-
        Discriminator(data_dim = data_dim, dropout_rate = 0.5)$to(device = device)
      } else {
        d_net <-
          Discriminator(data_dim = data_dim, dropout_rate = 0.5, sigmoid = TRUE)$to(device = device)
      }
    } else {
      d_net <- discriminator
    }

    if (is.null(discriminator_optimizer)) {
      d_optim <- torch::optim_adam(d_net$parameters, lr = base_lr * ttur_factor)
    } else {
      d_optim <- discriminator_optimizer
    }


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
        track_loss
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
            val_noise <- sample_noise(c(min(500, nrow(data)), noise_dim))$to(device = device)
            val_synth <- torch::with_no_grad(g_net(val_noise))

            # Compute discriminator accuracy on validation data
            if (!is.null(validation_data)) {
              val_batch <- validation_data[sample(nrow(validation_data),
                                                   size = min(batch_size, nrow(validation_data)))]$to(device = device)
              val_real_scores <- torch::with_no_grad(d_net(val_batch))
              val_fake_scores <- torch::with_no_grad(d_net(val_synth[1:min(batch_size, nrow(val_synth))]))

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
        }

    }

    output <-  list(
      generator = g_net,
      discriminator = d_net,
      generator_optimizer = g_optim,
      discriminator_optimizer = d_optim,
      losses = losses,
      validation_metrics = if (length(validation_metrics) > 0) validation_metrics else NULL,
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
                      patience = patience)
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
           track_loss = FALSE) {
    # Get a fresh batch of data ------------------------------------------------
    real_data <- get_batch(data, batch_size, device)

    # Get a fresh noise sample -------------------------------------------------
    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)
    # Produce fake data from noise ---------------------------------------------
    fake_data <- torch::with_no_grad(g_net(input = z))
    # Compute the discriminator scores on real and fake data -------------------
    dis_real <- d_net(real_data)
    dis_fake <- d_net(fake_data)
    # Calculate the discriminator loss
    d_loss <- value_function(dis_real, dis_fake)[["d_loss"]]

    # Add gradient penalty for WGAN-GP -----------------------------------------
    if (gp_lambda > 0) {
      gp <- gradient_penalty(d_net, real_data, fake_data$detach(), device)
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

    # Calculate discriminator score for fake data ------------------------------
    dis_fake <- d_net(fake_data)
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
