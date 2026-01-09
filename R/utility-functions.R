#' @importFrom utils tail
#' @importFrom R6 R6Class
NULL

#' @title Gumbel-Softmax Sampling
#'
#' @description Implements the Gumbel-Softmax (Concrete) distribution for differentiable
#'   sampling from categorical distributions. During training, returns soft samples that
#'   allow gradients to flow. During inference, can return hard one-hot samples.
#'
#' @param logits A torch tensor of unnormalized log probabilities
#' @param tau Temperature parameter. Lower values make the distribution more discrete.
#'   Defaults to 1.0.
#' @param hard If TRUE, returns hard one-hot samples but gradients are computed as if
#'   soft samples were used (straight-through estimator). Defaults to FALSE.
#' @param dim The dimension along which to apply softmax. Defaults to -1 (last dimension).
#'
#' @return A torch tensor of the same shape as logits, containing either soft or hard samples
#' @export
#'
#' @examples
#' \dontrun{
#' logits <- torch::torch_randn(c(10, 5))  # 10 samples, 5 categories
#' soft_samples <- gumbel_softmax(logits, tau = 0.5)
#' hard_samples <- gumbel_softmax(logits, tau = 0.5, hard = TRUE)
#' }
gumbel_softmax <- function(logits, tau = 1.0, hard = FALSE, dim = -1) {
  # Sample from Gumbel(0, 1) distribution
  # Using inverse CDF method: -log(-log(U)) where U ~ Uniform(0, 1)
  gumbels <- -torch::torch_log(-torch::torch_log(
    torch::torch_rand_like(logits)$clamp(min = 1e-10, max = 1 - 1e-10)
  ))

  # Add Gumbel noise to logits and apply softmax
  y_soft <- torch::nnf_softmax((logits + gumbels) / tau, dim = dim)

  if (hard) {
    # Get hard one-hot encoding
    index <- y_soft$max(dim = dim, keepdim = TRUE)[[2]]
    y_hard <- torch::torch_zeros_like(logits)$scatter_(dim, index, 1.0)
    # Straight-through estimator: use hard in forward, soft gradients in backward
    ret <- (y_hard - y_soft$detach()) + y_soft
  } else {
    ret <- y_soft
  }

  return(ret)
}


#' @title Uniform Random numbers between values a and b
#'
#' @description Provides a function to sample torch tensors from an arbitrary uniform distribution.
#'
#' @param shape Vector of dimensions of resulting tensor
#' @param a Lower bound of uniform distribution to sample from
#' @param b Upper bound of uniform distribution to sample from
#' @param ... Potential additional arguments
#'
#' @return A sample from the specified uniform distribution in a tensor with the specified shape
#' @export
torch_rand_ab <- function(shape, a = -1, b = 1, ...) {
  (a-b) * torch::torch_rand(shape, ...) + b
}

#' @title Sample Toydata
#'
#' @description Sample Toydata to reproduce the examples in the paper.
#'
#' @param n Number of observations to generate
#' @param sd Standard deviation of the normal distribution to generate y
#' @param seed A seed for the pseudo random number generator
#'
#' @return A matrix with two columns x and y
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
sample_toydata <- function(n = 1000, sd = 0.3, seed = 20211111) {
  set.seed(seed)
  x <- c(stats::rnorm(n))

  y <- c(stats::rnorm(n, x ^ 2, sd))

  cbind(x, y)
}

#' @title KL WGAN loss on real examples
#'
#' @description Utility function for the kl WGAN loss for real examples as described in [https://arxiv.org/abs/1910.09779](https://arxiv.org/abs/1910.09779)
#'   and implemented in python in [https://github.com/ermongroup/f-wgan](https://github.com/ermongroup/f-wgan).
#'
#' @param dis_real Discriminator scores on real examples ($D(x)$)
#'
#' @return The loss
kl_real <- function(dis_real) {
  loss_real <- torch::torch_mean(torch::nnf_relu(1 - dis_real))

  return(loss_real)
}

#' @title KL WGAN loss on fake examples
#'
#' @description  Utility function for the kl WGAN loss for fake examples as described in [https://arxiv.org/abs/1910.09779](https://arxiv.org/abs/1910.09779)
#'   and implemented in python in [https://github.com/ermongroup/f-wgan](https://github.com/ermongroup/f-wgan).
#'
#' @param dis_fake Discriminator scores on fake examples ($D(G(z))$)
#'
#' @return The loss
kl_fake <- function(dis_fake) {
  dis_fake_norm = torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch::torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss_fake = torch::torch_mean(torch::nnf_relu(1. + dis_fake))

  return(loss_fake)
}

#' @title KL WGAN loss for Generator training
#'
#' @description Utility function for the kl WGAN loss for Generator training as described in [https://arxiv.org/abs/1910.09779](https://arxiv.org/abs/1910.09779)
#'   and implemented in python in [https://github.com/ermongroup/f-wgan](https://github.com/ermongroup/f-wgan).
#'
#' @param dis_fake Discriminator scores on fake examples ($D(G(z))$)
#'
#' @return The loss
kl_gen <- function(dis_fake) {
  dis_fake_norm = torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch::torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss = -torch::torch_mean(dis_fake)
  return(loss)
}


#' @title Print Method for Trained RGAN Objects
#'
#' @description Displays a summary of a trained GAN model, including network
#'   architecture, training settings, and final losses.
#'
#' @param x A trained GAN object of class "trained_RGAN"
#' @param ... Additional arguments (currently unused)
#'
#' @return Invisibly returns the input object
#' @export
#'
#' @examples
#' \dontrun{
#' data <- sample_toydata()
#' transformer <- data_transformer$new()
#' transformer$fit(data)
#' transformed_data <- transformer$transform(data)
#' trained_gan <- gan_trainer(transformed_data, epochs = 10, track_loss = TRUE)
#' print(trained_gan)
#' }
print.trained_RGAN <- function(x, ...) {
  cat("Trained RGAN Model\n")
  cat("==================\n\n")

  # Training settings
  cat("Training Settings:\n")
  cat(sprintf("  Value function: %s\n", x$settings$value_function))
  cat(sprintf("  Epochs: %d\n", x$settings$epochs))
  cat(sprintf("  Batch size: %d\n", x$settings$batch_size))
  cat(sprintf("  Noise dimension: %d\n", x$settings$noise_dim))
  cat(sprintf("  Base learning rate: %s\n", format(x$settings$base_lr, scientific = FALSE)))
  cat(sprintf("  Device: %s\n", x$settings$device))

  if (x$settings$value_function == "wgan-gp") {
    cat(sprintf("  GP lambda: %s\n", x$settings$gp_lambda))
  }
  if (x$settings$early_stopping) {
    cat(sprintf("  Early stopping: enabled (patience=%d)\n", x$settings$patience))
  }
  if (!is.null(x$settings$lr_schedule) && x$settings$lr_schedule != "constant") {
    cat(sprintf("  LR schedule: %s", x$settings$lr_schedule))
    if (x$settings$lr_schedule == "step") {
      cat(sprintf(" (factor=%.2f, steps=%d)", x$settings$lr_decay_factor, x$settings$lr_decay_steps))
    } else if (x$settings$lr_schedule == "exponential") {
      cat(sprintf(" (factor=%.2f)", x$settings$lr_decay_factor))
    }
    cat("\n")
  }
  if (!is.null(x$settings$pac) && x$settings$pac > 1) {
    cat(sprintf("  PacGAN: enabled (pac=%d)\n", x$settings$pac))
  }
  if (!is.null(x$settings$output_info)) {
    cat(sprintf("  Gumbel-Softmax: enabled (tau=%.2f)\n", x$settings$gumbel_tau))
    cat(sprintf("  Generator architecture: %s normalization, %s activation\n",
                x$settings$generator_normalization, x$settings$generator_activation))
    if (x$settings$generator_residual) {
      cat("  Residual connections: enabled\n")
    }
  }
  cat("\n")

  # Network architecture - count parameters
  count_params <- function(net) {
    params <- net$parameters
    total <- 0
    for (p in params) {
      total <- total + prod(p$shape)
    }
    total
  }

  cat("Generator:\n")
  g_params <- count_params(x$generator)
  cat(sprintf("  Parameters: %s\n", format(g_params, big.mark = ",")))

  cat("\nDiscriminator:\n")
  d_params <- count_params(x$discriminator)
  cat(sprintf("  Parameters: %s\n", format(d_params, big.mark = ",")))

  cat(sprintf("\nTotal parameters: %s\n", format(g_params + d_params, big.mark = ",")))

  # Final losses
  if (!is.null(x$losses)) {
    cat("\nFinal Training Losses:\n")
    n_losses <- length(x$losses$g_loss)
    if (n_losses > 0) {
      # Average over last epoch's worth of steps or last 10, whichever is smaller
      n_avg <- min(10, n_losses)
      g_final <- mean(tail(x$losses$g_loss, n_avg))
      d_final <- mean(tail(x$losses$d_loss, n_avg))
      cat(sprintf("  Generator loss: %.4f (avg of last %d steps)\n", g_final, n_avg))
      cat(sprintf("  Discriminator loss: %.4f (avg of last %d steps)\n", d_final, n_avg))
    }
  } else {
    cat("\nLosses: Not tracked (use track_loss=TRUE)\n")
  }

  # Validation metrics
  if (!is.null(x$validation_metrics) && length(x$validation_metrics) > 0) {
    cat("\nFinal Validation Metrics:\n")
    last_metric <- x$validation_metrics[[length(x$validation_metrics)]]
    cat(sprintf("  Epoch: %d\n", last_metric$epoch))
    cat(sprintf("  Discriminator accuracy: %.2f%%\n", last_metric$d_accuracy * 100))
    cat(sprintf("  Generator diversity: %.4f\n", last_metric$diversity))
  }

  invisible(x)
}


#' @title Plot GAN Training Losses
#'
#' @description Plots the generator and discriminator loss curves from GAN training.
#'   Requires the GAN to have been trained with `track_loss = TRUE`.
#'
#' @param trained_gan A trained GAN object of class "trained_RGAN" with tracked losses
#' @param smooth Smoothing factor for the loss curves (0 = no smoothing, higher = more smoothing).
#'   Uses exponential moving average. Defaults to 0.
#' @param ... Additional arguments passed to plot()
#'
#' @return Invisibly returns NULL. Called for side effect of producing a plot.
#' @export
#'
#' @examples
#' \dontrun{
#' data <- sample_toydata()
#' transformer <- data_transformer$new()
#' transformer$fit(data)
#' transformed_data <- transformer$transform(data)
#' trained_gan <- gan_trainer(transformed_data, epochs = 50, track_loss = TRUE)
#' plot_losses(trained_gan)
#' plot_losses(trained_gan, smooth = 0.9)  # With smoothing
#' }
plot_losses <- function(trained_gan, smooth = 0, ...) {
  if (!inherits(trained_gan, "trained_RGAN")) {
    stop("trained_gan must be an object of class 'trained_RGAN'")
  }

  if (is.null(trained_gan$losses)) {
    stop("No loss data available. Train with track_loss = TRUE to record losses.")
  }

  g_loss <- trained_gan$losses$g_loss
  d_loss <- trained_gan$losses$d_loss

  # Apply exponential moving average smoothing
  if (smooth > 0 && smooth < 1) {
    ema <- function(x, alpha) {
      result <- numeric(length(x))
      result[1] <- x[1]
      for (i in 2:length(x)) {
        result[i] <- alpha * result[i-1] + (1 - alpha) * x[i]
      }
      result
    }
    g_loss <- ema(g_loss, smooth)
    d_loss <- ema(d_loss, smooth)
  }

  steps <- seq_along(g_loss)

  # Set up plot
  oldpar <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(oldpar))

  y_range <- range(c(g_loss, d_loss), na.rm = TRUE)

  graphics::plot(
    steps, g_loss,
    type = "l",
    col = viridis::viridis(2)[1],
    ylim = y_range,
    xlab = "Training Step",
    ylab = "Loss",
    main = "GAN Training Losses",
    bty = "n",
    las = 1,
    ...
  )

  graphics::lines(steps, d_loss, col = viridis::viridis(2)[2])

  graphics::legend(
    "topright",
    legend = c("Generator", "Discriminator"),
    col = viridis::viridis(2),
    lty = 1,
    bty = "n"
  )

  invisible(NULL)
}
