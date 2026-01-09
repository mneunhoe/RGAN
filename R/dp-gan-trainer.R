#' @title dp_gan_trainer
#'
#' @description Provides a function to train a GAN model with differential privacy
#'   guarantees using DP-SGD (Differentially Private Stochastic Gradient Descent).
#'   Uses OpenDP for cryptographically secure noise generation and Poisson subsampling.
#'
#' @param data Input a data set. Needs to be a matrix, array, or torch::torch_tensor.
#' @param noise_dim The dimensions of the GAN noise vector z. Defaults to 2.
#' @param noise_distribution The noise distribution. Expects a function that samples
#'   from a distribution and returns a torch_tensor. For convenience "normal" and
#'   "uniform" will automatically set a function. Defaults to "normal".
#' @param data_type "tabular" or "image", controls the data type, defaults to "tabular".
#' @param generator The generator network. Expects a neural network provided as
#'   torch::nn_module. Default is NULL which will create a simple fully connected network.
#' @param discriminator The discriminator network. Expects a neural network provided as
#'   torch::nn_module. Default is NULL which will create a simple fully connected network.
#' @param base_lr The base learning rate for the optimizers. Default is 0.0001.
#' @param target_epsilon Target epsilon for differential privacy. Training will
#'   stop if this budget is exhausted. Defaults to 1.0.
#' @param target_delta Target delta for differential privacy. Defaults to 1e-5.
#' @param max_grad_norm Maximum gradient norm for per-sample gradient clipping.
#'   Bounds the sensitivity of individual gradients. Defaults to 1.0.
#' @param noise_multiplier Multiplier for Gaussian noise added to gradients.
#'   If NULL (default), it will be calibrated to achieve target_epsilon over the
#'   specified epochs. Higher values provide more privacy but reduce utility.
#' @param sampling_rate Expected sampling rate for Poisson subsampling. If NULL
#'   (default), calculated as batch_size / nrow(data). Must be between 0 and 1.
#' @param batch_size Target batch size for training. With Poisson subsampling,
#'   actual batch sizes will vary. Defaults to 50.
#' @param epochs The number of training epochs. Defaults to 50.
#' @param plot_progress Monitor training progress with plots. Defaults to FALSE.
#' @param plot_interval Number of training steps between plots. Defaults to "epoch".
#' @param eval_dropout Should dropout be applied during sampling? Defaults to FALSE.
#' @param synthetic_examples Number of synthetic examples to generate. Defaults to 500.
#' @param plot_dimensions Which dimensions to plot. Defaults to c(1, 2).
#' @param track_loss Store training losses as output. Defaults to FALSE.
#' @param device Device for computation ("cpu", "cuda", "mps"). Defaults to "cpu".
#' @param seed Optional seed for reproducibility. Defaults to NULL.
#' @param verbose Print privacy accounting information during training. Defaults to TRUE.
#'
#' @return A list of class "trained_RGAN" containing:
#' \itemize{
#'   \item generator: The trained generator network
#'   \item discriminator: The trained discriminator network
#'   \item losses: Training losses if track_loss is TRUE
#'   \item privacy: Privacy accounting information including final epsilon
#'   \item settings: Training settings used
#' }
#'
#' @details
#' This function implements DP-SGD (Abadi et al., 2016) for training GANs with
#' formal differential privacy guarantees. Key privacy mechanisms include:
#'
#' \strong{Poisson Subsampling}: Each training example is included in a batch
#' independently with probability q = sampling_rate, providing privacy amplification.
#' Uses OpenDP's cryptographically secure random number generation.
#'
#' \strong{Per-Sample Gradient Clipping}: Each sample's gradient is computed
#' individually and clipped to bound sensitivity.
#'
#' \strong{Gaussian Noise}: Calibrated Gaussian noise is added to clipped gradients
#' using OpenDP's secure Gaussian mechanism.
#'
#' \strong{RDP Accounting}: Uses Renyi Differential Privacy for tight composition
#' of privacy loss across training steps.
#'
#' The discriminator is trained with DP-SGD while the generator is trained normally
#' (since it only sees synthetic data from the discriminator's gradients).
#'
#' @references
#' Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016).
#' Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference
#' on computer and communications security (pp. 308-318).
#'
#' Mironov, I. (2017). Renyi differential privacy. In 2017 IEEE 30th computer security
#' foundations symposium (CSF) (pp. 263-275).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Before running, install OpenDP: install.packages("opendp")
#' # and torch: torch::install_torch()
#'
#' # Load data
#' data <- sample_toydata()
#' transformer <- data_transformer$new()
#' transformer$fit(data)
#' transformed_data <- transformer$transform(data)
#'
#' # Train with differential privacy (epsilon = 1)
#' trained_gan <- dp_gan_trainer(
#'   transformed_data,
#'   target_epsilon = 1.0,
#'   target_delta = 1e-5,
#'   max_grad_norm = 1.0,
#'   epochs = 50
#' )
#'
#' # Check final privacy budget
#' print(trained_gan$privacy$final_epsilon)
#'
#' # Sample synthetic data
#' synthetic_data <- sample_synthetic_data(trained_gan, transformer)
#' }
dp_gan_trainer <- function(
    data,
    noise_dim = 2,
    noise_distribution = "normal",
    data_type = "tabular",
    generator = NULL,
    discriminator = NULL,
    base_lr = 0.0001,
    target_epsilon = 1.0,
    target_delta = 1e-5,
    max_grad_norm = 1.0,
    noise_multiplier = NULL,
    sampling_rate = NULL,
    batch_size = 50,
    epochs = 50,
    plot_progress = FALSE,
    plot_interval = "epoch",
    eval_dropout = FALSE,
    synthetic_examples = 500,
    plot_dimensions = c(1, 2),
    track_loss = FALSE,
    device = "cpu",
    seed = NULL,
    verbose = TRUE
) {
  # Check for OpenDP availability
  if (!requireNamespace("opendp", quietly = TRUE)) {
    stop("Package 'opendp' is required for dp_gan_trainer. Install it with: install.packages('opendp')")
  }

  # Set random seeds for reproducibility
  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }

  # Input validation
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
  if (target_epsilon <= 0) {
    stop("target_epsilon must be a positive number")
  }
  if (target_delta <= 0 || target_delta >= 1) {
    stop("target_delta must be between 0 and 1 (exclusive)")
  }
  if (max_grad_norm <= 0) {
    stop("max_grad_norm must be a positive number")
  }
  if (!is.null(noise_multiplier) && noise_multiplier <= 0) {
    stop("noise_multiplier must be a positive number")
  }
  if (!is.null(sampling_rate) && (sampling_rate <= 0 || sampling_rate > 1)) {
    stop("sampling_rate must be between 0 (exclusive) and 1 (inclusive)")
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

  # Check data format
  if (!(any(c("matrix", "array", "torch_tensor") %in% class(data)))) {
    stop("Data needs to be a matrix, array or torch::torch_tensor for DP training.")
  }

  # Convert to tensor if needed
  if (any(c("array", "matrix") %in% class(data))) {
    if (nrow(data) == 0) {
      stop("data cannot be empty")
    }
    if (ncol(data) == 0) {
      stop("data must have at least one column")
    }
    if (all(is.na(data))) {
      stop("data cannot be all NA values")
    }
    data <- torch::torch_tensor(data)$to(device = "cpu")
  }

  n_samples <- nrow(data)
  data_dim <- ncol(data)

  # Calculate sampling rate
  if (is.null(sampling_rate)) {
    sampling_rate <- min(batch_size / n_samples, 1.0)
  }

  # Calculate number of steps
  steps_per_epoch <- ceiling(1 / sampling_rate)

  # Set plotting interval
  plot_interval <- ifelse(plot_interval == "epoch", steps_per_epoch, plot_interval)

  # Calibrate noise multiplier if not provided
  if (is.null(noise_multiplier)) {
    total_steps <- epochs * steps_per_epoch
    noise_multiplier <- calibrate_noise_multiplier(
      target_epsilon = target_epsilon,
      target_delta = target_delta,
      sampling_rate = sampling_rate,
      total_steps = total_steps
    )
    if (verbose) {
      cli::cli_alert_info(sprintf(
        "Calibrated noise_multiplier = %.4f for epsilon = %.2f over %d steps",
        noise_multiplier, target_epsilon, total_steps
      ))
    }
  }

  # Initialize privacy accountant
  accountant <- dp_accountant_poisson$new(
    sampling_rate = sampling_rate,
    noise_multiplier = noise_multiplier,
    target_delta = target_delta
  )

  # Pre-calculate maximum steps before privacy budget is exhausted
  # This avoids computing epsilon at every step
  max_steps <- compute_max_steps(
    target_epsilon = target_epsilon,
    target_delta = target_delta,
    sampling_rate = sampling_rate,
    noise_multiplier = noise_multiplier
  )

  total_planned_steps <- epochs * steps_per_epoch
  if (max_steps < total_planned_steps && verbose) {
    cli::cli_alert_info(sprintf(
      "Privacy budget allows %d steps (planned: %d). Training will stop early.",
      max_steps, total_planned_steps
    ))
  }

  # Set up neural networks
  if (is.null(generator)) {
    g_net <- Generator(
      noise_dim = noise_dim,
      data_dim = data_dim,
      dropout_rate = 0.5
    )$to(device = device)
  } else {
    g_net <- generator
  }

  if (is.null(discriminator)) {
    d_net <- Discriminator(
      data_dim = data_dim,
      dropout_rate = 0.5,
      sigmoid = TRUE
    )$to(device = device)
  } else {
    d_net <- discriminator
  }

  # Set up optimizers (no momentum for DP-SGD)
  g_optim <- torch::optim_sgd(g_net$parameters, lr = base_lr)
  d_optim <- torch::optim_sgd(d_net$parameters, lr = base_lr)

  # Set up noise distribution
  if (inherits(noise_distribution, "function")) {
    sample_noise <- noise_distribution
  } else {
    if (noise_distribution == "normal") {
      sample_noise <- torch::torch_randn
    } else if (noise_distribution == "uniform") {
      sample_noise <- torch_rand_ab
    }
  }

  # Sample fixed noise for progress visualization
  fixed_z <- sample_noise(c(synthetic_examples, noise_dim))$to(device = device)

  # Initialize progress tracking
  cli::cli_progress_bar("Training DP-GAN", total = epochs * steps_per_epoch)
  losses <- NULL

  # Main training loop
  for (epoch in 1:epochs) {
    for (step in 1:steps_per_epoch) {
      # DP-SGD update for discriminator
      d_loss <- dp_discriminator_step(
        data = data,
        n_samples = n_samples,
        sampling_rate = sampling_rate,
        noise_dim = noise_dim,
        sample_noise = sample_noise,
        device = device,
        g_net = g_net,
        d_net = d_net,
        d_optim = d_optim,
        max_grad_norm = max_grad_norm,
        noise_multiplier = noise_multiplier
      )

      # Update step counter
      accountant$step()

      # Check if privacy budget exhausted (using pre-calculated max_steps)
      if (accountant$steps >= max_steps) {
        cli::cli_alert_warning(sprintf(
          "Privacy budget exhausted at epoch %d, step %d (max_steps = %d).",
          epoch, step, max_steps
        ))
        break
      }

      # Standard generator update (no DP needed - uses only synthetic data)
      g_loss <- generator_step(
        batch_size = ceiling(sampling_rate * n_samples),
        noise_dim = noise_dim,
        sample_noise = sample_noise,
        device = device,
        g_net = g_net,
        d_net = d_net,
        g_optim = g_optim
      )

      # Track losses
      if (track_loss) {
        if (is.null(losses)) {
          losses <- list(
            g_loss = g_loss,
            d_loss = d_loss
          )
        } else {
          losses$g_loss <- c(losses$g_loss, g_loss)
          losses$d_loss <- c(losses$d_loss, d_loss)
        }
      }

      cli::cli_progress_update()

      # Plot progress
      global_step <- (epoch - 1) * steps_per_epoch + step
      if (plot_progress && global_step %% plot_interval == 0) {
        synth_data <- expert_sample_synthetic_data(
          g_net, fixed_z, device, eval_dropout = eval_dropout
        )
        if (data_type == "tabular") {
          GAN_update_plot(
            data = data,
            dimensions = plot_dimensions,
            synth_data = synth_data,
            epoch = epoch
          )
        }
      }
    }

    # Check if privacy budget exhausted
    if (accountant$steps >= max_steps) {
      break
    }

    # Print privacy status (compute epsilon only for verbose output)
    if (verbose && epoch %% 10 == 0) {
      current_epsilon <- accountant$get_epsilon()
      cli::cli_alert_info(sprintf(
        "Epoch %d/%d - Steps: %d/%d - Current epsilon: %.4f",
        epoch, epochs, accountant$steps, max_steps, current_epsilon
      ))
    }
  }

  cli::cli_progress_done()

  # Final privacy accounting
  final_epsilon <- accountant$get_epsilon()
  if (verbose) {
    cli::cli_alert_success(sprintf(
      "Training complete. Final (epsilon, delta) = (%.4f, %.2e)",
      final_epsilon, target_delta
    ))
  }

  # Prepare output
  output <- list(
    generator = g_net,
    discriminator = d_net,
    generator_optimizer = g_optim,
    discriminator_optimizer = d_optim,
    losses = losses,
    privacy = list(
      final_epsilon = final_epsilon,
      delta = target_delta,
      noise_multiplier = noise_multiplier,
      max_grad_norm = max_grad_norm,
      sampling_rate = sampling_rate,
      total_steps = accountant$steps
    ),
    settings = list(
      noise_dim = noise_dim,
      noise_distribution = noise_distribution,
      sample_noise = sample_noise,
      data_type = data_type,
      base_lr = base_lr,
      batch_size = batch_size,
      epochs = epochs,
      target_epsilon = target_epsilon,
      target_delta = target_delta,
      device = device
    )
  )
  class(output) <- "trained_RGAN"
  return(output)
}


#' @title RDP Privacy Accountant for Poisson Subsampling
#'
#' @description R6 class for tracking privacy loss using Renyi Differential Privacy
#'   with Poisson subsampling amplification.
#'
#' @details
#' Implements the privacy accounting from Mironov (2017) with subsampling
#' amplification. RDP provides tighter composition than basic composition
#' and converts to (epsilon, delta)-DP guarantees.
#'
#' @keywords internal
dp_accountant_poisson <- R6::R6Class(
  "dp_accountant_poisson",

  public = list(
    #' @field steps Number of steps taken
    steps = 0,
    #' @field sampling_rate Poisson sampling rate
    sampling_rate = NULL,
    #' @field noise_multiplier Gaussian noise multiplier
    noise_multiplier = NULL,
    #' @field target_delta Target delta for conversion
    target_delta = NULL,

    #' @description Create a new privacy accountant
    #' @param sampling_rate Poisson sampling probability
    #' @param noise_multiplier Noise scale relative to sensitivity
    #' @param target_delta Target delta for DP guarantee
    initialize = function(sampling_rate, noise_multiplier, target_delta = 1e-5) {
      self$sampling_rate <- sampling_rate
      self$noise_multiplier <- noise_multiplier
      self$target_delta <- target_delta
    },

    #' @description Record one or more privacy-consuming steps
    #' @param n Number of steps to record
    step = function(n = 1) {
      self$steps <- self$steps + n
    },

    #' @description Compute RDP at a given order
    #' @param alpha RDP order (must be > 1)
    #' @return RDP value at order alpha for one step
    compute_rdp = function(alpha) {
      q <- self$sampling_rate
      sigma <- self$noise_multiplier

      if (q == 0) return(0)

      # Base Gaussian mechanism RDP
      base_rdp <- alpha / (2 * sigma^2)

      # Apply subsampling amplification
      # For small q, use simplified formula: RDP <= q^2 * base_rdp
      # For larger q, use the full formula from Mironov et al.
      if (q < 0.01) {
        return(q^2 * base_rdp)
      } else {
        # Full formula for subsampled Gaussian
        log_term1 <- log(1 - q)
        log_term2 <- log(q) + (alpha - 1) * base_rdp
        rdp <- (log(exp(log_term1) + exp(log_term2))) / (alpha - 1)
        return(max(0, rdp))
      }
    },

    #' @description Get (epsilon, delta)-DP guarantee
    #' @param delta Target delta (defaults to target_delta)
    #' @return Epsilon value for the given delta
    get_epsilon = function(delta = NULL) {
      if (is.null(delta)) delta <- self$target_delta

      # Handle zero steps case
      if (self$steps == 0) {
        return(0)
      }

      # Search over RDP orders to find best epsilon
      orders <- c(1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256)
      best_eps <- Inf

      for (alpha in orders) {
        # Total RDP via composition
        total_rdp <- self$compute_rdp(alpha) * self$steps

        # Convert to (epsilon, delta)-DP
        eps <- total_rdp + log(1 / delta) / (alpha - 1)

        # Check for valid epsilon (finite and positive)
        if (is.finite(eps) && eps > 0 && eps < best_eps) {
          best_eps <- eps
        }
      }

      return(best_eps)
    }
  )
)


#' @title Secure Poisson Subsampling using OpenDP
#'
#' @description Sample indices for a mini-batch using Poisson subsampling with
#'   cryptographically secure random numbers from OpenDP.
#'
#' @param n_samples Total number of samples in the dataset
#' @param sampling_rate Probability of including each sample
#'
#' @return Integer vector of selected indices
#'
#' @details
#' Implements Poisson subsampling where each record is included independently
#' with probability q = sampling_rate. This is required for standard DP-SGD
#' privacy analysis (privacy amplification by subsampling).
#'
#' Uses OpenDP's cryptographically secure Gaussian mechanism to generate
#' secure random numbers, then transforms to uniform via the CDF.
#'
#' @keywords internal
secure_poisson_subsample <- function(n_samples, sampling_rate) {
  # Enable OpenDP contrib features (required for make_gaussian)
  opendp::enable_features("contrib")

  # Use OpenDP's make_gaussian with proper domain and metric
  # Set nan = FALSE to exclude NaN values (required for Gaussian mechanism)
  input_space <- opendp::vector_domain(
    opendp::atom_domain(.T = "f64", nan = FALSE),
    size = as.integer(n_samples)  # Must be integer
  )
  input_metric <- opendp::l2_distance(.T = "f64")

  # Create measurement with Gaussian noise (scale=1 for standard normal)
  meas <- opendp::make_gaussian(input_space, input_metric, scale = 1.0)

  # Generate secure Gaussian samples
  zeros <- rep(0.0, n_samples)
  secure_normals <- meas(arg = zeros)  # OpenDP requires arg = parameter

  # Transform to uniform via standard normal CDF
  uniforms <- stats::pnorm(secure_normals)

  # Select samples where uniform < sampling_rate
  which(uniforms < sampling_rate)
}


#' @title Sample Secure Gaussian Noise using OpenDP
#'
#' @description Generate Gaussian noise with the same shape as a tensor using
#'   OpenDP's cryptographically secure random number generator.
#'
#' @param tensor A torch tensor whose shape to match
#' @param scale Standard deviation of the Gaussian noise
#'
#' @return A torch tensor of noise with the same shape and device as input
#'
#' @details
#' Uses OpenDP's Gaussian mechanism to generate cryptographically secure
#' noise samples, then reshapes to match the input tensor.
#'
#' @keywords internal
sample_secure_gaussian_like <- function(tensor, scale) {
  # Enable OpenDP contrib features (required for make_gaussian)
  opendp::enable_features("contrib")

  shape_vec <- as.integer(tensor$shape)
  n_elements <- as.integer(prod(shape_vec))

  # Create OpenDP measurement for secure Gaussian sampling
  # Set nan = FALSE to exclude NaN values (required for Gaussian mechanism)
  input_space <- opendp::vector_domain(
    opendp::atom_domain(.T = "f64", nan = FALSE),
    size = n_elements  # Already integer from as.integer above
  )
  input_metric <- opendp::l2_distance(.T = "f64")
  meas <- opendp::make_gaussian(input_space, input_metric, scale = scale)

  # Generate secure noise
  zeros <- rep(0.0, n_elements)
  noisy <- meas(arg = zeros)  # OpenDP requires arg = parameter

  # Convert to torch tensor with same device
  torch::torch_tensor(noisy, device = tensor$device)$view(shape_vec)
}


#' @title DP-SGD Discriminator Update Step
#'
#' @description Perform one DP-SGD update step for the discriminator with
#'   per-sample gradient clipping and secure Gaussian noise.
#'
#' @param data Full training dataset as torch tensor
#' @param n_samples Number of samples in dataset
#' @param sampling_rate Poisson sampling probability
#' @param noise_dim Dimension of generator noise
#' @param sample_noise Function to sample generator noise
#' @param device Computation device
#' @param g_net Generator network
#' @param d_net Discriminator network
#' @param d_optim Discriminator optimizer
#' @param max_grad_norm Maximum gradient norm for clipping
#' @param noise_multiplier Gaussian noise multiplier
#'
#' @return Scalar loss value
#'
#' @keywords internal
dp_discriminator_step <- function(
    data,
    n_samples,
    sampling_rate,
    noise_dim,
    sample_noise,
    device,
    g_net,
    d_net,
    d_optim,
    max_grad_norm,
    noise_multiplier
) {
  # Poisson subsample the batch using secure sampling
  batch_indices <- secure_poisson_subsample(n_samples, sampling_rate)

  # Handle empty batch
  if (length(batch_indices) == 0) {
    return(0)
  }

  # Get the batch
  real_data <- data[batch_indices]$to(device = device)
  actual_batch_size <- length(batch_indices)

  # Generate fake data
  z <- sample_noise(c(actual_batch_size, noise_dim))$to(device = device)
  fake_data <- torch::with_no_grad(g_net(z))

  # Zero gradients
  d_optim$zero_grad()

  # Compute per-sample gradients and aggregate with clipping
  # We process each sample individually for proper DP guarantees
  accumulated_grads <- NULL
  total_loss <- 0

  for (i in 1:actual_batch_size) {
    # Get single sample
    real_sample <- real_data[i:i]
    fake_sample <- fake_data[i:i]

    # Forward pass
    real_score <- d_net(real_sample)
    fake_score <- d_net(fake_sample)

    # Binary cross entropy loss for single sample
    loss <- -torch::torch_log(real_score + 1e-8) - torch::torch_log(1 - fake_score + 1e-8)
    total_loss <- total_loss + loss$item()

    # Backward to compute gradients
    loss$backward()

    # Collect and clip gradients for this sample
    sample_grads <- list()
    grad_norm_sq <- 0

    for (param in d_net$parameters) {
      if (!is.null(param$grad)) {
        sample_grads[[length(sample_grads) + 1]] <- param$grad$clone()
        grad_norm_sq <- grad_norm_sq + param$grad$pow(2)$sum()$item()
      }
    }

    # Clip gradient
    grad_norm <- sqrt(grad_norm_sq)
    clip_coef <- min(1.0, max_grad_norm / (grad_norm + 1e-8))

    # Accumulate clipped gradients
    if (is.null(accumulated_grads)) {
      accumulated_grads <- lapply(sample_grads, function(g) g * clip_coef)
    } else {
      for (j in seq_along(sample_grads)) {
        accumulated_grads[[j]] <- accumulated_grads[[j]] + sample_grads[[j]] * clip_coef
      }
    }

    # Zero gradients for next sample
    d_optim$zero_grad()
  }

  # Average gradients and add noise
  noise_scale <- noise_multiplier * max_grad_norm

  param_idx <- 1
  for (param in d_net$parameters) {
    if (!is.null(accumulated_grads[[param_idx]])) {
      # Average the clipped gradients
      avg_grad <- accumulated_grads[[param_idx]] / actual_batch_size

      # Add secure Gaussian noise
      noise <- sample_secure_gaussian_like(avg_grad, noise_scale / actual_batch_size)

      # Set noisy gradient
      param$grad <- avg_grad + noise
    }
    param_idx <- param_idx + 1
  }

  # Optimizer step with noisy gradients
  d_optim$step()

  return(total_loss / actual_batch_size)
}


#' @title Generator Update Step
#'
#' @description Standard (non-private) generator update step.
#'   The generator doesn't need DP since it only accesses synthetic data.
#'
#' @param batch_size Number of samples to generate
#' @param noise_dim Dimension of generator noise
#' @param sample_noise Function to sample noise
#' @param device Computation device
#' @param g_net Generator network
#' @param d_net Discriminator network
#' @param g_optim Generator optimizer
#'
#' @return Scalar loss value
#'
#' @keywords internal
generator_step <- function(
    batch_size,
    noise_dim,
    sample_noise,
    device,
    g_net,
    d_net,
    g_optim
) {
  # Ensure minimum batch size
  batch_size <- max(1, batch_size)

  # Generate fake samples
  z <- sample_noise(c(batch_size, noise_dim))$to(device = device)
  fake_data <- g_net(z)

  # Get discriminator score
  fake_score <- d_net(fake_data)

  # Generator wants to maximize log(D(G(z)))
  g_loss <- -torch::torch_mean(torch::torch_log(fake_score + 1e-8))

  # Update generator
  g_optim$zero_grad()
  g_loss$backward()
  g_optim$step()

  return(g_loss$item())
}


#' @title Calibrate Noise Multiplier for Target Privacy
#'
#' @description Find the noise multiplier that achieves a target epsilon
#'   for a given number of training steps using binary search.
#'
#' @param target_epsilon Target epsilon for differential privacy
#' @param target_delta Target delta for differential privacy
#' @param sampling_rate Poisson sampling probability
#' @param total_steps Total number of training steps
#' @param min_noise Minimum noise multiplier to search. Defaults to 0.1.
#' @param max_noise Maximum noise multiplier to search. Defaults to 100.
#' @param tolerance Convergence tolerance. Defaults to 0.01.
#'
#' @return Calibrated noise multiplier
#'
#' @keywords internal
calibrate_noise_multiplier <- function(
    target_epsilon,
    target_delta,
    sampling_rate,
    total_steps,
    min_noise = 0.1,
    max_noise = 100,
    tolerance = 0.01
) {
  # Binary search for the noise multiplier
  low <- min_noise
  high <- max_noise

  for (iter in 1:100) {
    mid <- (low + high) / 2

    # Create accountant with this noise multiplier
    accountant <- dp_accountant_poisson$new(
      sampling_rate = sampling_rate,
      noise_multiplier = mid,
      target_delta = target_delta
    )
    accountant$step(total_steps)

    eps <- accountant$get_epsilon()

    if (abs(eps - target_epsilon) < tolerance) {
      return(mid)
    }

    if (eps > target_epsilon) {
      # Need more noise (larger multiplier)
      low <- mid
    } else {
      # Can use less noise (smaller multiplier)
      high <- mid
    }

    # Check convergence
    if (high - low < 0.001) {
      break
    }
  }

  return(mid)
}


#' @title Compute Maximum Steps for Privacy Budget
#'
#' @description Find the maximum number of training steps that keeps epsilon
#'   at or below the target using binary search. This allows pre-computing
#'   when training must stop, avoiding per-step epsilon computation.
#'
#' @param target_epsilon Target epsilon for differential privacy
#' @param target_delta Target delta for differential privacy
#' @param sampling_rate Poisson sampling probability
#' @param noise_multiplier Gaussian noise multiplier
#' @param max_steps_search Upper bound for binary search. Defaults to 1000000.
#'
#' @return Maximum number of steps that keeps epsilon <= target_epsilon
#'
#' @keywords internal
compute_max_steps <- function(
    target_epsilon,
    target_delta,
    sampling_rate,
    noise_multiplier,
    max_steps_search = 1000000
) {
  # First check if even 1 step exceeds the budget
  accountant <- dp_accountant_poisson$new(
    sampling_rate = sampling_rate,
    noise_multiplier = noise_multiplier,
    target_delta = target_delta
  )
  accountant$step(1)
  if (accountant$get_epsilon() > target_epsilon) {
    # Even 1 step exceeds budget - return 0
    return(0)
  }

  # Check if max_steps_search is within budget
  accountant <- dp_accountant_poisson$new(
    sampling_rate = sampling_rate,
    noise_multiplier = noise_multiplier,
    target_delta = target_delta
  )
  accountant$step(max_steps_search)
  if (accountant$get_epsilon() <= target_epsilon) {
    return(max_steps_search)
  }

  # Binary search for maximum steps
  low <- 1
  high <- max_steps_search

  for (iter in 1:50) {
    mid <- floor((low + high) / 2)

    accountant <- dp_accountant_poisson$new(
      sampling_rate = sampling_rate,
      noise_multiplier = noise_multiplier,
      target_delta = target_delta
    )
    accountant$step(mid)
    eps <- accountant$get_epsilon()

    if (eps <= target_epsilon) {
      # Can do more steps
      low <- mid
    } else {
      # Need fewer steps
      high <- mid
    }

    # Converged
    if (high - low <= 1) {
      break
    }
  }

  # Return low (the last value that was <= target_epsilon)
  return(low)
}
