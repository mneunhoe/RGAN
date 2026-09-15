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
#' @param secure_rng Use cryptographically secure RNG from OpenDP for noise generation.
#'   Defaults to TRUE. Set to FALSE for faster training during development/testing
#'   (uses torch's standard RNG which is not cryptographically secure).
#' @param checkpoint_epochs Interval for saving model checkpoints (in epochs). If NULL (default),
#'   no checkpoints are saved. Checkpoints are required for post-GAN boosting.
#' @param checkpoint_path Optional path for disk-based checkpoint persistence. If NULL (default),
#'   checkpoints are stored in memory only.
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
    verbose = TRUE,
    secure_rng = TRUE,
    checkpoint_epochs = NULL,
    checkpoint_path = NULL
) {
  # Check for OpenDP availability (only required if secure_rng = TRUE)
  if (secure_rng && !requireNamespace("opendp", quietly = TRUE)) {
    stop("Package 'opendp' is required when secure_rng = TRUE. Install it with: install.packages('opendp')")
  }

  if (!secure_rng && verbose) {
    cli::cli_alert_warning(
      "secure_rng = FALSE: Using torch RNG (faster but not cryptographically secure)"
    )
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

  # Initialize checkpoint storage
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
        noise_multiplier = noise_multiplier,
        secure_rng = secure_rng
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

    # Save checkpoints at specified intervals
    if (!is.null(checkpoint_epochs) && epoch %% checkpoint_epochs == 0) {
      if (!is.null(checkpoint_path)) {
        # Disk-based storage
        d_path <- file.path(checkpoint_path, sprintf("discriminator_epoch_%04d.pt", epoch))
        g_path <- file.path(checkpoint_path, sprintf("generator_epoch_%04d.pt", epoch))
        torch::torch_save(d_net$state_dict(), d_path)
        torch::torch_save(g_net$state_dict(), g_path)
        checkpoints$epochs <- c(checkpoints$epochs, epoch)
        checkpoints$discriminators[[length(checkpoints$discriminators) + 1]] <- d_path
        checkpoints$generators[[length(checkpoints$generators) + 1]] <- g_path
      } else {
        # In-memory storage (clone to avoid reference issues)
        checkpoints$epochs <- c(checkpoints$epochs, epoch)
        d_state <- lapply(d_net$state_dict(), function(x) x$clone())
        g_state <- lapply(g_net$state_dict(), function(x) x$clone())
        checkpoints$discriminators[[length(checkpoints$discriminators) + 1]] <- d_state
        checkpoints$generators[[length(checkpoints$generators) + 1]] <- g_state
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
    checkpoints = checkpoints,
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
      device = device,
      eval_dropout = eval_dropout,
      synthetic_examples = synthetic_examples
    )
  )
  class(output) <- "trained_RGAN"
  return(output)
}


#' @title Numerically Stable log-sum-exp
#'
#' @param x Numeric vector of log-scale terms.
#' @return \code{log(sum(exp(x)))}, computed without overflow.
#' @keywords internal
log_sum_exp <- function(x) {
  m <- max(x)
  if (!is.finite(m)) return(m)
  m + log(sum(exp(x - m)))
}


#' @title Log-Space Addition and Subtraction
#'
#' @param a,b Log-scale scalars.
#' @return \code{log(exp(a) + exp(b))} and \code{log(exp(a) - exp(b))}.
#' @keywords internal
#' @name log_arith
NULL

#' @rdname log_arith
log_add_exp <- function(a, b) {
  if (!is.finite(a) && a < 0) return(b)
  if (!is.finite(b) && b < 0) return(a)
  m <- max(a, b)
  m + log1p(exp(-abs(a - b)))
}

#' @rdname log_arith
log_sub_exp <- function(a, b) {
  if (!is.finite(b) && b < 0) return(a)
  if (a <= b) return(-Inf)
  a + log1p(-exp(b - a))
}

#' @title Log of the Complementary Error Function
#'
#' @description \code{log(erfc(x))}, evaluated through the normal distribution
#'   function so that it stays accurate deep in the tail where \code{erfc}
#'   itself underflows.
#'
#' @param x Numeric vector.
#' @return \code{log(erfc(x))}.
#' @keywords internal
log_erfc <- function(x) log(2) + stats::pnorm(-x * sqrt(2), log.p = TRUE)


#' @title Log Moment of the Sampled Gaussian at an Integer Order
#'
#' @param q Poisson sampling rate in (0, 1).
#' @param sigma Noise multiplier.
#' @param alpha Integer order, at least 2.
#' @return \code{log A_alpha}.
#' @keywords internal
log_moment_sgm_int <- function(q, sigma, alpha) {
  k <- 0:alpha
  log_sum_exp(lchoose(alpha, k) +
                (alpha - k) * log1p(-q) +
                k * log(q) +
                k * (k - 1) / (2 * sigma^2))
}


#' @title Log Moment of the Sampled Gaussian at a Fractional Order
#'
#' @description For non-integer orders the binomial expansion no longer
#'   terminates. The real line is split at \code{z0}, where the two components
#'   of the mixture cross, and the moment is summed as two series whose terms
#'   carry Gaussian tail factors.
#'
#' @details
#' The terms decay only polynomially, as \code{i^-(alpha + 2)}: the Gaussian
#' factor and the complementary error function very nearly cancel, leaving the
#' binomial coefficient's own decay. An absolute threshold on the term size is
#' therefore the wrong stopping rule -- at \code{alpha = 1.25} it would need
#' about \code{1e10} terms -- so the series stops on a bound for the whole
#' remaining tail instead. Since the terms are eventually decreasing, that tail
#' is at most \code{i * term_i}, which is the conservative estimate used here
#' (the true tail is smaller by a factor of about \code{alpha + 1}).
#'
#' Coefficients alternate in sign beyond \code{ceiling(alpha)}. Positive and
#' negative contributions are accumulated separately and subtracted once, which
#' avoids repeated cancellation in log space; the single final subtraction is
#' benign because neither total is much larger than the result.
#'
#' @param q Poisson sampling rate in (0, 1).
#' @param sigma Noise multiplier.
#' @param alpha Non-integer order, greater than 1.
#' @param max_terms Safety cap on the number of series terms.
#' @param reltol Relative accuracy required of the truncated tail.
#' @return \code{log A_alpha}.
#' @keywords internal
log_moment_sgm_frac <- function(q, sigma, alpha, max_terms = 100000L,
                                reltol = 1e-10) {
  z0 <- sigma^2 * log(1 / q - 1) + 0.5
  log_pos <- -Inf
  log_neg <- -Inf
  converged <- FALSE
  block <- 2048L
  start <- 0

  repeat {
    # Doubles, not integers: the term needs i^2, which overflows R's 32-bit
    # integer type past i = 46341 and would silently become NA.
    i <- start + as.numeric(seq_len(block)) - 1
    coef <- choose(alpha, i)
    log_coef <- log(abs(coef))
    j <- alpha - i

    log_s0 <- log_coef + i * log(q) + j * log1p(-q) +
      (i * i - i) / (2 * sigma^2) +
      log(0.5) + log_erfc((i - z0) / (sqrt(2) * sigma))
    log_s1 <- log_coef + j * log(q) + i * log1p(-q) +
      (j * j - j) / (2 * sigma^2) +
      log(0.5) + log_erfc((z0 - j) / (sqrt(2) * sigma))

    terms <- c(log_s0, log_s1)
    if (anyNA(terms)) {
      stop("The series for the sampled-Gaussian moment at order ", alpha,
           " produced a non-finite term (q = ", q, ", sigma = ", sigma, ").",
           call. = FALSE)
    }

    # `coef == 0` marks a terminating expansion (integer alpha reached through
    # this path); those terms are -Inf and contribute nothing.
    add <- coef > 0
    sub <- coef < 0
    if (any(add)) {
      log_pos <- log_add_exp(log_pos, log_sum_exp(c(log_s0[add], log_s1[add])))
    }
    if (any(sub)) {
      log_neg <- log_add_exp(log_neg, log_sum_exp(c(log_s0[sub], log_s1[sub])))
    }

    total <- log_sub_exp(log_pos, log_neg)
    last <- max(log_s0[block], log_s1[block])

    # Bound the remaining tail by i * term_i and compare it with the running
    # total. `last == -Inf` is exact termination.
    if (!is.finite(last) && last < 0) {
      converged <- TRUE
      break
    }
    if (is.finite(total) && last + log(max(i)) < log(reltol) + total) {
      converged <- TRUE
      break
    }

    start <- start + block
    if (start >= max_terms) break
  }

  if (!converged) {
    stop("The series for the sampled-Gaussian moment at order ", alpha,
         " did not reach a relative accuracy of ", reltol, " within ",
         max_terms, " terms (q = ", q, ", sigma = ", sigma,
         "). Refusing to report a privacy loss that may be understated; ",
         "use integer orders only.", call. = FALSE)
  }

  log_sub_exp(log_pos, log_neg)
}


#' @title RDP of the Sampled Gaussian Mechanism
#'
#' @description Renyi differential privacy of a single step of the Sampled
#'   Gaussian Mechanism: Poisson subsampling at rate \code{q} followed by
#'   Gaussian noise with multiplier \code{sigma} relative to the L2 sensitivity.
#'
#' @details
#' For integer orders \code{alpha >= 2} the Renyi divergence is bounded by
#' \deqn{\frac{1}{\alpha-1}\log\sum_{k=0}^{\alpha}\binom{\alpha}{k}
#'   (1-q)^{\alpha-k}q^{k}\exp\!\left(\frac{k(k-1)}{2\sigma^{2}}\right),}
#' the bound of Mironov, Talwar and Zhang (2019). It follows from expanding the
#' density ratio \eqn{(1-q) + q\exp((2z-1)/2\sigma^{2})} with the binomial
#' theorem and applying the Gaussian moment generating function term by term.
#'
#' The sum is evaluated in log space, which keeps it finite for the large
#' exponents that arise when \code{sigma} is small.
#'
#' Non-integer orders are supported as well, through the series of
#' \code{log_moment_sgm_frac()}. They matter: when the optimal order is small,
#' restricting the search to integers inflates the reported epsilon by up to
#' 11 percent, because the relative gap between consecutive integer orders is
#' largest exactly there.
#'
#' @section Machine-checked statement:
#' At integer orders the value returned here is the conclusion of a Lean 4
#' proof against Mathlib, not only a transcription of the published bound. In
#' the accompanying formalization the quantity
#' \code{sum_k choose(a, k) (1-q)^(a-k) q^k exp(k(k-1)/(2 v))} is
#' \code{sgmMoment q v a} with \code{v = sigma^2}, and:
#' \itemize{
#'   \item \code{renyiDiv_sgmMixture} proves \code{log(sgmMoment)/(a-1)} is
#'     exactly the Renyi divergence of the two-point mixture pair;
#'   \item \code{renyiDiv_sgmMechanism_le} proves it bounds the divergence of
#'     the mechanism's own outputs on adjacent datasets -- the subsampling
#'     reduction of Mironov, Talwar and Zhang, proved rather than assumed.
#' }
#' Both are free of \code{sorry} and depend only on \code{propext},
#' \code{Classical.choice} and \code{Quot.sound}.
#'
#' The proof fixes the setting the caller must respect: \strong{add/remove-one
#' adjacency}; per-record contributions clipped in L2 norm to the sensitivity
#' that \code{sigma} is measured against; and the release being the noisy
#' \emph{sum} over the Poisson draw, divided by nothing data-dependent. See
#' \code{dp_discriminator_step()}.
#'
#' Two gaps remain between the proof and this function. The formalization is
#' carried out for a scalar release, the multivariate case being the same
#' argument with the Gaussian density in product form; and it covers
#' \strong{integer orders only}, because the binomial sum is finite only there.
#' Fractional orders are therefore computed by an uncertified series. They are
#' checked at run time against the certified integer orders that bracket them,
#' using monotonicity of the Renyi divergence in its order: a fractional value
#' must lie between the values at \code{floor(alpha)} and \code{ceiling(alpha)},
#' and an error is raised if it does not.
#'
#' @param q Poisson sampling rate, in \code{[0, 1]}.
#' @param sigma Noise multiplier, strictly positive.
#' @param alpha RDP order, any real number greater than 1.
#' @param validate Check a fractional order against the certified integer
#'   orders that bracket it. Has no effect at integer orders.
#'
#' @return RDP at order \code{alpha} for one step.
#'
#' @references
#' Mironov, I., Talwar, K., & Zhang, L. (2019). Renyi Differential Privacy of
#' the Sampled Gaussian Mechanism. arXiv:1908.10530.
#'
#' @keywords internal
rdp_sgm <- function(q, sigma, alpha, validate = TRUE) {
  if (!is.numeric(sigma) || length(sigma) != 1L || !is.finite(sigma) || sigma <= 0) {
    stop("sigma must be a single positive finite number.", call. = FALSE)
  }
  if (!is.numeric(q) || length(q) != 1L || !is.finite(q) || q < 0 || q > 1) {
    stop("q must be a single number in [0, 1].", call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
      alpha <= 1) {
    stop("alpha must be a single order greater than 1; got ", alpha, ".",
         call. = FALSE)
  }

  # No sampling: nothing is released. Full sampling: no amplification, so the
  # bound is the plain Gaussian mechanism's alpha / (2 sigma^2).
  if (q == 0) return(0)
  if (q == 1) return(alpha / (2 * sigma^2))

  if (isTRUE(all.equal(alpha, round(alpha)))) {
    a <- as.integer(round(alpha))
    return(log_moment_sgm_int(q, sigma, a) / (a - 1))
  }

  value <- log_moment_sgm_frac(q, sigma, alpha) / (alpha - 1)
  if (validate) check_rdp_bracket(value, q, sigma, alpha)
  value
}


#' @title Check a Fractional-Order RDP Against the Certified Integer Orders
#'
#' @description The Renyi divergence is non-decreasing in its order, so the
#'   value at a fractional order must lie between the values at the integer
#'   orders on either side of it. Those are computed by the terminating
#'   binomial sum, which is the form carrying a Lean proof, so this turns the
#'   uncertified series into something checked at run time.
#'
#' @details Only the upper side of the bracket exists for \code{alpha} in
#'   \code{(1, 2)}, since order 1 is the KL divergence and is not covered by
#'   \code{log_moment_sgm_int()}.
#'
#' @param value The RDP returned by the series.
#' @param q Poisson sampling rate.
#' @param sigma Noise multiplier.
#' @param alpha Fractional order.
#'
#' @return \code{value}, invisibly; raises an error if the bracket fails.
#'
#' @keywords internal
check_rdp_bracket <- function(value, q, sigma, alpha) {
  int_rdp <- function(a) log_moment_sgm_int(q, sigma, a) / (a - 1)
  lo_order <- floor(alpha)
  hi_order <- ceiling(alpha)
  tol <- 1e-8 * max(1, abs(value))

  if (lo_order >= 2 && value < int_rdp(lo_order) - tol) {
    stop("The series for the sampled-Gaussian moment at order ", alpha,
         " returned ", value, ", below the certified value at order ",
         lo_order, " (", int_rdp(lo_order), "). The Renyi divergence is ",
         "non-decreasing in its order, so this privacy loss is understated.",
         call. = FALSE)
  }
  if (hi_order >= 2 && value > int_rdp(hi_order) + tol) {
    stop("The series for the sampled-Gaussian moment at order ", alpha,
         " returned ", value, ", above the certified value at order ",
         hi_order, " (", int_rdp(hi_order), ").", call. = FALSE)
  }

  invisible(value)
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
#' Every step of the accounting has a machine-checked counterpart in the
#' accompanying Lean 4 formalization:
#' \itemize{
#'   \item the per-step bound is \code{renyiDiv_sgmMechanism_le}; see
#'     \code{rdp_sgm()} for the setting it assumes;
#'   \item that RDP adds over the steps of an adaptively chosen sequence is
#'     \code{renyiDivFin_transcript_le};
#'   \item the conversion to \code{(epsilon, delta)}-DP is
#'     \code{approxDP_transcript_cks}, which yields exactly the expression
#'     \code{get_epsilon()} minimises;
#'   \item \code{cksEps_le_classicalEps} proves that expression is never worse
#'     than the classical \code{rdp + log(1 / delta) / (alpha - 1)}.
#' }
#' Minimising over a grid of orders is sound because each order gives a valid
#' guarantee on its own.
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
    #' @field orders RDP orders searched when converting to (eps, delta)-DP
    orders = NULL,
    #' @field certified_only Restrict the search to integer orders, where the
    #'   per-step bound is machine-checked
    certified_only = FALSE,

    #' @description Create a new privacy accountant
    #' @param sampling_rate Poisson sampling probability
    #' @param noise_multiplier Noise scale relative to sensitivity
    #' @param target_delta Target delta for DP guarantee
    #' @param orders RDP orders to search. The default grid is fine between 1
    #'   and 11, where the optimum usually sits and where integer orders are far
    #'   apart in relative terms, and coarsens above that.
    #' @param certified_only Keep only the integer orders in \code{orders}. At
    #'   those the per-step bound is the conclusion of a Lean proof; at
    #'   fractional orders it is an uncertified series, checked at run time
    #'   against the integer orders bracketing it. Restricting to integers
    #'   raises the reported epsilon -- by up to 11 percent when the optimal
    #'   order is small -- so it trades tightness for a fully checked number.
    initialize = function(sampling_rate, noise_multiplier, target_delta = 1e-5,
                          orders = c(seq(1.1, 10.9, by = 0.1), 11:64,
                                     seq(66, 256, by = 2)),
                          certified_only = FALSE) {
      self$sampling_rate <- sampling_rate
      self$noise_multiplier <- noise_multiplier
      self$target_delta <- target_delta
      self$certified_only <- isTRUE(certified_only)
      if (self$certified_only) {
        orders <- orders[abs(orders - round(orders)) < 1e-8]
        if (length(orders) == 0) {
          stop("certified_only = TRUE, but `orders` contains no integer order.",
               call. = FALSE)
        }
      }
      self$orders <- orders
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
      rdp_sgm(self$sampling_rate, self$noise_multiplier, alpha)
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

      orders <- self$orders

      # RDP composes by addition across the steps, since the sampling rate and
      # the noise multiplier are constant throughout training.
      total_rdp <- vapply(orders, function(a) self$compute_rdp(a), numeric(1)) *
        self$steps

      # Conversion of Canonne, Kamath and Steinke (2020), which is uniformly
      # tighter than the classical eps = rdp + log(1 / delta) / (alpha - 1).
      eps <- total_rdp -
        (log(delta) + log(orders)) / (orders - 1) +
        log((orders - 1) / orders)

      eps <- eps[is.finite(eps)]
      if (length(eps) == 0) return(Inf)

      max(0, min(eps))
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


#' @title Fast Poisson Subsampling (Non-Cryptographic)
#'
#' @description Fast Poisson subsampling using R's standard RNG.
#'   Not cryptographically secure but much faster for development/testing.
#'
#' @param n_samples Total number of samples in the dataset
#' @param sampling_rate Probability of including each sample
#'
#' @return Integer vector of selected indices
#'
#' @keywords internal
fast_poisson_subsample <- function(n_samples, sampling_rate) {
  # Use R's fast uniform RNG
  uniforms <- stats::runif(n_samples)
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


#' @title Noise-Only DP-SGD Step
#'
#' @description Perform the release step of DP-SGD when the Poisson draw
#'   selected no records: the clipped-gradient sum is zero, so the mechanism
#'   releases pure noise.
#'
#' @param d_net Discriminator network
#' @param d_optim Discriminator optimizer
#' @param max_grad_norm Maximum gradient norm for clipping
#' @param noise_multiplier Gaussian noise multiplier
#' @param expected_batch_size Divisor, \code{sampling_rate * n_samples}
#' @param secure_rng Use cryptographically secure RNG
#'
#' @return \code{0}, the loss over an empty batch
#'
#' @keywords internal
dp_noise_only_step <- function(d_net, d_optim, max_grad_norm, noise_multiplier,
                               expected_batch_size, secure_rng = TRUE) {
  noise_scale <- noise_multiplier * max_grad_norm

  d_optim$zero_grad()
  for (param in d_net$parameters) {
    zero_grad <- torch::torch_zeros_like(param)
    param$grad <- if (secure_rng) {
      sample_secure_gaussian_like(zero_grad, noise_scale / expected_batch_size)
    } else {
      torch::torch_randn_like(zero_grad) * (noise_scale / expected_batch_size)
    }
  }
  d_optim$step()

  0
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
#' @param secure_rng Use cryptographically secure RNG (slower but secure)
#'
#' @return Scalar loss value
#'
#' @details
#' The quantity released to the optimizer is the Poisson-subsampled sum of
#' per-sample gradients, each clipped to L2 norm \code{max_grad_norm}, perturbed
#' by Gaussian noise of scale \code{noise_multiplier * max_grad_norm}, and then
#' divided by the \emph{expected} batch size \code{sampling_rate * n_samples}.
#'
#' The divisor must not depend on the realised batch size. The privacy analysis
#' -- and the accountant in \code{rdp_sgm()} -- covers the noisy \emph{sum}
#' only; dividing by a constant is post-processing and so is free, but dividing
#' by \code{length(batch_indices)} would make both the location and the scale of
#' the released quantity depend on how many records the Poisson draw selected.
#' That count is itself data-dependent under add/remove-one adjacency, and its
#' contribution to the privacy loss is not accounted for by the Sampled Gaussian
#' Mechanism bound.
#'
#' For the same reason an empty Poisson draw is not a free step: the mechanism
#' still releases noise, and the accountant still charges for it.
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
    noise_multiplier,
    secure_rng = TRUE
) {
  # Poisson subsample the batch
  if (secure_rng) {
    batch_indices <- secure_poisson_subsample(n_samples, sampling_rate)
  } else {
    batch_indices <- fast_poisson_subsample(n_samples, sampling_rate)
  }

  actual_batch_size <- length(batch_indices)

  # The divisor is the expected batch size, which is a constant fixed by the
  # sampling rate and the dataset size. See the details section: using the
  # realised batch size here would leak it.
  expected_batch_size <- sampling_rate * n_samples
  if (!is.finite(expected_batch_size) || expected_batch_size <= 0) {
    stop("sampling_rate * n_samples must be positive; got ",
         expected_batch_size, ". A zero sampling rate releases nothing and ",
         "leaves the normalisation undefined.", call. = FALSE)
  }

  # An empty Poisson draw still releases noise. Returning early instead would
  # make "no optimizer step" an observable event whose probability, (1 - q)^n,
  # depends on the number of records.
  if (actual_batch_size == 0) {
    return(dp_noise_only_step(d_net, d_optim, max_grad_norm, noise_multiplier,
                              expected_batch_size, secure_rng))
  }

  # Get the batch
  real_data <- data[batch_indices]$to(device = device)

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
      # Normalise by the expected batch size, not the realised one.
      avg_grad <- accumulated_grads[[param_idx]] / expected_batch_size

      # Add Gaussian noise (secure or fast)
      if (secure_rng) {
        noise <- sample_secure_gaussian_like(avg_grad, noise_scale / expected_batch_size)
      } else {
        noise <- torch::torch_randn_like(avg_grad) * (noise_scale / expected_batch_size)
      }

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
  eps_at <- function(sigma) {
    accountant <- dp_accountant_poisson$new(
      sampling_rate = sampling_rate,
      noise_multiplier = sigma,
      target_delta = target_delta
    )
    accountant$step(total_steps)
    accountant$get_epsilon()
  }

  # Epsilon is decreasing in the noise multiplier, so the search is only
  # well posed if the largest multiplier considered is already inside budget.
  eps_max <- eps_at(max_noise)
  if (eps_max > target_epsilon) {
    stop("Cannot reach target_epsilon = ", format(target_epsilon),
         " with noise_multiplier <= ", format(max_noise), " over ",
         total_steps, " steps (epsilon = ", format(eps_max), "). Increase ",
         "max_noise, reduce total_steps, or raise target_epsilon.",
         call. = FALSE)
  }
  if (eps_at(min_noise) <= target_epsilon) return(min_noise)

  # Invariant: eps_at(low) > target_epsilon >= eps_at(high). The search only
  # ever returns `high`, so the multiplier handed back is always inside the
  # budget. Returning the midpoint, as earlier versions did, could overshoot
  # the target by up to `tolerance` - the wrong direction for a privacy
  # parameter.
  low <- min_noise
  high <- max_noise
  eps_high <- eps_max

  for (iter in 1:100) {
    if (target_epsilon - eps_high < tolerance || high - low < 1e-6) break
    mid <- (low + high) / 2
    eps_mid <- eps_at(mid)
    if (eps_mid <= target_epsilon) {
      high <- mid
      eps_high <- eps_mid
    } else {
      low <- mid
    }
  }

  high
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
