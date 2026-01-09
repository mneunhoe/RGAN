logit <- function(p) {
  log(p / (1 - p))
}


logistic <- function(x) {
  1 / (1 + exp(-x))
}


# The rejection_sample function adapts source code from
# https: https://github.com/uber-research/metropolis-hastings-gans/blob/master/mhgan/mh.py,
# Copyright (c) 2018 Uber Technologies, Inc.

rejection_sample <-
  function(d_score,
           epsilon = 1e-6,
           shift_percent = 0.95,
           score_max = NULL,
           random = stats::runif,
           ...) {
    # Rejection scheme from:
    # https://arxiv.org/pdf/1810.06758.pdf


    # Chop off first since we assume that is real point and reject does not
    # start with real point.
    d_score <- d_score[-1]

    # Make sure logit finite
    d_score[d_score == 0] <- 1e-14
    d_score[d_score == 1] <- 1 - 1e-14
    max_burnin_d_score = score_max

    log_M <- logit(max_burnin_d_score)

    D_tilde <- logit(d_score)
    # Bump up M if found something bigger
    D_tilde_M <- max(log_M, cummax(D_tilde))

    D_delta <- D_tilde - D_tilde_M
    Fct <- D_delta - log(1 - exp(D_delta - epsilon))

    if (!is.null(shift_percent)) {
      gamma <- stats::quantile(Fct, shift_percent)
    }
    Fct <- Fct - gamma

    P <- logistic(Fct)
    accept <- random(length(d_score)) <= P


    idx <- which(accept) + 1
    # if np.any(accept):
    #   idx = np.argmax(accept)  # Stop at first true, default to 0
    # else:
    #   idx = np.argmax(d_score)  # Revert to cherry if no accept

    # Now shift idx because we took away the real init point
    return(idx)
  }



#' @title Post-GAN Boosting
#'
#' @description Implements the Post-GAN Boosting algorithm from Neunhoeffer et al. (2021)
#'   "Private Post-GAN Boosting" (ICLR 2021). This algorithm improves the quality of GAN
#'   samples by learning a distribution over candidate samples that fools an ensemble of
#'   discriminators using multiplicative weights.
#'
#' @param d_score_fake Matrix of discriminator scores on fake samples (N_discriminators x N_samples).
#'   Each row contains scores from one discriminator checkpoint for all candidate samples.
#' @param d_score_real Vector of mean discriminator scores on real data (length N_discriminators).
#' @param B Matrix of candidate synthetic samples (N_samples x data_dim).
#' @param real_N Number of real training samples (used for privacy calibration).
#' @param steps Number of boosting iterations. Defaults to 400.
#' @param N_generators Number of discriminator checkpoints used. Defaults to 200.
#' @param uniform_init Initialize phi with uniform distribution. Defaults to TRUE.
#' @param dp Use differential privacy for discriminator selection via exponential mechanism.
#'   Defaults to FALSE.
#' @param MW_epsilon Total privacy budget for multiplicative weights (only if dp=TRUE).
#'   Defaults to 0.1.
#' @param weighted_average Use weighted averaging (weights proportional to sqrt(step)).
#'   Defaults to FALSE.
#' @param averaging_window Number of final steps to average over. Defaults to NULL (all steps).
#'
#' @return A list with:
#'   \itemize{
#'     \item PGB_sample: Matrix of selected high-quality samples
#'     \item d_score_PGB: Discriminator scores for selected samples
#'   }
#'
#' @references
#' Neunhoeffer, M., Wu, Z. S., & Dwork, C. (2021). Private Post-GAN Boosting.
#' International Conference on Learning Representations (ICLR).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Typically called via apply_post_gan_boosting(), but can be used directly:
#' result <- post_gan_boosting(
#'   d_score_fake = discriminator_scores_matrix,
#'   d_score_real = real_scores_vector,
#'   B = candidate_samples,
#'   real_N = 10000,
#'   steps = 200
#' )
#' boosted_samples <- result$PGB_sample
#' }
post_gan_boosting <-
  function(d_score_fake,
           d_score_real,
           B,
           real_N,
           steps = 400,
           N_generators = 200,
           uniform_init = TRUE,
           dp = FALSE,
           MW_epsilon = 0.1,
           weighted_average = FALSE,
           averaging_window = NULL) {

    # Initialize phi (the distribution over the generated examples B).
    # Either with a uniform distribution (as in the paper).
    # Or with a random distribution.

    if (uniform_init) {
      # Initialize phi to be the uniform distribution over B
      phi <- rep(1 / nrow(B), nrow(B))
    } else {
      # Initialize phi to a random distribution over B
      phi_unnormalized <- stats::runif(nrow(B))
      # Normalize to get a valid probability distribution
      phi <- phi_unnormalized / sum(phi_unnormalized)
    }

    # Set epsilon0 the privacy parameter for each update step
    epsilon0 <- MW_epsilon / steps

    # Initialize matrices to hold the results to calculate the mixture distribution.
    # phi_matrix stores all distributions of phi from 1 to T (steps)
    phi_matrix <- matrix(NA, nrow = steps, ncol = nrow(B))

    # best_D_matrix stores the best response after each step
    best_D_matrix <-
      matrix(NA, nrow = N_generators, ncol = steps)

    for (step_i in 1:steps) {
      # Set learning rate (eta in the paper)
      learning_rate <- 1 / sqrt(step_i)

      # Calculate the payoff U
      U_fake <- d_score_fake %*% phi
      # U as defined in Part 3.1
      U <- -(((1 - d_score_real) + U_fake))

      # Selecting a discriminator using the payoff U as the quality score

      if (dp == T) {
        # With dp the exponential mechanism is applied to choose D
        # The best D is sampled with probability porportional to
        # exp(epsilon0 * U * nrow(X))/2
        # add a small constant for numerical stability
        exp_U <- exp((epsilon0 * U * real_N) / (2)) + exp(-500)

        sum_exp_U <- sum(exp_U)

        # Normalize to get valid probability distribution
        p_U <- exp_U / sum_exp_U

        # Sample the best discriminator. If multiple Discriminators have the same score,
        # pick the first one.
        best_D <-
          1:length(U) == which(U == sample(
            x = U,
            size = 1,
            prob = p_U
          ))[1]


      }
      if (dp == F) {
        # Without privacy, directly pick the Discriminator with the best quality score.
        best_D <- 1:length(U) == which.max(U)
      }

      # Store the results

      best_D_matrix[, step_i] <- best_D

      # Update phi using the picked Discriminator
      phi_update <-
        phi * exp(learning_rate * d_score_fake[best_D, ])

      # Normalize to get a valid probability distribution
      phi <-  phi_update / sum(phi_update)

      # Store updated phi
      phi_matrix[step_i, ] <- phi

      # Print progress
      cat("Step: ", step_i, "\n")

    }

    if (weighted_average) {
      PGB_sel <-
        sample(1:nrow(B),
               prob = as.numeric(t(phi_matrix) %*% sqrt(1:steps) / sum(t(phi_matrix) %*% sqrt(1:steps))),
               replace = T)
    } else {
      if (is.null(averaging_window)) {
        averaging_window <- steps
      }
      PGB_sel <-
        sample(1:nrow(B),
               prob = apply(phi_matrix[(nrow(phi_matrix) - averaging_window + 1):nrow(phi_matrix), , drop = F], 2, mean, na.rm = T),
               replace = T)
    }

    # Select each sample at most once
    PGB_sel <- unique(PGB_sel)

    # Subset B to the PGB examples
    PGB_sample <- B[PGB_sel, ]

    # Calculate weighted discriminator scores D bar

    mix_D <- apply(best_D_matrix, 1, mean)
    d_bar_fake <- apply(sweep(d_score_fake, 1, mix_D, "*"), 2, sum)

    d_score_PGB <- d_bar_fake[PGB_sel]

    # Collect results in a list
    res <- list(PGB_sample = PGB_sample, d_score_PGB = d_score_PGB)

    # Return results
    return(res)
  }


#' @title Compute Discriminator Scores from Checkpointed Discriminators
#'
#' @description Evaluates generated samples using multiple discriminator checkpoints
#'   to create the discriminator score matrix required for post-GAN boosting.
#'
#' @param trained_gan A trained GAN object of class "trained_RGAN" with checkpoints
#' @param generated_samples A matrix of generated samples (N_samples x data_dim)
#' @param real_data A matrix of real data for computing real scores
#' @param batch_size Batch size for scoring (to manage memory). Defaults to 1000.
#' @param device Device for computation ("cpu", "cuda", "mps"). Defaults to trained_gan's device.
#'
#' @return A list with:
#'   \itemize{
#'     \item d_score_fake: Matrix of discriminator scores (N_discriminators x N_samples)
#'     \item d_score_real: Vector of mean discriminator scores on real data (length N_discriminators)
#'     \item epochs: Vector of epoch numbers corresponding to each discriminator
#'   }
#' @export
compute_discriminator_scores <- function(trained_gan,
                                          generated_samples,
                                          real_data,
                                          batch_size = 1000,
                                          device = NULL) {

  if (!inherits(trained_gan, "trained_RGAN")) {
    stop("trained_gan must be an object of class 'trained_RGAN'")
  }

  if (is.null(trained_gan$checkpoints) || length(trained_gan$checkpoints$epochs) == 0) {
    stop("trained_gan does not contain any checkpoints. Train with checkpoint_epochs parameter.")
  }

  # Set device
  if (is.null(device)) {
    device <- trained_gan$settings$device
  }

  # Convert inputs to tensors if needed
  if (!inherits(generated_samples, "torch_tensor")) {
    generated_samples <- torch::torch_tensor(generated_samples)$to(device = device)
  }
  if (!inherits(real_data, "torch_tensor")) {
    real_data <- torch::torch_tensor(real_data)$to(device = device)
  }

  n_samples <- nrow(generated_samples)
  n_discriminators <- length(trained_gan$checkpoints$epochs)

  # Initialize score matrices
  d_score_fake <- matrix(NA, nrow = n_discriminators, ncol = n_samples)
  d_score_real <- numeric(n_discriminators)

  # Create a discriminator template with same architecture
  value_function <- trained_gan$settings$value_function
  data_dim <- ncol(generated_samples)
  pac <- trained_gan$settings$pac
  d_input_dim <- data_dim * pac

  if (value_function != "original") {
    d_template <- Discriminator(data_dim = d_input_dim, dropout_rate = 0.5, sigmoid = FALSE)
  } else {
    d_template <- Discriminator(data_dim = d_input_dim, dropout_rate = 0.5, sigmoid = TRUE)
  }
  d_template <- d_template$to(device = device)

  cli::cli_progress_bar("Computing discriminator scores", total = n_discriminators)

  for (i in seq_along(trained_gan$checkpoints$discriminators)) {
    # Load discriminator state
    d_state <- trained_gan$checkpoints$discriminators[[i]]

    if (trained_gan$checkpoints$on_disk) {
      # Load from disk
      d_state <- torch::torch_load(d_state)
    }

    d_template$load_state_dict(d_state)
    d_template$eval()

    # Score fake samples in batches
    fake_scores <- c()
    for (start in seq(1, n_samples, by = batch_size)) {
      end <- min(start + batch_size - 1, n_samples)
      batch <- generated_samples[start:end, , drop = FALSE]

      # Handle PacGAN if needed
      if (pac > 1) {
        batch_n <- nrow(batch)
        usable_n <- batch_n - (batch_n %% pac)
        if (usable_n > 0) {
          batch <- batch[1:usable_n, , drop = FALSE]
          batch <- pack_samples(batch, pac)
        }
      }

      scores <- torch::with_no_grad({
        torch::as_array(d_template(batch)$cpu())
      })
      fake_scores <- c(fake_scores, as.vector(scores))
    }

    # For unpacked samples when pac > 1, we need to expand scores
    if (pac > 1) {
      # Each score represents pac samples, expand accordingly
      fake_scores <- rep(fake_scores, each = pac)[1:n_samples]
    }

    d_score_fake[i, ] <- fake_scores

    # Score real samples
    real_scores <- c()
    n_real <- nrow(real_data)
    for (start in seq(1, n_real, by = batch_size)) {
      end <- min(start + batch_size - 1, n_real)
      batch <- real_data[start:end, , drop = FALSE]

      if (pac > 1) {
        batch_n <- nrow(batch)
        usable_n <- batch_n - (batch_n %% pac)
        if (usable_n > 0) {
          batch <- batch[1:usable_n, , drop = FALSE]
          batch <- pack_samples(batch, pac)
        }
      }

      scores <- torch::with_no_grad({
        torch::as_array(d_template(batch)$cpu())
      })
      real_scores <- c(real_scores, as.vector(scores))
    }

    d_score_real[i] <- mean(real_scores)

    cli::cli_progress_update()
  }

  return(list(
    d_score_fake = d_score_fake,
    d_score_real = d_score_real,
    epochs = trained_gan$checkpoints$epochs
  ))
}


#' @title Apply Post-GAN Boosting to a Trained GAN
#'
#' @description High-level wrapper that orchestrates the full post-GAN boosting workflow:
#'   1. Generates candidate samples from checkpointed generators
#'   2. Computes discriminator scores using checkpointed discriminators
#'   3. Applies the post-GAN boosting algorithm to select high-quality samples
#'
#' @param trained_gan A trained GAN object of class "trained_RGAN" with checkpoints
#' @param real_data The original training data (matrix)
#' @param transformer Optional data_transformer for inverse transformation
#' @param n_candidates Number of candidate samples to generate per generator. Defaults to 1000.
#' @param steps Number of boosting steps. Defaults to 400.
#' @param dp Use differential privacy for discriminator selection. Defaults to FALSE.
#' @param MW_epsilon Privacy budget for multiplicative weights (only used if dp=TRUE). Defaults to 0.1.
#' @param weighted_average Use weighted averaging for final distribution. Defaults to FALSE.
#' @param averaging_window Window size for averaging phi distributions. Defaults to NULL (use all steps).
#' @param device Device for computation. Defaults to trained_gan's device.
#' @param seed Optional seed for reproducibility.
#'
#' @return A list with:
#'   \itemize{
#'     \item samples: Matrix of selected high-quality synthetic samples
#'     \item scores: Discriminator scores for selected samples
#'     \item n_unique: Number of unique samples selected
#'   }
#' @export
#'
#' @examples
#' \dontrun{
#' # Train a GAN with checkpoints
#' trained_gan <- gan_trainer(
#'   transformed_data,
#'   epochs = 100,
#'   checkpoint_epochs = 10  # Save every 10 epochs
#' )
#'
#' # Apply post-GAN boosting
#' boosted <- apply_post_gan_boosting(
#'   trained_gan,
#'   real_data = transformed_data,
#'   n_candidates = 5000,
#'   steps = 200
#' )
#'
#' # Use the boosted samples
#' high_quality_samples <- boosted$samples
#' }
apply_post_gan_boosting <- function(trained_gan,
                                     real_data,
                                     transformer = NULL,
                                     n_candidates = 1000,
                                     steps = 400,
                                     dp = FALSE,
                                     MW_epsilon = 0.1,
                                     weighted_average = FALSE,
                                     averaging_window = NULL,
                                     device = NULL,
                                     seed = NULL) {

  if (!inherits(trained_gan, "trained_RGAN")) {
    stop("trained_gan must be an object of class 'trained_RGAN'")
  }

  if (is.null(trained_gan$checkpoints) || length(trained_gan$checkpoints$epochs) == 0) {
    stop("trained_gan does not contain any checkpoints. Train with checkpoint_epochs parameter.")
  }

  # Set seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }

  # Set device
  if (is.null(device)) {
    device <- trained_gan$settings$device
  }

  # Convert real_data to tensor if needed
  if (!inherits(real_data, "torch_tensor")) {
    real_data_tensor <- torch::torch_tensor(real_data)$to(device = device)
  } else {
    real_data_tensor <- real_data$to(device = device)
  }
  real_N <- nrow(real_data_tensor)

  n_generators <- length(trained_gan$checkpoints$generators)
  noise_dim <- trained_gan$settings$noise_dim
  sample_noise <- trained_gan$settings$sample_noise

  # Generate candidate samples from all generator checkpoints
  cli::cli_alert_info(sprintf("Generating %d candidates from %d generator checkpoints",
                               n_candidates * n_generators, n_generators))

  all_samples <- list()

  # Create generator template
  data_dim <- ncol(real_data_tensor)
  if (!is.null(trained_gan$settings$output_info)) {
    g_template <- TabularGenerator(
      noise_dim = noise_dim,
      output_info = trained_gan$settings$output_info,
      hidden_units = trained_gan$settings$generator_hidden_units,
      dropout_rate = 0.5,
      tau = trained_gan$settings$gumbel_tau,
      normalization = trained_gan$settings$generator_normalization,
      activation = trained_gan$settings$generator_activation,
      init_method = trained_gan$settings$generator_init,
      residual = trained_gan$settings$generator_residual
    )
  } else {
    g_template <- Generator(noise_dim = noise_dim, data_dim = data_dim, dropout_rate = 0.5)
  }
  g_template <- g_template$to(device = device)

  for (i in seq_along(trained_gan$checkpoints$generators)) {
    g_state <- trained_gan$checkpoints$generators[[i]]

    if (trained_gan$checkpoints$on_disk) {
      g_state <- torch::torch_load(g_state)
    }

    g_template$load_state_dict(g_state)
    g_template$eval()

    # Generate samples
    z <- sample_noise(c(n_candidates, noise_dim))$to(device = device)
    samples <- torch::with_no_grad({
      torch::as_array(g_template(z)$cpu())
    })
    all_samples[[i]] <- samples
  }

  # Combine all generated samples into matrix B
  B <- do.call(rbind, all_samples)

  cli::cli_alert_info(sprintf("Computing discriminator scores for %d samples", nrow(B)))

  # Compute discriminator scores
  scores <- compute_discriminator_scores(
    trained_gan = trained_gan,
    generated_samples = B,
    real_data = real_data,
    device = device
  )

  cli::cli_alert_info(sprintf("Running post-GAN boosting with %d steps", steps))

  # Apply post-GAN boosting
  result <- post_gan_boosting(
    d_score_fake = scores$d_score_fake,
    d_score_real = scores$d_score_real,
    B = B,
    real_N = real_N,
    steps = steps,
    N_generators = n_generators,
    uniform_init = TRUE,
    dp = dp,
    MW_epsilon = MW_epsilon,
    weighted_average = weighted_average,
    averaging_window = averaging_window
  )

  # Inverse transform if transformer provided
  if (!is.null(transformer)) {
    result$PGB_sample <- transformer$inverse_transform(result$PGB_sample)
  }

  cli::cli_alert_success(sprintf("Post-GAN boosting selected %d unique samples", nrow(result$PGB_sample)))

  return(list(
    samples = result$PGB_sample,
    scores = result$d_score_PGB,
    n_unique = nrow(result$PGB_sample)
  ))
}
