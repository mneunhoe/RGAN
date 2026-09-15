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
#' @param real_N Number of *independently scored* records behind `d_score_real`.
#'   The exponential mechanism is calibrated to a sensitivity of `1 / real_N`, so
#'   this must be the number of scores that were averaged, not the number of rows
#'   in the training data. With PacGAN packing (`pac > 1`) each score covers `pac`
#'   records and the correct value is `n / pac`; `compute_discriminator_scores()`
#'   reports it as `n_real_units`.
#' @param steps Number of boosting iterations. Defaults to 400.
#' @param N_generators Number of discriminator checkpoints used. Must equal
#'   `nrow(d_score_fake)`. Defaults to 200.
#' @param uniform_init Initialize phi with uniform distribution. Defaults to TRUE.
#' @param dp Use differential privacy for discriminator selection via exponential
#'   mechanism. Defaults to FALSE. Requires all discriminator scores to lie in
#'   `[0, 1]`; unbounded critics (WGAN, KLWGAN) are rejected with an error, since
#'   the `1 / real_N` sensitivity the mechanism is calibrated to does not hold for
#'   them. Re-score with `score_transform = "sigmoid"` to use them.
#' @param MW_epsilon Total privacy budget for multiplicative weights (only if
#'   dp=TRUE). Split as `MW_epsilon / steps` per round, so basic composition over
#'   the `steps` rounds spends `MW_epsilon` in total. Defaults to 0.1.
#' @param weighted_average Use weighted averaging (weights proportional to sqrt(step)).
#'   Defaults to FALSE.
#' @param averaging_window Number of final steps to average over. Defaults to NULL (all steps).
#'
#' @return A list with:
#'   \itemize{
#'     \item PGB_sample: Matrix of selected high-quality samples
#'     \item d_score_PGB: Discriminator scores for selected samples
#'     \item D_bar: The mixed discriminator strategy, i.e. the fraction of the
#'       `steps` rounds in which each discriminator was selected (length
#'       `N_generators`).
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

    # ---- Input validation -------------------------------------------------

    d_score_fake <- as.matrix(d_score_fake)
    d_score_real <- as.vector(d_score_real)
    n_disc <- nrow(d_score_fake)

    if (length(d_score_real) != n_disc) {
      stop("d_score_real must hold one score per discriminator: ",
           length(d_score_real), " scores for ", n_disc,
           " rows of d_score_fake.", call. = FALSE)
    }
    if (ncol(d_score_fake) != nrow(B)) {
      stop("d_score_fake must hold one column per candidate sample: ",
           ncol(d_score_fake), " columns for ", nrow(B), " rows of B.",
           call. = FALSE)
    }
    if (N_generators != n_disc) {
      # A mismatch silently recycles best_D into best_D_matrix when one count
      # divides the other, corrupting the averaged discriminator D-bar.
      stop("N_generators (", N_generators, ") must equal the number of ",
           "discriminators, nrow(d_score_fake) (", n_disc, ").", call. = FALSE)
    }

    if (isTRUE(dp)) {
      # The exponential mechanism below is calibrated to a sensitivity of
      # 1 / real_N, which holds only while the discriminator scores lie in
      # [0, 1]. Unbounded critics (WGAN, KLWGAN) break that calibration.
      rng <- range(c(d_score_fake, d_score_real), na.rm = TRUE)
      if (!all(is.finite(rng)) || rng[1] < -1e-6 || rng[2] > 1 + 1e-6) {
        stop("dp = TRUE requires discriminator scores in [0, 1]; observed ",
             "range [", format(rng[1]), ", ", format(rng[2]), "]. Non-sigmoid ",
             "critics (WGAN, KLWGAN) are unbounded, so the 1 / real_N ",
             "sensitivity the exponential mechanism is calibrated to does not ",
             "hold. Re-score with score_transform = \"sigmoid\", or set ",
             "dp = FALSE.", call. = FALSE)
      }
      if (!is.numeric(real_N) || length(real_N) != 1L ||
          !is.finite(real_N) || real_N < 1) {
        stop("dp = TRUE requires real_N to be the number of independently ",
             "scored records backing d_score_real (a positive number).",
             call. = FALSE)
      }
    }

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

    # Per-round privacy parameter. Basic composition over the `steps` rounds
    # spends steps * epsilon0 = MW_epsilon in total.
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

      # Payoff of each discriminator against the current phi, as in Section 3.1
      # of the paper:  U(phi, D_j) = E_real[D_j] + (1 - E_phi[D_j]).
      U_fake <- as.vector(d_score_fake %*% phi)
      U <- d_score_real + (1 - U_fake)

      # Selecting a discriminator using the payoff U as the quality score

      if (isTRUE(dp)) {
        # Exponential mechanism over discriminator INDICES.
        #
        # Sampling a score *value* and mapping it back with which(U == v)[1]
        # gives every tied discriminator after the first a selection
        # probability of zero. When a tie is present on one dataset and broken
        # on an adjacent one, the ratio of output probabilities is unbounded
        # and the eps-DP guarantee fails. Sampling the index directly is what
        # the exponential mechanism is.
        #
        # The log-sum-exp shift keeps the weights finite: without it the
        # exponent epsilon0 * real_N * U / 2 under- or overflows for large
        # real_N and p_U silently collapses to the uniform distribution.
        log_w <- (epsilon0 * real_N / 2) * U
        p_U <- exp(log_w - max(log_w))
        p_U <- p_U / sum(p_U)

        if (!all(is.finite(p_U)) || abs(sum(p_U) - 1) > 1e-8) {
          stop("Exponential mechanism weights are not a valid distribution. ",
               "Check d_score_fake and d_score_real for NA or non-finite ",
               "values.", call. = FALSE)
        }

        best_idx <- sample.int(n_disc, size = 1L, prob = p_U)
      } else {
        # Without privacy, play an exact best response.
        best_idx <- which.max(U)
      }

      best_D <- seq_len(n_disc) == best_idx

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

    # Collect results in a list. D_bar is the paper's mixed discriminator
    # strategy: the fraction of rounds in which each discriminator was selected.
    res <- list(PGB_sample = PGB_sample,
                d_score_PGB = d_score_PGB,
                D_bar = mix_D)

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
#' @param score_transform Either "none" (default) to return the discriminator
#'   outputs as they are, or "sigmoid" to squash them into `[0, 1]`. The squash is
#'   monotone, so it preserves the ranking of samples, and it is what makes the
#'   scores of an unbounded critic (WGAN, KLWGAN) usable with `dp = TRUE`.
#'
#' @return A list with:
#'   \itemize{
#'     \item d_score_fake: Matrix of discriminator scores (N_discriminators x N_samples)
#'     \item d_score_real: Vector of mean discriminator scores on real data (length N_discriminators)
#'     \item n_real_units: Number of independently scored units averaged into each
#'       entry of d_score_real. Equals the number of real records unless
#'       `pac > 1`, in which case it is `n / pac`. Pass this as `real_N` to
#'       `post_gan_boosting()`.
#'     \item epochs: Vector of epoch numbers corresponding to each discriminator
#'   }
#' @export
compute_discriminator_scores <- function(trained_gan,
                                          generated_samples,
                                          real_data,
                                          batch_size = 1000,
                                          device = NULL,
                                          score_transform = c("none", "sigmoid")) {

  score_transform <- match.arg(score_transform)
  # A monotone squash into [0, 1] makes unbounded critic outputs usable by the
  # differentially private selection, whose sensitivity analysis needs bounded
  # scores. It has to be asked for: silently changing the score scale would
  # change what the returned numbers mean.
  squash <- if (score_transform == "sigmoid") logistic else identity

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
  # Number of independently scored units behind each entry of d_score_real.
  # With PacGAN each score covers `pac` records, so this is n / pac, not n.
  n_real_units <- NA_integer_

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
      fake_scores <- c(fake_scores, squash(as.vector(scores)))
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
      real_scores <- c(real_scores, squash(as.vector(scores)))
    }

    d_score_real[i] <- mean(real_scores)
    if (i == 1L) n_real_units <- length(real_scores)

    cli::cli_progress_update()
  }

  return(list(
    d_score_fake = d_score_fake,
    d_score_real = d_score_real,
    n_real_units = n_real_units,
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
#' @param dp Use differential privacy for discriminator selection. Defaults to
#'   FALSE. This privatises the *selection* of discriminators only: if
#'   `trained_gan` came from `gan_trainer()` rather than `dp_gan_trainer()`, the
#'   checkpoints already encode the training data and the boosted samples are not
#'   differentially private end to end. Total budget is the training epsilon plus
#'   `MW_epsilon`.
#' @param MW_epsilon Privacy budget for multiplicative weights (only used if dp=TRUE). Defaults to 0.1.
#' @param weighted_average Use weighted averaging for final distribution. Defaults to FALSE.
#' @param averaging_window Window size for averaging phi distributions. Defaults to NULL (use all steps).
#' @param score_transform Either "none" (default) or "sigmoid"; see
#'   `compute_discriminator_scores()`. Use "sigmoid" with `dp = TRUE` when the GAN
#'   was trained with a value function other than "original", whose discriminator
#'   has no sigmoid output layer.
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
                                     score_transform = c("none", "sigmoid"),
                                     device = NULL,
                                     seed = NULL) {

  score_transform <- match.arg(score_transform)

  if (!inherits(trained_gan, "trained_RGAN")) {
    stop("trained_gan must be an object of class 'trained_RGAN'")
  }

  if (is.null(trained_gan$checkpoints) || length(trained_gan$checkpoints$epochs) == 0) {
    stop("trained_gan does not contain any checkpoints. Train with checkpoint_epochs parameter.")
  }

  # dp = TRUE privatises the *selection* of discriminators only. Checkpoints
  # from non-private training already encode the data, so the boosted samples
  # are not differentially private end to end.
  if (isTRUE(dp) && is.null(trained_gan$privacy)) {
    cli::cli_alert_warning(paste0(
      "dp = TRUE privatises discriminator selection only. These checkpoints ",
      "carry no privacy record, so the boosted samples are NOT differentially ",
      "private end to end. Train the GAN with dp_gan_trainer()."
    ))
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
    device = device,
    score_transform = score_transform
  )

  # Sensitivity of d_score_real is 1 / (number of independently scored units).
  # With PacGAN each score covers `pac` records, so that count is n / pac and
  # using nrow(real_data) would understate the privacy loss by a factor of pac.
  real_N <- scores$n_real_units
  if (isTRUE(dp) && real_N != nrow(real_data_tensor)) {
    cli::cli_alert_info(paste0(
      "Sensitivity calibrated to ", real_N, " independently scored units ",
      "(pac = ", trained_gan$settings$pac, ", ",
      nrow(real_data_tensor), " records)."
    ))
  }

  cli::cli_alert_info(sprintf("Running post-GAN boosting with %d steps", steps))

  # Apply post-GAN boosting
  result <- post_gan_boosting(
    d_score_fake = scores$d_score_fake,
    d_score_real = scores$d_score_real,
    B = B,
    real_N = real_N,
    steps = steps,
    N_generators = nrow(scores$d_score_fake),
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
