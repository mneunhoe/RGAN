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
#'   With `dp = TRUE` it must be a public count under add/remove-one adjacency:
#'   either a number fixed independently of the data, with `d_score_real`
#'   computed as `min(1, sum / real_N)`, or the number of records when that is
#'   treated as public.
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
#' @param secure_rng Only used when `dp = TRUE`. Draw the randomness of the
#'   exponential-mechanism selection from OpenDP's cryptographically secure
#'   generator, as `dp_gan_trainer(secure_rng = TRUE)` does for training.
#'   Defaults to TRUE. With FALSE the selection uses R's Mersenne Twister, which
#'   is predictable: anyone who knows or guesses the seed can recompute every
#'   selection, so use FALSE for testing only. The weights are still computed
#'   in floating point, with the usual caveats for the exponential mechanism
#'   (Ilvento, 2020).
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
           averaging_window = NULL,
           secure_rng = TRUE) {

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
      if (isTRUE(secure_rng) && !requireNamespace("opendp", quietly = TRUE)) {
        stop("Package 'opendp' is required for dp = TRUE with secure_rng = TRUE. ",
             "Install it with install.packages('opendp'), or set ",
             "secure_rng = FALSE for testing only.", call. = FALSE)
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

        best_idx <- if (isTRUE(secure_rng)) {
          # Inverse-CDF draw from a cryptographically secure uniform, so the
          # selection is not a function of R's (seedable) RNG state.
          min(n_disc, findInterval(secure_uniform(1L), cumsum(p_U)) + 1L)
        } else {
          sample.int(n_disc, size = 1L, prob = p_U)
        }
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
#'     \item d_score_real_sum: The corresponding sums, for normalising by a
#'       public count instead of the number of records
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
  n_real <- nrow(real_data)

  # Settings missing from older or DP-trained models default to the original
  # loss on unpacked samples, which is what dp_gan_trainer() trains.
  pac <- trained_gan$settings$pac
  if (is.null(pac)) pac <- 1L
  if (n_real < pac) {
    stop(sprintf("real_data has %d rows, fewer than pac = %d.", n_real, pac),
         call. = FALSE)
  }

  # Initialize score matrices
  d_score_fake <- matrix(NA_real_, nrow = n_discriminators, ncol = n_samples)
  d_score_real <- numeric(n_discriminators)
  d_score_real_sum <- numeric(n_discriminators)
  # Number of independently scored units behind each entry of d_score_real.
  # With PacGAN each score covers `pac` records, so this is n %/% pac, not n.
  n_real_units <- n_real %/% pac

  # Score with the trained model's own discriminator, loading each checkpoint
  # into it, so that custom architectures and every trainer are supported.
  # Its weights, mode and device are restored on exit.
  # A conditionally trained discriminator also takes the condition of each
  # sample; score with the all-unobserved condition, the mode condition
  # dropout trains. A discriminator conditioned on the instance-noise level
  # takes sigma as one extra input after packing; score at sigma = 0.
  cond_dim <- trained_gan$settings$cond_dim
  if (is.null(cond_dim)) cond_dim <- 0L
  sigma_column <- isTRUE(trained_gan$settings$instance_noise$condition_d)

  d_template <- trained_gan$discriminator
  restore_d <- borrow_module(d_template, device)
  on.exit(restore_d(), add = TRUE)

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

    # Every candidate gets the score of the pack it is in. The candidates are
    # packed once, consecutively, over the whole set; the last pack is padded
    # by wrapping around to the first candidates.
    d_score_fake[i, ] <- squash(score_rows(d_template, generated_samples, pac,
                                           batch_size, pad = TRUE,
                                           cond_dim = cond_dim,
                                           sigma_column = sigma_column))

    # Real data: complete packs only, so no record is counted twice.
    real_scores <- squash(score_rows(d_template, real_data, pac, batch_size,
                                     pad = FALSE, cond_dim = cond_dim,
                                     sigma_column = sigma_column))
    d_score_real[i] <- mean(real_scores)
    d_score_real_sum[i] <- sum(real_scores)

    cli::cli_progress_update()
  }

  return(list(
    d_score_fake = d_score_fake,
    d_score_real = d_score_real,
    d_score_real_sum = d_score_real_sum,
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
#' @param seed Optional seed for reproducibility of the candidate generation.
#'   With `dp = TRUE` and `secure_rng = TRUE` the private selection does not
#'   use R's RNG, so the seed does not make it reproducible.
#' @param secure_rng Passed to `post_gan_boosting()`: draw the private
#'   selection from a cryptographically secure source. Only used with
#'   `dp = TRUE`. Defaults to TRUE.
#'
#' @details
#' With `dp = TRUE` each real record is scored on its own (`pac = 1`), so the
#' real-data payoff has sensitivity `1 / N` under add/remove-one adjacency,
#' matching DP-SGD training. For a `dp_gan_trainer()` model trained with an
#' explicit `sampling_rate`, N is the public count
#' `expected_batch_size / sampling_rate` and the payoff is
#' `min(1, sum of scores / N)`, so the number of records stays protected.
#' Otherwise N is the number of records, which is then treated as public.
#'
#' `dp = TRUE` is refused for `pac > 1`: consecutive records are packed
#' together, inserting or deleting one shifts every later pack, and the bound
#' would hold only under replace-one adjacency.
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
                                     seed = NULL,
                                     secure_rng = TRUE) {

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

  pac <- trained_gan$settings$pac
  if (is.null(pac)) pac <- 1L
  if (isTRUE(dp) && pac > 1) {
    stop(paste0(
      "dp = TRUE is not supported with pac = ", pac, ". Real records are ",
      "scored in consecutive packs, so inserting or deleting one record ",
      "changes every later pack: the 1 / n_real_units sensitivity the ",
      "selection is calibrated to holds under replace-one adjacency only, not ",
      "the add/remove-one adjacency of dp_gan_trainer(). Use a model trained ",
      "with pac = 1 (every dp_gan_trainer() model is), or set dp = FALSE."
    ), call. = FALSE)
  }
  if (isTRUE(dp) && !isTRUE(secure_rng) && !is.null(seed)) {
    cli::cli_alert_warning(paste0(
      "dp = TRUE with secure_rng = FALSE and a seed: the private selection is ",
      "a deterministic function of the data and the seed, so it is not ",
      "differentially private. Use secure_rng = TRUE."
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

  # Generate with the trained model's own generator, loading each checkpoint
  # into it; its weights, mode and device are restored on exit.
  g_template <- trained_gan$generator
  restore_g <- borrow_module(g_template, device)
  on.exit(restore_g(), add = TRUE)

  for (i in seq_along(trained_gan$checkpoints$generators)) {
    g_state <- trained_gan$checkpoints$generators[[i]]

    if (trained_gan$checkpoints$on_disk) {
      g_state <- torch::torch_load(g_state)
    }

    g_template$load_state_dict(g_state)
    g_template$eval()

    # Generate samples
    z <- sample_noise(c(n_candidates, noise_dim))$to(device = device)
    # Conditionally trained generators are sampled with the all-unobserved
    # condition (the mode cond_dropout trains).
    cd <- trained_gan$settings$cond_dim
    if (!is.null(cd) && cd > 0) {
      z <- torch::torch_cat(
        list(z, torch::torch_zeros(c(n_candidates, cd),
                                   device = z$device)$to(dtype = z$dtype)),
        dim = 2)
    }
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

  # Sensitivity of the real-data payoff. By default it is the mean over the
  # n scored records, with sensitivity 1 / n -- which treats n as public. A DP
  # model trained with an explicit sampling_rate protects n, so normalise by
  # the public count N = batch_size / sampling_rate instead: min(1, sum / N)
  # has sensitivity 1 / N under add/remove-one adjacency, and no part of the
  # selection depends on n.
  real_N <- scores$n_real_units
  d_score_real <- scores$d_score_real
  privacy <- trained_gan$privacy
  if (isTRUE(dp) && !is.null(privacy) && identical(privacy$n_public, FALSE)) {
    real_N <- public_record_count(privacy)
    d_score_real <- pmin(1, scores$d_score_real_sum / real_N)
    cli::cli_alert_info(sprintf(paste0(
      "Real-data scores normalised by the public count %s ",
      "(expected_batch_size / sampling_rate), so n stays protected."),
      format(real_N)
    ))
  }

  cli::cli_alert_info(sprintf("Running post-GAN boosting with %d steps", steps))

  # Apply post-GAN boosting
  result <- post_gan_boosting(
    d_score_fake = scores$d_score_fake,
    d_score_real = d_score_real,
    B = B,
    real_N = real_N,
    steps = steps,
    N_generators = nrow(scores$d_score_fake),
    uniform_init = TRUE,
    dp = dp,
    MW_epsilon = MW_epsilon,
    weighted_average = weighted_average,
    averaging_window = averaging_window,
    secure_rng = secure_rng
  )

  # Inverse transform if transformer provided
  if (!is.null(transformer)) {
    warn_if_data_fitted_transformer(trained_gan, transformer)
    result$PGB_sample <- transformer$inverse_transform(result$PGB_sample)
  }

  cli::cli_alert_success(sprintf("Post-GAN boosting selected %d unique samples", nrow(result$PGB_sample)))

  return(list(
    samples = result$PGB_sample,
    scores = result$d_score_PGB,
    n_unique = nrow(result$PGB_sample)
  ))
}


#' @title Score Rows with a (Packed) Discriminator
#'
#' @description Scores the rows of `x` in batches. With `pac > 1` the rows are
#'   packed consecutively over the whole of `x` -- not per batch, which would
#'   drop rows and shift later scores onto the wrong samples -- and each row
#'   receives the score of its pack.
#'
#' @param net The discriminator.
#' @param x A torch tensor with one row per sample.
#' @param pac Pack size.
#' @param batch_size Approximate number of rows per forward pass; rounded down
#'   to a multiple of `pac`.
#' @param pad If TRUE, pad the last incomplete pack by wrapping around to the
#'   first rows and return one score per row of `x`. If FALSE, drop the
#'   incomplete pack and return one score per complete pack.
#' @param cond_dim Width of the all-unobserved condition appended to every
#'   sample before packing, for conditionally trained discriminators.
#' @param sigma_column Append an instance-noise level of 0 after packing, for
#'   discriminators trained with `instance_noise$condition_d`.
#'
#' @return A numeric vector of scores.
#' @keywords internal
score_rows <- function(net, x, pac, batch_size, pad, cond_dim = 0L,
                       sigma_column = FALSE) {
  n <- nrow(x)
  if (pac > 1) {
    if (pad) {
      idx <- c(seq_len(n), rep_len(seq_len(n), (-n) %% pac))
    } else {
      idx <- seq_len(n - n %% pac)
    }
  } else {
    idx <- seq_len(n)
  }
  rows_per_batch <- max(pac, batch_size - batch_size %% pac)

  scores <- c()
  for (start in seq(1, length(idx), by = rows_per_batch)) {
    batch_idx <- idx[start:min(start + rows_per_batch - 1, length(idx))]
    batch <- x[batch_idx, , drop = FALSE]
    if (cond_dim > 0) {
      batch <- torch::torch_cat(list(
        batch, torch::torch_zeros(c(nrow(batch), cond_dim),
                                  dtype = batch$dtype, device = batch$device)
      ), dim = 2)
    }
    if (pac > 1) batch <- pack_samples(batch, pac)
    if (sigma_column) batch <- append_sigma_column(batch, 0)
    s <- torch::with_no_grad(torch::as_array(net(batch)$cpu()))
    scores <- c(scores, as.vector(s))
  }

  if (pac > 1 && pad) {
    scores <- rep(scores, each = pac)[seq_len(n)]
  }
  scores
}

#' @title Temporarily Use a Trained Module
#'
#' @description Records a module's weights, training mode and device, moves it
#'   to `device`, and returns a function that restores all three. Used to load
#'   checkpoints into a trained model's own networks without changing the
#'   model.
#'
#' @param net A torch::nn_module.
#' @param device Device to move the module to.
#' @return A function of no arguments that restores the module.
#' @keywords internal
borrow_module <- function(net, device) {
  params <- net$parameters
  original_device <- if (length(params) > 0) params[[1]]$device else NULL
  original_state <- lapply(net$state_dict(), function(x) x$clone())
  was_training <- isTRUE(net$training)
  net$to(device = device)
  function() {
    net$load_state_dict(original_state)
    if (!is.null(original_device)) net$to(device = original_device)
    if (was_training) net$train() else net$eval()
    invisible(NULL)
  }
}


#' @title Public Record Count of a DP Model
#'
#' @description The count `expected_batch_size / sampling_rate` implied by the
#'   public mechanism parameters of a `dp_gan_trainer()` model.
#'
#' @param privacy The `privacy` element of a trained model.
#' @return A positive number.
#' @keywords internal
public_record_count <- function(privacy) {
  L <- privacy$expected_batch_size
  if (is.null(L)) {
    stop("This model does not record its expected batch size; retrain it ",
         "with the current version of dp_gan_trainer().", call. = FALSE)
  }
  L / privacy$sampling_rate
}
