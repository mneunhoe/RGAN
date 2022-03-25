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
