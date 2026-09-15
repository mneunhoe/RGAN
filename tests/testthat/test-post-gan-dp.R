# Differential-privacy behaviour of post_gan_boosting().
#
# The first three tests are regressions for defects present in 0.2.0 and fail
# against that version:
#   - tied discriminators were given selection probability zero after the first
#   - the resulting output distribution was not the exponential mechanism
#   - the weights underflowed and collapsed to the uniform distribution

# The selection probabilities the exponential mechanism is supposed to realise.
exp_mech_probs <- function(U, epsilon0, real_N) {
  log_w <- (epsilon0 * real_N / 2) * U
  p <- exp(log_w - max(log_w))
  p / sum(p)
}

# Rows constant across candidates, so with a uniform phi the payoff of
# discriminator j is d_score_real[j] + 1 - row_value[j].
fixture <- function(row_values, real_value = 0.9, n_b = 4) {
  list(
    d_fake = matrix(rep(row_values, each = n_b), nrow = length(row_values),
                    byrow = TRUE),
    d_real = rep(real_value, length(row_values)),
    B = matrix(seq_len(n_b * 2) / (n_b * 2), nrow = n_b),
    U = real_value + 1 - row_values
  )
}

# Index chosen in a single boosting round.
pick_once <- function(fx, real_N, MW_epsilon) {
  res <- post_gan_boosting(
    d_score_fake = fx$d_fake, d_score_real = fx$d_real, B = fx$B,
    real_N = real_N, steps = 1, N_generators = nrow(fx$d_fake),
    dp = TRUE, MW_epsilon = MW_epsilon
  )
  which.max(res$D_bar)
}


test_that("tied discriminators are selected with equal probability", {
  fx <- fixture(c(0.20, 0.20, 0.50, 0.80))   # discriminators 1 and 2 tie exactly
  expected <- exp_mech_probs(fx$U, epsilon0 = 1, real_N = 10)

  set.seed(20240611)
  invisible(capture.output(
    picks <- replicate(3000, pick_once(fx, real_N = 10, MW_epsilon = 1))
  ))
  observed <- as.numeric(table(factor(picks, levels = seq_along(fx$U)))) / 3000

  # 0.2.0 put all of the tied mass on index 1: observed was ~0.88 / 0.00.
  expect_gt(observed[2], 0.30)
  expect_lt(max(abs(observed - expected)), 0.04)
})


test_that("selection probabilities respect the per-round privacy budget", {
  # Two datasets one record apart: every mean discriminator score moves by at
  # most 1 / real_N. On the first, discriminators 1 and 2 tie exactly; on the
  # second the tie is broken. A value-based tie-break gives index 2 probability
  # zero on the first and ~0.44 on the second, an unbounded ratio.
  real_N <- 10
  epsilon0 <- 1
  fx <- fixture(c(0.20, 0.20, 0.50, 0.80))
  gx <- fixture(c(0.20, 0.20, 0.50, 0.80))
  gx$d_real <- fx$d_real + c(1, -1, 1, -1) / real_N
  gx$U <- gx$d_real + 1 - c(0.20, 0.20, 0.50, 0.80)

  draws <- function(fixt, n) {
    invisible(capture.output(
      picks <- replicate(n, pick_once(fixt, real_N = real_N, MW_epsilon = epsilon0))
    ))
    as.numeric(table(factor(picks, levels = seq_along(fixt$U)))) / n
  }

  set.seed(4242)
  p1 <- draws(fx, 2000)
  p2 <- draws(gx, 2000)

  # Each realised distribution is the exponential mechanism ...
  expect_lt(max(abs(p1 - exp_mech_probs(fx$U, epsilon0, real_N))), 0.05)
  expect_lt(max(abs(p2 - exp_mech_probs(gx$U, epsilon0, real_N))), 0.05)

  # ... and every outcome stays reachable on both, within exp(epsilon0).
  expect_true(all(p1 > 0) && all(p2 > 0))
  expect_lt(max(abs(log(p1 / p2))), epsilon0 + 0.2)

  # The calibration itself, free of Monte-Carlo noise.
  expect_lte(max(abs(log(exp_mech_probs(fx$U, epsilon0, real_N) /
                         exp_mech_probs(gx$U, epsilon0, real_N)))),
             epsilon0 + 1e-9)
})


test_that("the mechanism does not collapse to uniform for large real_N", {
  # epsilon0 * real_N / 2 = 5e5 made every weight underflow in 0.2.0, leaving
  # exp(-500) as the only surviving term and p_U exactly uniform.
  fx <- fixture(c(0.20, 0.50, 0.80))
  set.seed(11)
  invisible(capture.output(
    picks <- replicate(200, pick_once(fx, real_N = 1e6, MW_epsilon = 1))
  ))
  expect_gt(mean(picks == which.max(fx$U)), 0.99)
})


test_that("dp = TRUE rejects discriminator scores outside [0, 1]", {
  fx <- fixture(c(0.20, 0.50, 0.80))
  logits <- fx$d_fake * 10 - 5           # an unbounded critic, e.g. WGAN

  expect_error(
    post_gan_boosting(d_score_fake = logits, d_score_real = fx$d_real, B = fx$B,
                      real_N = 100, steps = 2, N_generators = 3, dp = TRUE),
    "scores in \\[0, 1\\]"
  )
  # The same scores are fine without the privacy claim.
  expect_no_error(invisible(capture.output(
    post_gan_boosting(d_score_fake = logits, d_score_real = fx$d_real, B = fx$B,
                      real_N = 100, steps = 2, N_generators = 3, dp = FALSE)
  )))
})


test_that("inconsistent discriminator counts are rejected", {
  fx <- fixture(c(0.20, 0.50, 0.80))
  expect_error(
    post_gan_boosting(d_score_fake = fx$d_fake, d_score_real = fx$d_real,
                      B = fx$B, real_N = 100, steps = 2, N_generators = 6),
    "must equal the number of discriminators"
  )
  expect_error(
    post_gan_boosting(d_score_fake = fx$d_fake, d_score_real = fx$d_real[1:2],
                      B = fx$B, real_N = 100, steps = 2, N_generators = 3),
    "one score per discriminator"
  )
})


test_that("sensitivity is calibrated to scored units, not records", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(transformed_data, epochs = 4, batch_size = 20,
                             checkpoint_epochs = 2, pac = 2, seed = 123)

  samples <- matrix(stats::rnorm(20 * ncol(transformed_data)), nrow = 20)
  scores <- suppressMessages(compute_discriminator_scores(
    trained_gan = trained_gan, generated_samples = samples,
    real_data = transformed_data
  ))

  # With pac = 2 each score covers two records, so d_score_real averages n / 2
  # values and the exponential mechanism must be calibrated to that count.
  expect_equal(scores$n_real_units, nrow(transformed_data) %/% 2)
  expect_lt(scores$n_real_units, nrow(transformed_data))
})


test_that("score_transform = 'sigmoid' bounds unbounded critic outputs", {
  expect_equal(logistic(0), 0.5)
  expect_true(all(logistic(c(-40, -1, 0, 1, 40)) >= 0))
  expect_true(all(logistic(c(-40, -1, 0, 1, 40)) <= 1))
})
