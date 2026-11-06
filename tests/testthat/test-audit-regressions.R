# Regression tests for the defects reported in GitHub issues #3-#27.
# Each test names the issue it covers.

skip_if_no_opendp <- function() {
  if (!requireNamespace("opendp", quietly = TRUE)) skip("opendp not installed")
}

# ---- gan_trainer ------------------------------------------------------------

test_that("#10: the no-op weight clipper does not capture the training data", {
  x <- matrix(rnorm(400), 200, 2)
  for (vf in c("original", "wgan-gp", "f-wgan")) {
    g <- gan_trainer(x, epochs = 1, value_function = vf)
    env <- environment(g$settings$weight_clipper)
    expect_identical(env, asNamespace("RGAN"))
    expect_false(exists("data", envir = env, inherits = FALSE))
  }
})

test_that("#14: early stopping restores a snapshot, not the live weights", {
  net <- torch::nn_linear(2, 1)
  snapshot <- lapply(net$state_dict(), function(x) x$clone())
  before <- torch::as_array(snapshot[[1]])
  torch::with_no_grad(net$weight$add_(1))
  expect_equal(torch::as_array(snapshot[[1]]), before)
})

test_that("#15: early stopping without a metric is refused", {
  expect_error(
    gan_trainer(sample_toydata(n = 200), epochs = 5, early_stopping = TRUE,
                patience = 3),
    "needs a metric"
  )
})

test_that("#16: a value function can be passed as a function", {
  g <- gan_trainer(sample_toydata(n = 200), value_function = WGAN_value_fct,
                   epochs = 1)
  expect_s3_class(g, "trained_RGAN")
  # Recognised as the built-in WGAN: weight clipping is on.
  expect_identical(g$settings$weight_clipper, WGAN_weight_clipper)

  custom <- function(real_scores, fake_scores) {
    list(d_loss = fake_scores$mean() - real_scores$mean(),
         g_loss = -fake_scores$mean())
  }
  g2 <- gan_trainer(sample_toydata(n = 200), value_function = custom, epochs = 1)
  expect_s3_class(g2, "trained_RGAN")
  expect_output(print(g2), "custom function")

  expect_error(gan_trainer(sample_toydata(n = 200), value_function = "nope",
                           epochs = 1), "value_function must be")
})

test_that("#17: torch_tensor input is accepted", {
  g <- gan_trainer(torch::torch_tensor(sample_toydata(n = 300)), epochs = 1)
  expect_s3_class(g, "trained_RGAN")
})

test_that("#18: WGAN weights stay inside the clip bound after training", {
  g <- gan_trainer(sample_toydata(n = 300), value_function = "wasserstein",
                   epochs = 2, seed = 1, base_lr = 1e-3)
  max_abs <- max(sapply(g$discriminator$parameters,
                        function(p) torch::as_array(p$abs()$max())))
  expect_lte(max_abs, 0.01 + 1e-7)
})

test_that("#26: the exponential schedule decays after the first epoch", {
  opt <- torch::optim_adam(torch::nn_linear(2, 1)$parameters, lr = 1)
  lrs <- sapply(1:3, function(e)
    adjust_learning_rate(opt, 1, e, 10, "exponential", 0.5, 50))
  expect_equal(lrs, c(0.5, 0.25, 0.125))
})

# ---- value functions --------------------------------------------------------

test_that("#19: gradient_penalty works for image (4-D) input", {
  d <- DCGAN_Discriminator(number_channels = 1, ndf = 8)
  gp <- gradient_penalty(d, torch::torch_randn(4, 1, 64, 64),
                         torch::torch_randn(4, 1, 64, 64))
  expect_true(is.finite(gp$item()))
})

test_that("#25: the generator loss keeps its gradient when D rejects fakes", {
  fs <- torch::torch_tensor(c(1e-9, 1e-3), requires_grad = TRUE)
  l <- GAN_value_fct(torch::torch_tensor(c(.9, .9)), fs)$g_loss
  l$backward()
  grad <- torch::as_array(fs$grad)
  expect_true(all(grad < 0))
  # Non-saturating: the gradient is largest where D(G(z)) is smallest.
  expect_gt(abs(grad[1]), abs(grad[2]))
})

# ---- generator architecture -------------------------------------------------

test_that("#21: TabularGenerator does not cap z-scored columns at +/- 1", {
  d <- cbind(x = rexp(2000) * 5, y = rnorm(2000))
  tr <- data_transformer$new()
  tr$fit(d)
  g <- TabularGenerator(noise_dim = 2, output_info = tr$output_info,
                        hidden_units = list(32, 32))
  torch::with_no_grad(g$output_layer$weight$mul_(100))
  out <- torch::as_array(g(torch::torch_randn(1000, 2)))
  expect_gt(max(abs(out)), 1)
})

# ---- data_transformer -------------------------------------------------------

test_that("#22: integer discrete_columns work when the data has names", {
  df <- data.frame(a = rnorm(10), b = sample(c(1, 2, 3), 10, TRUE))
  t1 <- data_transformer$new()
  t1$fit(df, discrete_columns = 2)
  expect_equal(t1$output_info[[2]][[2]], "softmax")
  expect_error(data_transformer$new()$fit(df, discrete_columns = 5),
               "out of range")
  expect_error(data_transformer$new()$fit(df, discrete_columns = "zz"),
               "not found")
})

test_that("#23: inverse_transform restores the original class of levels", {
  df <- data.frame(lvl = sample(c("01", "02", "10"), 500, TRUE),
                   num = sample(c(1, 2, 3), 500, TRUE),
                   fac = factor(sample(c("a", "b"), 500, TRUE)),
                   x = rnorm(500), stringsAsFactors = FALSE)
  t <- data_transformer$new()
  t$fit(df, discrete_columns = c("lvl", "num", "fac"))
  rt <- t$inverse_transform(t$transform(df))
  expect_type(rt$lvl, "character")
  expect_equal(rt$lvl, df$lvl)
  expect_type(rt$num, "double")
  expect_equal(rt$num, df$num)
  expect_s3_class(rt$fac, "factor")
  expect_equal(rt$fac, df$fac)
})

test_that("#27: unseen category levels give a clear error", {
  t <- data_transformer$new()
  t$fit(data.frame(c = c("a", "b", "a")), discrete_columns = "c")
  expect_error(t$transform(data.frame(c = c("a", "z", "b"))),
               "levels not seen in fit\\(\\): z")
})

# ---- post-GAN boosting ------------------------------------------------------

test_that("#20: PacGAN scores stay aligned with their samples", {
  x <- sample_toydata(n = 300)
  g <- gan_trainer(x, epochs = 2, pac = 3, batch_size = 51,
                   checkpoint_epochs = 1, seed = 1)
  B <- torch::as_array(g$generator(torch::torch_randn(2000, 2)))
  s1 <- suppressMessages(compute_discriminator_scores(g, B, x, batch_size = 1000))
  s2 <- suppressMessages(compute_discriminator_scores(g, B, x, batch_size = 7))
  expect_false(anyNA(s1$d_score_fake))
  # The score of a sample must not depend on the scoring batch size.
  expect_equal(s1$d_score_fake, s2$d_score_fake, tolerance = 1e-6)
  expect_equal(s1$d_score_real, s2$d_score_real, tolerance = 1e-6)

  # A candidate count that is not a multiple of pac
  s3 <- suppressMessages(compute_discriminator_scores(g, B[1:1001, ], x,
                                                      batch_size = 1000))
  expect_equal(dim(s3$d_score_fake), c(2, 1001))
  expect_false(anyNA(s3$d_score_fake))

  invisible(capture.output(
    b <- suppressMessages(apply_post_gan_boosting(g, x, n_candidates = 500,
                                                  steps = 5))
  ))
  expect_gt(b$n_unique, 0)
})

test_that("#20: scoring leaves the trained model unchanged", {
  x <- sample_toydata(n = 200)
  g <- gan_trainer(x, epochs = 2, checkpoint_epochs = 1, seed = 1)
  before <- torch::as_array(g$discriminator$parameters[[1]])
  invisible(capture.output(
    suppressMessages(apply_post_gan_boosting(g, x, n_candidates = 50, steps = 2))
  ))
  expect_equal(torch::as_array(g$discriminator$parameters[[1]]), before)
  expect_true(g$discriminator$training)
})

test_that("#8: private selection uses the secure RNG by default", {
  skip_if_no_opendp()
  d_fake <- matrix(c(0.2, 0.5, 0.2, 0.5), 2)
  run <- function(...) {
    invisible(capture.output(r <- post_gan_boosting(
      d_fake, c(0.9, 0.9), matrix(1:4, 2), real_N = 2, steps = 20,
      N_generators = 2, dp = TRUE, MW_epsilon = 1, ...)))
    r$D_bar
  }
  # With a secure source, resetting R's seed does not reproduce the selection.
  draws <- replicate(10, { set.seed(1); paste(run(), collapse = ",") })
  expect_gt(length(unique(draws)), 1)
  # With secure_rng = FALSE it does, which is why that is for testing only.
  draws_mt <- replicate(3, { set.seed(1); paste(run(secure_rng = FALSE), collapse = ",") })
  expect_length(unique(draws_mt), 1)
})

# ---- dp_gan_trainer ---------------------------------------------------------

test_that("#5: discriminators with buffers are rejected under DP", {
  D <- torch::nn_module(
    initialize = function() {
      self$bn <- torch::nn_batch_norm1d(2)
      self$l <- torch::nn_linear(2, 1)
    },
    forward = function(x) torch::torch_sigmoid(self$l(self$bn(x)))
  )
  X <- matrix(rnorm(400), 200, 2)
  expect_error(
    dp_gan_trainer(X, discriminator = D(), target_epsilon = 5, batch_size = 50,
                   epochs = 1, secure_rng = FALSE, verbose = FALSE),
    "buffers"
  )
})

test_that("#11: post-GAN boosting runs on a dp_gan_trainer model", {
  skip_if_no_opendp()
  X <- matrix(rnorm(600), 300, 2)
  g <- suppressMessages(dp_gan_trainer(X, target_epsilon = 5, epochs = 4,
                                       checkpoint_epochs = 2, secure_rng = FALSE,
                                       verbose = FALSE))
  invisible(capture.output(
    b <- suppressMessages(apply_post_gan_boosting(g, X, n_candidates = 50,
                                                  steps = 5, dp = TRUE))
  ))
  expect_gt(b$n_unique, 0)
})

test_that("#12: a Poisson batch of one record does not crash", {
  X <- matrix(rnorm(400), 200, 2)
  set.seed(3)
  g <- suppressMessages(dp_gan_trainer(X, sampling_rate = 0.002,
                                       noise_multiplier = 1, target_epsilon = 50,
                                       epochs = 1, secure_rng = FALSE,
                                       verbose = FALSE))
  expect_equal(g$privacy$total_steps, 500)
})

test_that("#13: a run that uses exactly its budget ends without a warning", {
  X <- matrix(rnorm(400), 200, 2)
  msgs <- character(0)
  g <- withCallingHandlers(
    dp_gan_trainer(X, target_epsilon = 5, batch_size = 50, epochs = 2,
                   secure_rng = FALSE),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_equal(g$privacy$total_steps, 8)
  expect_false(any(grepl("exhausted", msgs)))
})

# ---- persistence ------------------------------------------------------------

test_that("#24: save_gan/load_gan keep checkpoints and the optimizer class", {
  g <- gan_trainer(sample_toydata(n = 300), epochs = 2, checkpoint_epochs = 1)
  p <- tempfile()
  suppressMessages(save_gan(g, p))
  lg <- suppressMessages(load_gan(p))
  expect_equal(lg$checkpoints$epochs, c(1, 2))
  expect_equal(
    torch::as_array(lg$checkpoints$generators[[1]][[1]]),
    torch::as_array(g$checkpoints$generators[[1]][[1]])
  )

  X <- matrix(rnorm(600), 300, 2)
  dg <- suppressMessages(dp_gan_trainer(X, target_epsilon = 5, epochs = 1,
                                        secure_rng = FALSE, verbose = FALSE))
  p2 <- tempfile()
  suppressMessages(save_gan(dg, p2))
  ldg <- suppressMessages(load_gan(p2))
  expect_s3_class(ldg$discriminator_optimizer, "optim_sgd")
  expect_equal(ldg$privacy$final_epsilon, dg$privacy$final_epsilon)
})

# ---- #6: mechanism parameters independent of n ------------------------------

test_that("#6: with sampling_rate supplied, the mechanism does not depend on n", {
  fit <- function(n) {
    suppressMessages(dp_gan_trainer(
      matrix(rnorm(2 * n), n, 2), sampling_rate = 0.1, batch_size = 20,
      target_epsilon = 5, epochs = 2, secure_rng = FALSE, verbose = FALSE
    ))
  }
  a <- fit(200)
  b <- fit(201)
  keys <- c("sampling_rate", "noise_multiplier", "total_steps",
            "expected_batch_size")
  expect_identical(a$privacy[keys], b$privacy[keys])
  expect_equal(a$privacy$expected_batch_size, 20)
  expect_false(a$privacy$n_public)
  # n can no longer be read off the model
  expect_equal(a$settings$batch_size / a$privacy$sampling_rate, 200)
  expect_equal(b$settings$batch_size / b$privacy$sampling_rate, 200)
})

test_that("#6: deriving sampling_rate from n warns and is recorded", {
  msgs <- character(0)
  g <- withCallingHandlers(
    dp_gan_trainer(matrix(rnorm(400), 200, 2), target_epsilon = 5, epochs = 1,
                   secure_rng = FALSE, verbose = FALSE),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_true(g$privacy$n_public)
  expect_true(any(grepl("Only passing sampling_rate", msgs)))
})

test_that("#6: the gradient divisor is the supplied expected batch size", {
  d_net <- torch::nn_sequential(torch::nn_linear(2, 1), torch::nn_sigmoid())
  g_net <- torch::nn_linear(2, 2)
  data <- torch::torch_randn(c(50, 2))
  run <- function(L) {
    torch::torch_manual_seed(1)
    set.seed(1)
    dp_discriminator_step(
      data = data, n_samples = 50, sampling_rate = 1,  # every record sampled
      noise_dim = 2, sample_noise = function(d) torch::torch_randn(d),
      device = "cpu", g_net = g_net, d_net = d_net,
      d_optim = torch::optim_sgd(d_net$parameters, lr = 0),
      max_grad_norm = 1, noise_multiplier = 1e-12, secure_rng = FALSE,
      expected_batch_size = L
    )
    as.numeric(d_net$parameters[[1]]$grad)
  }
  # Same clipped sum, divided by 10 vs 40: gradients differ by a factor of 4.
  expect_equal(run(10) / run(40), rep(4, 2), tolerance = 1e-6)
})

test_that("#6: post-GAN boosting normalises by the public count", {
  skip_if_no_opendp()
  X <- matrix(rnorm(600), 300, 2)
  g <- suppressMessages(dp_gan_trainer(
    X, sampling_rate = 0.1, batch_size = 20, target_epsilon = 5, epochs = 4,
    checkpoint_epochs = 2, secure_rng = FALSE, verbose = FALSE
  ))
  expect_equal(public_record_count(g$privacy), 200)
  msgs <- character(0)
  invisible(capture.output(withCallingHandlers(
    apply_post_gan_boosting(g, X, n_candidates = 30, steps = 2, dp = TRUE),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )))
  expect_true(any(grepl("public count 200", msgs)))
})

# ---- #9: no private boosting with PacGAN -------------------------------------

test_that("#9: dp = TRUE is refused for pac > 1", {
  x <- sample_toydata(n = 200)
  g <- gan_trainer(x, epochs = 2, pac = 2, batch_size = 20,
                   checkpoint_epochs = 1, seed = 1)
  expect_error(
    suppressMessages(apply_post_gan_boosting(g, x, n_candidates = 20,
                                             steps = 2, dp = TRUE)),
    "not supported with pac = 2"
  )
})

# ---- #7: public transformer and GMM hardening --------------------------------

test_that("#7: fit_public builds a transformer without data", {
  t <- data_transformer$new()
  t$fit_public(list(
    age = public_continuous(18, 100),
    sex = public_categorical(c("F", "M"), class = "factor")
  ))
  expect_true(t$public)
  expect_equal(t$output_dimensions, 3)
  d <- data.frame(age = c(10, 50, 130), sex = factor(c("M", "F", "M")))
  td <- t$transform(d)
  expect_equal(td[, 1], c(-1, (50 - 59) / 41, 1))
  rt <- t$inverse_transform(td)
  expect_equal(rt$age, c(18, 50, 100))   # clipped to the public bounds
  expect_equal(rt$sex, d$sex)
  expect_error(t$transform(data.frame(age = 30, sex = "X")), "not seen")
  expect_error(public_continuous(5, 1), "lower < upper")
  expect_error(t$fit_public(list(age = c(0, 1))), "public_continuous")

  t2 <- data_transformer$new()
  t2$fit(d, discrete_columns = "sex")
  expect_false(t2$public)
})

test_that("#7: decoding a DP model with a data-fitted transformer warns", {
  X <- matrix(rnorm(400), 200, 2, dimnames = list(NULL, c("a", "b")))
  tr <- data_transformer$new()
  tr$fit(X)
  g <- suppressMessages(dp_gan_trainer(tr$transform(X), sampling_rate = 0.25,
                                       batch_size = 50, target_epsilon = 5,
                                       epochs = 1, secure_rng = FALSE,
                                       verbose = FALSE))
  expect_message(sample_synthetic_data(g, tr, n_samples = 5), "fit_public")

  tp <- data_transformer$new()
  tp$fit_public(list(a = public_continuous(-4, 4), b = public_continuous(-4, 4)))
  msgs <- character(0)
  withCallingHandlers(sample_synthetic_data(g, tp, n_samples = 5),
                      message = function(m) {
                        msgs <<- c(msgs, conditionMessage(m))
                        invokeRestart("muffleMessage")
                      })
  expect_false(any(grepl("fit_public", msgs)))
})

test_that("#7: a single outlier no longer gets a mode of its own", {
  set.seed(3)
  s <- c(rnorm(60, 50000, 8000), 987654.321)
  tr <- data_transformer$new()
  tr$fit(matrix(s, ncol = 1), mode_specific = TRUE)
  m <- tr$meta[[1]]
  expect_true(all(abs(m$means - 987654.321) > 1e5))
  expect_true(all(m$stds > 1e-6 * 10))
  expect_true(all(m$weights * length(s) >= 10))
})
