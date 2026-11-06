# Experiment 6: Diffusion2GAN-style pair-regression distillation of a
# flow-matching teacher into a one-step GAN generator.

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

tiny_teacher <- function(X, output_info = NULL, seed = 1) {
  flow_matching_trainer(
    X, output_info = output_info,
    epochs = 2, batch_size = 50, base_lr = 1e-3,
    vector_field = FlowVectorField(ncol(X), hidden_units = list(16, 16),
                                   time_embed_dim = 8),
    device = "cpu", seed = seed, track_loss = TRUE, verbose = FALSE)
}

toy_frame <- function(n = 300, seed = 7) {
  set.seed(seed)
  data.frame(
    x1 = stats::rnorm(n),
    x2 = c(stats::rnorm(n %/% 2, -3), stats::rnorm(n - n %/% 2, 3)),
    c1 = sample(c("a", "b", "c"), n, TRUE, prob = c(.6, .3, .1)),
    c2 = sample(c("u", "v"), n, TRUE),
    stringsAsFactors = FALSE
  )
}


# ---------------------------------------------------------------------------
# The regression loss itself
# ---------------------------------------------------------------------------

test_that("distill_regression_loss is zero at the targets and positive away from them", {
  x <- torch::torch_randn(8, 5)
  expect_equal(as.numeric(distill_regression_loss(x, x$clone())), 0)
  expect_equal(as.numeric(distill_regression_loss(x, x$clone(), loss = "huber")), 0)

  w <- torch::torch_ones(c(1, 5))
  expect_equal(as.numeric(distill_regression_loss(x, x$clone(), weights = w)), 0)

  y <- x + 1
  expect_equal(as.numeric(distill_regression_loss(x, y)), 1, tolerance = 1e-6)
  # Huber with delta 1 on a residual of exactly 1 is 0.5 * 1^2.
  expect_equal(as.numeric(distill_regression_loss(x, y, loss = "huber")), 0.5,
               tolerance = 1e-6)
  # Huber is below MSE for large residuals.
  z <- x + 10
  expect_lt(as.numeric(distill_regression_loss(x, z, loss = "huber")),
            as.numeric(distill_regression_loss(x, z)))
})


test_that("distill_regression_loss honours the per-column weights and is differentiable", {
  fake <- torch::torch_zeros(c(4, 3), requires_grad = TRUE)
  target <- torch::torch_ones(c(4, 3))
  w <- torch::torch_tensor(matrix(c(3, 0, 0), nrow = 1))
  l <- distill_regression_loss(fake, target, weights = w)
  # only the first column contributes: mean over 12 cells of 3 * 1 on 4 cells
  expect_equal(as.numeric(l), 3 * 4 / 12, tolerance = 1e-6)
  l$backward()
  expect_true(all(is.finite(as.matrix(fake$grad))))
  expect_equal(as.numeric(torch::torch_sum(torch::torch_abs(fake$grad[, 2:3]))), 0)

  # shape mismatch is caught
  expect_error(distill_regression_loss(fake, torch::torch_ones(c(4, 2))),
               "same shape")
})


test_that("the per-type weights come from flow_loss_weights", {
  oi <- list(list(1, "quantile"), list(4, "softmax"))
  w <- flow_loss_weights(oi, 5)
  expect_equal(length(w), 5)
  expect_equal(mean(w), 1)
  expect_gt(w[1], w[2])
})


# ---------------------------------------------------------------------------
# Pair validation and the noise_dim contract
# ---------------------------------------------------------------------------

test_that("pair shapes are validated", {
  good <- list(z = matrix(0, 10, 3), x = matrix(0, 10, 3))
  v <- validate_flow_pairs(good, 3)
  expect_equal(dim(v$z), c(10L, 3L))

  expect_error(validate_flow_pairs(list(z = matrix(0, 10, 3)), 3), "`z` and `x`")
  expect_error(validate_flow_pairs(list(z = matrix(0, 9, 3), x = matrix(0, 10, 3)), 3),
               "9 rows but x has 10")
  expect_error(validate_flow_pairs(list(z = matrix(0, 10, 3), x = matrix(0, 10, 2)), 3),
               "2 columns but the training data has 3")
  bad <- good; bad$x[1, 1] <- NA
  expect_error(validate_flow_pairs(bad, 3), "NA/NaN/Inf")

  expect_error(resolve_distillation(list(lambda = 1), 3, 3, 10),
               "either `pairs`")
  expect_error(resolve_distillation(list(pairs = good, lambda = -1), 3, 3, 10),
               "non-negative")
  expect_error(resolve_distillation(list(pairs = good, refresh_every = 2), 3, 3, 10),
               "needs a `teacher`")
  expect_error(resolve_distillation(list(teacher = "nope"), 3, 3, 10),
               "trained_RFlow")
})


test_that("noise_match = TRUE requires noise_dim == ncol(pairs$z)", {
  pairs <- list(z = matrix(stats::rnorm(40), 10, 4),
                x = matrix(stats::rnorm(40), 10, 4))
  expect_error(resolve_distillation(list(pairs = pairs), noise_dim = 2,
                                    data_dim = 4, n_rows = 10),
               "noise_dim = 2 but ncol\\(pairs\\$z\\) = 4")
  # the error tells the user what to do
  expect_error(resolve_distillation(list(pairs = pairs), noise_dim = 2,
                                    data_dim = 4, n_rows = 10),
               "Set noise_dim = ncol\\(pairs\\$z\\)")
  # ... and noise_match = FALSE lifts the requirement
  r <- resolve_distillation(list(pairs = pairs, noise_match = FALSE),
                            noise_dim = 2, data_dim = 4, n_rows = 10)
  expect_false(r$noise_match)
  expect_equal(r$n_pairs, 10L)

  # matching dimensions resolve cleanly, with the documented defaults
  ok <- resolve_distillation(list(pairs = pairs), noise_dim = 4,
                             data_dim = 4, n_rows = 10)
  expect_equal(ok$lambda, 10)
  expect_equal(ok$loss, "mse")
  expect_true(ok$adversarial)
  expect_true(ok$noise_match)
  expect_equal(dim(ok$z_t), c(10L, 4L))
})


test_that("a bare teacher generates 4 x nrow(data) pairs", {
  X <- scale(as.matrix(RGAN::sample_toydata(n = 200)))
  fm <- tiny_teacher(X)
  r <- resolve_distillation(fm, noise_dim = 2, data_dim = 2, n_rows = 25)
  expect_equal(r$n_pairs, 100L)
  expect_equal(dim(r$pairs$x), c(100L, 2L))
})


# ---------------------------------------------------------------------------
# gan_trainer with distillation
# ---------------------------------------------------------------------------

test_that("gan_trainer trains with distillation, adversarial TRUE and FALSE", {
  X <- scale(as.matrix(RGAN::sample_toydata(n = 400)))
  fm <- tiny_teacher(X)
  pairs <- flow_pairs(fm, n_pairs = 300, n_steps = 3, seed = 2)

  adv <- gan_trainer(X, noise_dim = 2, epochs = 2, batch_size = 50, seed = 3,
                     track_loss = TRUE,
                     distillation = list(pairs = pairs, lambda = 2))
  expect_s3_class(adv, "trained_RGAN")
  expect_equal(adv$settings$n_d_steps, adv$settings$n_g_steps)
  expect_gt(adv$settings$n_d_steps, 0)
  expect_equal(length(adv$losses$distill_loss), adv$settings$n_g_steps)
  expect_true(all(is.finite(adv$losses$distill_loss)))
  expect_true(all(adv$losses$distill_loss >= 0))
  expect_equal(adv$settings$distillation$lambda, 2)
  expect_null(adv$settings$distillation$pairs)   # the matrices are not stored

  # Pure regression: no discriminator step at all.
  reg <- gan_trainer(X, noise_dim = 2, epochs = 2, batch_size = 50, seed = 3,
                     track_loss = TRUE,
                     distillation = list(pairs = pairs, lambda = 2,
                                         adversarial = FALSE))
  expect_equal(reg$settings$n_d_steps, 0L)
  expect_gt(reg$settings$n_g_steps, 0)
  expect_true(all(reg$losses$d_loss == 0))
  expect_true(all(is.finite(reg$losses$g_loss)))
  # With lambda 2 and no adversarial term, g_loss is exactly lambda * distill.
  # (`losses$g_loss` records the first step twice, a long-standing quirk of the
  # trainer's accumulator; `losses$distill_loss` is one value per step.)
  expect_equal(reg$losses$g_loss[-1], 2 * reg$losses$distill_loss,
               tolerance = 1e-5)
  # and the regression term actually falls
  expect_lt(mean(utils::tail(reg$losses$distill_loss, 5)),
            mean(utils::head(reg$losses$distill_loss, 5)))
})


test_that("distillation works with pac, rpgan + r1r2 and huber", {
  X <- scale(as.matrix(RGAN::sample_toydata(n = 500)))
  fm <- tiny_teacher(X)
  pairs <- flow_pairs(fm, n_pairs = 200, n_steps = 3, seed = 4)

  fit <- gan_trainer(X, noise_dim = 2, epochs = 2, batch_size = 50, pac = 10,
                     seed = 5, track_loss = TRUE,
                     value_function = "rpgan",
                     regularization = list(type = "r1r2", gamma = 10),
                     adam_betas = c(0, 0.99), discriminator_dropout = 0,
                     distillation = list(pairs = pairs, lambda = 1,
                                         loss = "huber"))
  expect_s3_class(fit, "trained_RGAN")
  expect_equal(fit$settings$distillation$loss, "huber")
  expect_true(all(is.finite(fit$losses$g_loss)))
  expect_true(all(is.finite(fit$losses$distill_loss)))
})


test_that("distillation works with quantile + cossin output_info and conditioning", {
  d <- toy_frame(n = 300)
  tr <- data_transformer$new()
  tr$fit(d, discrete_columns = c("c1", "c2"),
         continuous_transform = "quantile", categorical_encoding = "cossin",
         n_quantiles = 100)
  X <- tr$transform(d)
  X[!is.finite(X)] <- 0
  fm <- tiny_teacher(X, output_info = tr$output_info)
  pairs <- flow_pairs(fm, n_pairs = 200, n_steps = 3, seed = 6)

  fit <- gan_trainer(X, noise_dim = ncol(X), epochs = 2, batch_size = 50,
                     seed = 7, track_loss = TRUE, output_info = tr$output_info,
                     value_function = "rpgan",
                     distillation = list(pairs = pairs, lambda = 10))
  expect_true(all(is.finite(fit$losses$distill_loss)))
  # the per-type weights are the flow teacher's
  expect_equal(fit$settings$distillation$per_type_weight, TRUE)
  syn <- sample_synthetic_data(fit, tr)
  expect_equal(nrow(syn), fit$settings$synthetic_examples)
  expect_setequal(names(syn), names(d))

  # ... and with conditioning switched on as well: the regression term uses
  # the all-unobserved condition, so the generator input width still matches.
  fit_c <- gan_trainer(X, noise_dim = ncol(X), epochs = 2, batch_size = 50,
                       seed = 7, track_loss = TRUE,
                       output_info = tr$output_info,
                       value_function = "rpgan",
                       conditioning = list(type = "random_mask"),
                       distillation = list(pairs = pairs, lambda = 10))
  expect_true(all(is.finite(fit_c$losses$distill_loss)))
  expect_gt(fit_c$settings$cond_dim, 0)
})


test_that("refresh_every regenerates the pairs from the teacher", {
  X <- scale(as.matrix(RGAN::sample_toydata(n = 400)))
  fm <- tiny_teacher(X)
  fit <- gan_trainer(X, noise_dim = 2, epochs = 4, batch_size = 100, seed = 8,
                     track_loss = TRUE,
                     distillation = list(teacher = fm, n_pairs = 200,
                                         n_steps = 3, refresh_every = 2,
                                         lambda = 1))
  expect_equal(fit$settings$distillation$refresh_every, 2L)
  expect_equal(fit$settings$distillation$n_pairs, 200L)
  expect_true(all(is.finite(fit$losses$distill_loss)))
})


# ---------------------------------------------------------------------------
# distill_to_gan
# ---------------------------------------------------------------------------

test_that("distill_to_gan works end to end and its output samples", {
  d <- toy_frame(n = 300)
  tr <- data_transformer$new()
  tr$fit(d, discrete_columns = c("c1", "c2"),
         continuous_transform = "quantile", categorical_encoding = "cossin",
         n_quantiles = 100)
  X <- tr$transform(d)
  X[!is.finite(X)] <- 0
  fm <- tiny_teacher(X, output_info = tr$output_info)

  student <- distill_to_gan(fm, X, transformer = tr, n_pairs = 200,
                            n_steps = 3, lambda = 5, pair_seed = 1,
                            epochs = 2, batch_size = 50, seed = 9,
                            track_loss = TRUE, value_function = "rpgan",
                            regularization = list(type = "r1r2", gamma = 10))
  expect_s3_class(student, "trained_RGAN")
  # noise_dim is the teacher's data_dim
  expect_equal(student$settings$noise_dim, ncol(X))
  expect_equal(student$settings$distillation$lambda, 5)
  expect_equal(student$settings$distillation$n_pairs, 200L)
  expect_equal(student$settings$distillation$n_steps, 3)
  expect_true(all(is.finite(student$losses$distill_loss)))

  syn <- sample_synthetic_data(student, tr, n_samples = 100)
  expect_equal(nrow(syn), 100)
  expect_setequal(names(syn), names(d))
  expect_true(all(stats::complete.cases(syn)))

  # a data.frame `data` is transformed for the caller
  student2 <- distill_to_gan(fm, d, transformer = tr, n_pairs = 100,
                             n_steps = 3, lambda = 1, epochs = 1,
                             batch_size = 50, seed = 10)
  expect_s3_class(student2, "trained_RGAN")

  # the ablation flows through
  reg_only <- distill_to_gan(fm, X, transformer = tr, n_pairs = 100,
                             n_steps = 3, epochs = 1, batch_size = 50,
                             seed = 11, track_loss = TRUE,
                             distillation = list(adversarial = FALSE))
  expect_equal(reg_only$settings$n_d_steps, 0L)

  expect_error(distill_to_gan(fm, X, noise_dim = 3), "do not pass noise_dim")
  expect_error(distill_to_gan("not a teacher", X), "trained_RFlow")
  expect_error(distill_to_gan(fm, d), "needs a fitted `transformer`")
})


# ---------------------------------------------------------------------------
# The NULL path stays bit-identical
# ---------------------------------------------------------------------------

test_that("distillation = NULL is a no-op (bit-identical training)", {
  d <- toy_frame(n = 300)
  tr <- data_transformer$new()
  tr$fit(d, discrete_columns = c("c1", "c2"), mode_specific = TRUE, n_modes = 3)
  td <- tr$transform(d)
  run <- function(...) {
    set.seed(1)
    torch::torch_manual_seed(1)
    # No zero-centred penalty here: the double backward of r1/r2 is not
    # bit-reproducible on a multi-threaded CPU backend.
    gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 4, pac = 10,
                seed = 42, track_loss = TRUE, output_info = tr$output_info,
                value_function = "rpgan", ...)
  }
  a <- run()
  b <- run(distillation = NULL)
  expect_identical(a$losses$g_loss, b$losses$g_loss)
  expect_identical(a$losses$d_loss, b$losses$d_loss)
  expect_null(a$settings$distillation)
  expect_null(a$losses$distill_loss)
  expect_equal(a$settings$n_d_steps, a$settings$n_g_steps)

  # And the same as the reference recorded before Experiment 6 was added, the
  # same reference the conditioning no-op test uses (see
  # tests/testthat/test-conditioning.R for why this one is numeric).
  ref_path <- test_path("ref_losses_conditioning.rds")
  skip_if_not(file.exists(ref_path))
  ref <- readRDS(ref_path)
  expect_equal(a$losses$g_loss, ref$g_loss, tolerance = 1e-6)
  expect_equal(a$losses$d_loss, ref$d_loss, tolerance = 1e-6)
})
