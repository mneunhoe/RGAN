test_that("resolve_instance_noise fills in defaults and validates", {
  expect_null(resolve_instance_noise(NULL))
  n <- resolve_instance_noise(list())
  expect_equal(n$sigma_max, 0.5)
  expect_equal(n$sigma_min, 0)
  expect_true(n$adaptive)
  expect_equal(n$target_rd, 0.6)
  expect_true(n$per_type)
  expect_false(n$condition_d)
  expect_equal(n$update_every, 4L)
  expect_error(resolve_instance_noise(list(sigma_max = -1)), "sigma_max")
  expect_error(resolve_instance_noise(list(sigma_min = 1, sigma_max = 0.5)),
               "sigma_min")
  expect_error(resolve_instance_noise(list(update_every = 0)), "update_every")
  expect_error(resolve_instance_noise("x"), "must be NULL or a list")
})

test_that("instance_noise_scale halves the categorical blocks", {
  oi <- list(list(4, "mode_specific"), list(3, "softmax"), list(2, "cossin"),
             list(1, "quantile"), list(1, "linear"))
  s <- instance_noise_scale(oi, 11, per_type = TRUE)
  # mode_specific: 3 indicator columns at 0.5, value column at 1
  expect_equal(s, c(0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1))
  expect_equal(instance_noise_scale(oi, 11, per_type = FALSE), rep(1, 11))
  expect_equal(instance_noise_scale(NULL, 4, per_type = TRUE), rep(1, 4))
})

test_that("add_instance_noise keeps the shape, scales per column and is differentiable", {
  torch::torch_manual_seed(1)
  x <- torch::torch_zeros(c(2000, 2), requires_grad = TRUE)
  scale <- torch::torch_tensor(matrix(c(1, 0.5), nrow = 1))
  y <- add_instance_noise(x, sigma = 1, noise_scale = scale)
  expect_equal(y$shape, c(2000, 2))
  sds <- as.numeric(torch::as_array(y$std(dim = 1)$detach()$cpu()))
  expect_equal(sds[1], 1, tolerance = 0.1)
  expect_equal(sds[2], 0.5, tolerance = 0.1)
  y$sum()$backward()
  expect_true(as.numeric(torch::as_array(x$grad$abs()$sum()$cpu())) > 0)
})

test_that("append_sigma_column adds exactly one column of sigma", {
  x <- torch::torch_zeros(c(5, 3))
  y <- append_sigma_column(x, 0.25)
  expect_equal(y$shape, c(5, 4))
  expect_equal(as.numeric(torch::as_array(y[, 4]$cpu())), rep(0.25, 5))
})

# ---------------------------------------------------------------------------
# Adaptive controller
# ---------------------------------------------------------------------------

test_that("the noise ceiling stays within bounds and rises for a confident D", {
  # A discriminator that separates real from fake trivially: the real data sits
  # far away from anything the generator can produce at initialisation, the
  # discriminator has no dropout and a large learning rate, so r_d saturates at
  # 1 > target_rd and the noise ceiling must climb.
  set.seed(7)
  d <- matrix(stats::rnorm(400 * 2, mean = 50, sd = 0.1), ncol = 2)
  g <- gan_trainer(d, epochs = 10, batch_size = 50, noise_dim = 2,
                   value_function = "non-saturating",
                   discriminator_dropout = 0, base_lr = 0.01, ttur_factor = 4,
                   seed = 7, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.5, update_every = 1,
                                         target_rd = 0.6))
  tr <- g$losses$noise_T
  expect_true(length(tr) > 20)
  expect_true(all(tr >= 0 & tr <= 0.5))
  expect_true(utils::tail(tr, 1) > tr[1])
  # Monotone increase while the discriminator is confident.
  expect_true(mean(diff(tr) > 0) > 0.8)
  expect_equal(g$settings$noise_T_final, utils::tail(tr, 1))
})

test_that("resolve_rd_mode picks the statistic that matches the output", {
  expect_equal(resolve_rd_mode("original"), "sigmoid")
  expect_equal(resolve_rd_mode("non-saturating"), "sign_real")
  expect_equal(resolve_rd_mode("hinge"), "sign_real")
  expect_equal(resolve_rd_mode("wasserstein"), "paired")
  expect_equal(resolve_rd_mode("wgan-gp"), "paired")
  expect_equal(resolve_rd_mode("f-wgan"), "paired")
  # rpgan reaches the trainer with paired_loss = TRUE.
  expect_equal(resolve_rd_mode("rpgan", paired_loss = TRUE), "paired")
  # A user-supplied function is paired only when it declares that it needs the
  # real scores.
  expect_equal(resolve_rd_mode(function(r, f) NULL), "sign_real")
  expect_equal(resolve_rd_mode(function(r, f) NULL, paired_loss = TRUE), "paired")
})

test_that("the rpgan controllers react to a confident D via the paired statistic", {
  # The relativistic loss leaves the absolute level of D unconstrained, so
  # mean(sign(D(real))) carries no overfitting information. With the paired
  # statistic a discriminator that ranks real above fake drives T and p up.
  set.seed(31)
  d <- matrix(stats::rnorm(200 * 2, mean = 50, sd = 0.1), ncol = 2)
  g <- gan_trainer(d, epochs = 12, batch_size = 50, noise_dim = 2,
                   value_function = "rpgan",
                   discriminator_dropout = 0, base_lr = 0.01, ttur_factor = 4,
                   seed = 31, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.5, update_every = 1),
                   augmentation = list(p_max = 0.5, update_every = 1))
  tr <- g$losses$noise_T
  pp <- g$losses$aug_p
  expect_true(length(tr) > 20)
  expect_true(all(tr >= 0 & tr <= 0.5))
  expect_true(all(pp >= 0 & pp <= 0.5))
  # Both controllers must actually leave their floor. Under the old
  # sign(D(real)) statistic they stayed pinned at 0 for the whole run.
  expect_true(utils::tail(tr, 1) > 0.05)
  expect_true(utils::tail(pp, 1) > 0.05)
  # It is a closed loop, so single steps down are expected once the noise
  # starts hurting the discriminator; the trend must still be upwards.
  expect_true(mean(diff(tr) > 0) > 0.7)
  third <- max(1, length(tr) %/% 3)
  expect_true(mean(utils::tail(tr, third)) > mean(utils::head(tr, third)))
})

test_that("the wgan-gp controller also uses the paired statistic", {
  set.seed(32)
  d <- matrix(stats::rnorm(200 * 2, mean = 50, sd = 0.1), ncol = 2)
  g <- gan_trainer(d, epochs = 12, batch_size = 50, noise_dim = 2,
                   value_function = "wgan-gp", gp_lambda = 10,
                   discriminator_dropout = 0, base_lr = 0.01, ttur_factor = 4,
                   seed = 32, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.5, update_every = 1))
  expect_true(utils::tail(g$losses$noise_T, 1) > 0)
  expect_true(all(g$losses$noise_T <= 0.5))
})

test_that("the noise ceiling respects sigma_min and sigma_max bounds", {
  d <- matrix(stats::rnorm(400 * 2), ncol = 2)
  g <- gan_trainer(d, epochs = 6, batch_size = 50, noise_dim = 2,
                   seed = 8, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.2, sigma_min = 0.1,
                                         update_every = 1))
  expect_true(all(g$losses$noise_T >= 0.1 - 1e-12))
  expect_true(all(g$losses$noise_T <= 0.2 + 1e-12))
})

test_that("non-adaptive instance noise keeps T at sigma_max", {
  d <- matrix(stats::rnorm(200 * 2), ncol = 2)
  g <- gan_trainer(d, epochs = 4, batch_size = 50, noise_dim = 2,
                   seed = 9, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.3, adaptive = FALSE,
                                         update_every = 1))
  expect_true(all(g$losses$noise_T == 0.3))
  expect_equal(g$settings$noise_T_final, 0.3)
})

# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------

mixed_noise_data <- function(n = 200) {
  set.seed(21)
  data.frame(
    num1 = c(stats::rnorm(n / 2, -3), stats::rnorm(n / 2, 3)),
    num2 = stats::rnorm(n),
    cat1 = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    cat2 = factor(sample(c("x", "y"), n, replace = TRUE))
  )
}

test_that("instance noise works on mode_specific + onehot data", {
  df <- mixed_noise_data(200)
  tr <- data_transformer$new()
  tr$fit(df, discrete_columns = c("cat1", "cat2"),
         mode_specific = TRUE, continuous_transform = "mode_specific", n_modes = 3)
  td <- tr$transform(df)
  g <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 4,
                   output_info = tr$output_info, seed = 10, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.5, update_every = 1))
  expect_s3_class(g, "trained_RGAN")
  expect_equal(ncol(sample_synthetic_data(g, tr)), 4)
})

test_that("instance noise works on quantile + cossin data, also with both features", {
  df <- mixed_noise_data(200)
  tr <- data_transformer$new()
  tr$fit(df, discrete_columns = c("cat1", "cat2"),
         continuous_transform = "quantile", categorical_encoding = "cossin")
  td <- tr$transform(df)
  g <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 4,
                   output_info = tr$output_info, seed = 11, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.5, update_every = 1),
                   augmentation = list(p_max = 0.5, update_every = 1))
  expect_s3_class(g, "trained_RGAN")
  expect_true(length(g$losses$noise_T) > 0)
  expect_true(length(g$losses$aug_p) > 0)
  expect_equal(ncol(sample_synthetic_data(g, tr)), 4)
})

test_that("instance noise works with pac > 1 and with rpgan + r1r2", {
  d <- sample_toydata(n = 200)
  tr <- data_transformer$new()
  tr$fit(d)
  td <- tr$transform(d)
  g <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2, pac = 10,
                   seed = 12, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.5, update_every = 1),
                   augmentation = list(p_max = 0.5, update_every = 1))
  expect_s3_class(g, "trained_RGAN")

  g2 <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2,
                    value_function = "rpgan",
                    regularization = list(type = "r1r2", gamma = 10),
                    seed = 13, track_loss = TRUE,
                    instance_noise = list(sigma_max = 0.5, update_every = 1))
  expect_s3_class(g2, "trained_RGAN")
  expect_false(any(is.na(g2$losses$d_loss)))
})

test_that("condition_d adds one input column to the default discriminator", {
  d <- sample_toydata(n = 200)
  tr <- data_transformer$new()
  tr$fit(d)
  td <- tr$transform(d)
  data_dim <- ncol(td)

  g <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2,
                   seed = 14, track_loss = TRUE,
                   instance_noise = list(sigma_max = 0.5, condition_d = TRUE,
                                         update_every = 1))
  in_features <- g$discriminator$parameters[["seq.Linear_1.weight"]]$shape[2]
  expect_equal(in_features, data_dim + 1)

  g2 <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2, pac = 10,
                    seed = 15, track_loss = TRUE,
                    instance_noise = list(sigma_max = 0.5, condition_d = TRUE,
                                          update_every = 1))
  in_features2 <- g2$discriminator$parameters[["seq.Linear_1.weight"]]$shape[2]
  expect_equal(in_features2, data_dim * 10 + 1)
})

test_that("condition_d works together with validation data and early stopping", {
  d <- sample_toydata(n = 200)
  tr <- data_transformer$new()
  tr$fit(d)
  td <- tr$transform(d)
  g <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2,
                   validation_data = td, early_stopping = TRUE, patience = 10,
                   seed = 16,
                   instance_noise = list(sigma_max = 0.5, condition_d = TRUE))
  expect_s3_class(g, "trained_RGAN")
  expect_true(length(g$validation_metrics) > 0)
})

test_that("instance_noise = NULL is a no-op (bit-identical training)", {
  d <- sample_toydata(n = 200)
  tr <- data_transformer$new()
  tr$fit(d)
  td <- tr$transform(d)
  run <- function(...) {
    set.seed(1)
    torch::torch_manual_seed(1)
    gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2, pac = 10,
                seed = 42, track_loss = TRUE, ...)
  }
  a <- run()
  b <- run(instance_noise = NULL)
  expect_identical(a$losses$g_loss, b$losses$g_loss)
  expect_identical(a$losses$d_loss, b$losses$d_loss)
  expect_null(a$losses$noise_T)
  expect_null(a$settings$noise_T_final)
})
