test_that("output_info_blocks splits mode_specific into indicator and value", {
  oi <- list(list(6, "mode_specific"), list(3, "softmax"), list(1, "quantile"))
  bl <- output_info_blocks(oi, 10)
  expect_true(bl$ok)
  expect_equal(length(bl$blocks), 4)
  expect_equal(bl$blocks[[1]]$start, 1)
  expect_equal(bl$blocks[[1]]$end, 5)
  expect_true(bl$blocks[[1]]$categorical)
  expect_equal(bl$blocks[[2]]$start, 6)
  expect_equal(bl$blocks[[2]]$end, 6)
  expect_false(bl$blocks[[2]]$categorical)
  expect_true(bl$blocks[[3]]$categorical)
  expect_false(bl$blocks[[4]]$categorical)
})

test_that("output_info_blocks falls back to per-column numeric blocks", {
  bl <- output_info_blocks(NULL, 3)
  expect_false(bl$ok)
  expect_equal(length(bl$blocks), 3)
  expect_true(all(vapply(bl$blocks, function(b) !b$categorical, logical(1))))

  # Inconsistent output_info also falls back rather than erroring.
  bl2 <- output_info_blocks(list(list(5, "softmax")), 3)
  expect_false(bl2$ok)
  expect_equal(length(bl2$blocks), 3)
})

test_that("augment_batch keeps the shape and is a no-op for p = 0", {
  torch::torch_manual_seed(1)
  x <- torch::torch_randn(c(16, 5))
  y <- augment_batch(x, p = 0, ops = c("mask", "mixup", "swap"),
                     output_info = list(list(2, "linear"), list(3, "softmax")))
  expect_equal(y$shape, x$shape)
  expect_true(as.logical(torch::torch_equal(y, x)))
})

test_that("augment_batch with p = 1 and mask returns zeros", {
  torch::torch_manual_seed(2)
  x <- torch::torch_randn(c(12, 4)) + 5
  y <- augment_batch(x, p = 1, ops = "mask")
  expect_equal(as.numeric(torch::as_array(y$abs()$sum()$cpu())), 0)
})

test_that("mixup keeps every value inside the convex hull of the column", {
  torch::torch_manual_seed(3)
  x <- torch::torch_randn(c(64, 3))
  y <- augment_batch(x, p = 1, ops = "mixup")
  xa <- torch::as_array(x)
  ya <- torch::as_array(y)
  expect_equal(dim(ya), dim(xa))
  for (j in seq_len(ncol(xa))) {
    expect_true(all(ya[, j] >= min(xa[, j]) - 1e-5))
    expect_true(all(ya[, j] <= max(xa[, j]) + 1e-5))
  }
  # Something actually happened.
  expect_true(max(abs(ya - xa)) > 0)
})

test_that("mixup leaves categorical blocks untouched", {
  torch::torch_manual_seed(4)
  oi <- list(list(2, "linear"), list(3, "softmax"))
  cats <- matrix(0, nrow = 20, ncol = 3)
  cats[cbind(1:20, sample(1:3, 20, replace = TRUE))] <- 1
  x <- torch::torch_tensor(cbind(matrix(rnorm(40), ncol = 2), cats))
  y <- augment_batch(x, p = 1, ops = "mixup", output_info = oi)
  expect_true(as.logical(torch::torch_equal(y[, 3:5], x[, 3:5])))
  expect_false(as.logical(torch::torch_equal(y[, 1:2], x[, 1:2])))
})

test_that("swap preserves the multiset of rows in a categorical block", {
  torch::torch_manual_seed(5)
  oi <- list(list(1, "linear"), list(4, "softmax"))
  cats <- matrix(0, nrow = 40, ncol = 4)
  cats[cbind(1:40, sample(1:4, 40, replace = TRUE))] <- 1
  x <- torch::torch_tensor(cbind(matrix(rnorm(40), ncol = 1), cats))
  y <- augment_batch(x, p = 1, ops = "swap", output_info = oi)

  block_rows <- function(t) {
    a <- torch::as_array(t)[, 2:5, drop = FALSE]
    sort(apply(a, 1, paste, collapse = "-"))
  }
  expect_identical(block_rows(y), block_rows(x))
  # The numeric column is never touched by swap.
  expect_true(as.logical(torch::torch_equal(y[, 1], x[, 1])))
  # And with p = 1 some rows really did change.
  expect_true(max(abs(torch::as_array(y) - torch::as_array(x))) > 0)
})

test_that("augment_batch is deterministic given torch_manual_seed", {
  oi <- list(list(2, "linear"), list(3, "softmax"))
  x <- torch::torch_randn(c(24, 5))
  torch::torch_manual_seed(99)
  y1 <- augment_batch(x, p = 0.5, ops = c("mask", "mixup", "swap"), output_info = oi)
  torch::torch_manual_seed(99)
  y2 <- augment_batch(x, p = 0.5, ops = c("mask", "mixup", "swap"), output_info = oi)
  expect_true(as.logical(torch::torch_equal(y1, y2)))
})

test_that("augment_batch accepts an explicit torch generator", {
  oi <- list(list(2, "linear"), list(3, "softmax"))
  x <- torch::torch_randn(c(16, 5))
  g1 <- torch::torch_generator(); g1$set_current_seed(7L)
  g2 <- torch::torch_generator(); g2$set_current_seed(7L)
  y1 <- augment_batch(x, p = 0.5, ops = c("mask", "mixup"), output_info = oi,
                      generator = g1)
  y2 <- augment_batch(x, p = 0.5, ops = c("mask", "mixup"), output_info = oi,
                      generator = g2)
  expect_true(as.logical(torch::torch_equal(y1, y2)))
})

test_that("gradients flow through augment_batch", {
  torch::torch_manual_seed(11)
  oi <- list(list(2, "linear"), list(3, "softmax"))
  z <- torch::torch_randn(c(16, 4), requires_grad = TRUE)
  lin <- torch::nn_linear(4, 5)
  fake <- lin(z)
  aug <- augment_batch(fake, p = 0.5, ops = c("mask", "mixup", "swap"),
                       output_info = oi)
  loss <- aug$pow(2)$mean()
  loss$backward()
  expect_false(is.null(z$grad))
  expect_true(as.numeric(torch::as_array(z$grad$abs()$sum()$cpu())) > 0)
})

test_that("resolve_augmentation fills in defaults and validates", {
  expect_null(resolve_augmentation(NULL))
  a <- resolve_augmentation(list())
  expect_equal(a$p_max, 0.5)
  expect_true(a$adaptive)
  expect_equal(a$target_rd, 0.6)
  expect_equal(a$ops, c("mask", "mixup", "swap"))
  expect_equal(a$update_every, 4L)
  expect_error(resolve_augmentation(list(ops = "nope")), "ops")
  expect_error(resolve_augmentation(list(p_max = 2)), "p_max")
  expect_error(resolve_augmentation(list(update_every = 0)), "update_every")
  expect_error(resolve_augmentation("x"), "must be NULL or a list")
})

# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------

mixed_mode_specific_data <- function(n = 200) {
  set.seed(20)
  data.frame(
    num1 = c(stats::rnorm(n / 2, -3), stats::rnorm(n / 2, 3)),
    num2 = stats::rnorm(n),
    cat1 = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    cat2 = factor(sample(c("x", "y"), n, replace = TRUE))
  )
}

test_that("gan_trainer runs with augmentation on toy data", {
  d <- sample_toydata(n = 120)
  tr <- data_transformer$new()
  tr$fit(d)
  td <- tr$transform(d)
  g <- gan_trainer(td, epochs = 2, batch_size = 40, noise_dim = 2,
                   track_loss = TRUE, seed = 1,
                   augmentation = list(p_max = 0.5, update_every = 1))
  expect_s3_class(g, "trained_RGAN")
  expect_true(length(g$losses$aug_p) > 0)
  expect_true(all(g$losses$aug_p >= 0 & g$losses$aug_p <= 0.5))
  expect_true(g$settings$aug_p_final >= 0 && g$settings$aug_p_final <= 0.5)
})

test_that("gan_trainer runs with augmentation on mode_specific + onehot data", {
  df <- mixed_mode_specific_data(200)
  tr <- data_transformer$new()
  tr$fit(df, discrete_columns = c("cat1", "cat2"),
         mode_specific = TRUE, continuous_transform = "mode_specific", n_modes = 3)
  td <- tr$transform(df)
  g <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 4,
                   output_info = tr$output_info, seed = 2, track_loss = TRUE,
                   augmentation = list(p_max = 0.5, update_every = 1))
  expect_s3_class(g, "trained_RGAN")
  syn <- sample_synthetic_data(g, tr)
  expect_equal(ncol(syn), 4)
})

test_that("gan_trainer runs with augmentation on quantile + cossin data", {
  df <- mixed_mode_specific_data(200)
  tr <- data_transformer$new()
  tr$fit(df, discrete_columns = c("cat1", "cat2"),
         continuous_transform = "quantile", categorical_encoding = "cossin")
  td <- tr$transform(df)
  g <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 4,
                   output_info = tr$output_info, seed = 3, track_loss = TRUE,
                   augmentation = list(p_max = 0.5, update_every = 1))
  expect_s3_class(g, "trained_RGAN")
  syn <- sample_synthetic_data(g, tr)
  expect_equal(ncol(syn), 4)
})

test_that("augmentation works with pac > 1 and with rpgan + r1r2", {
  d <- sample_toydata(n = 200)
  tr <- data_transformer$new()
  tr$fit(d)
  td <- tr$transform(d)
  g <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2, pac = 10,
                   seed = 4, track_loss = TRUE,
                   augmentation = list(p_max = 0.4, update_every = 2))
  expect_s3_class(g, "trained_RGAN")

  g2 <- gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2,
                    value_function = "rpgan",
                    regularization = list(type = "r1r2", gamma = 10),
                    seed = 5, track_loss = TRUE,
                    augmentation = list(p_max = 0.4, update_every = 2))
  expect_s3_class(g2, "trained_RGAN")
  expect_false(any(is.na(g2$losses$d_loss)))
})

test_that("augmentation = NULL is a no-op (bit-identical training)", {
  d <- sample_toydata(n = 200)
  tr <- data_transformer$new()
  tr$fit(d)
  td <- tr$transform(d)
  run <- function(...) {
    set.seed(1)
    torch::torch_manual_seed(1)
    gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 2, seed = 42,
                track_loss = TRUE, ...)
  }
  a <- run()
  b <- run(augmentation = NULL, instance_noise = NULL)
  expect_identical(a$losses$g_loss, b$losses$g_loss)
  expect_identical(a$losses$d_loss, b$losses$d_loss)
  expect_null(a$losses$aug_p)
})
