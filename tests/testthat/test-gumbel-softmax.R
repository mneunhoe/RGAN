test_that("gumbel_softmax produces valid probability distribution", {
  skip_if_not_installed("torch")

  logits <- torch::torch_randn(c(10, 5))
  soft_samples <- gumbel_softmax(logits, tau = 1.0, hard = FALSE)

  # Output should be same shape
  expect_equal(soft_samples$shape[1], 10)
  expect_equal(soft_samples$shape[2], 5)

  # Each row should sum to 1 (valid probability distribution)
  row_sums <- torch::as_array(soft_samples$sum(dim = 2))
  expect_true(all(abs(row_sums - 1) < 1e-5))

  # All values should be between 0 and 1
  expect_true(all(torch::as_array(soft_samples) >= 0))
  expect_true(all(torch::as_array(soft_samples) <= 1))
})

test_that("gumbel_softmax hard mode produces one-hot vectors", {
  skip_if_not_installed("torch")

  logits <- torch::torch_randn(c(10, 5))
  hard_samples <- gumbel_softmax(logits, tau = 0.5, hard = TRUE)

  hard_array <- torch::as_array(hard_samples)

  # Each row should have exactly one 1 and rest 0s
  for (i in 1:10) {
    row <- hard_array[i, ]
    expect_equal(sum(row), 1)
    expect_equal(sum(row == 1), 1)
    expect_equal(sum(row == 0), 4)
  }
})

test_that("gumbel_softmax lower temperature produces sharper distributions", {
  skip_if_not_installed("torch")

  torch::torch_manual_seed(123)
  logits <- torch::torch_randn(c(100, 5))

  # High temperature - more uniform
  soft_high <- gumbel_softmax(logits, tau = 2.0, hard = FALSE)
  # Low temperature - more peaked
  soft_low <- gumbel_softmax(logits, tau = 0.1, hard = FALSE)

  # Lower temperature should have higher max probability on average
  max_high <- torch::as_array(soft_high$max(dim = 2)[[1]]$mean())
  max_low <- torch::as_array(soft_low$max(dim = 2)[[1]]$mean())

  expect_gt(max_low, max_high)
})

test_that("gumbel_softmax has gradients in soft mode", {
  skip_if_not_installed("torch")

  logits <- torch::torch_randn(c(5, 3), requires_grad = TRUE)
  soft_samples <- gumbel_softmax(logits, tau = 1.0, hard = FALSE)

  # Compute a simple loss and backpropagate

  loss <- soft_samples$sum()
  loss$backward()

  # Gradients should exist
  expect_true(!is.null(logits$grad))
  expect_true(any(torch::as_array(logits$grad) != 0))
})

test_that("gumbel_softmax hard mode has gradients (straight-through)", {
  skip_if_not_installed("torch")

  logits <- torch::torch_randn(c(5, 3), requires_grad = TRUE)
  hard_samples <- gumbel_softmax(logits, tau = 1.0, hard = TRUE)

  # For straight-through to show gradients, we need a loss that
  # depends on the continuous soft values. The trick is that hard_samples
  # uses soft values in the backward pass. Multiply by weights to get varied gradients.
  weights <- torch::torch_randn(c(5, 3))
  loss <- (hard_samples * weights)$sum()
  loss$backward()

  # Gradients should exist (via straight-through estimator)
  expect_true(!is.null(logits$grad))
  # With varied weights, some gradients should be non-zero
  expect_true(any(torch::as_array(logits$grad) != 0))
})

test_that("TabularGenerator creates valid architecture", {
  skip_if_not_installed("torch")

  output_info <- list(
    list(1, "linear"),      # continuous column
    list(3, "softmax"),     # categorical with 3 classes
    list(1, "linear")       # another continuous
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    hidden_units = list(64, 64),
    tau = 0.5
  )

  # Test forward pass
  z <- torch::torch_randn(c(5, 10))
  output <- gen(z)

  # Output should have correct dimensions: 1 + 3 + 1 = 5
  expect_equal(output$shape[1], 5)
  expect_equal(output$shape[2], 5)
})

test_that("TabularGenerator applies correct activations", {
  skip_if_not_installed("torch")

  output_info <- list(
    list(1, "linear"),      # continuous: tanh
    list(3, "softmax")      # categorical: gumbel-softmax
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    hidden_units = list(64),
    tau = 0.5
  )

  z <- torch::torch_randn(c(10, 10))
  output <- gen(z, hard = FALSE)

  output_array <- torch::as_array(output)

  # First column (linear/tanh) should be in [-1, 1]
  expect_true(all(output_array[, 1] >= -1))
  expect_true(all(output_array[, 1] <= 1))

  # Columns 2-4 (softmax) should sum to 1 for each row
  softmax_sums <- rowSums(output_array[, 2:4])
  expect_true(all(abs(softmax_sums - 1) < 1e-5))
})

test_that("TabularGenerator hard mode produces one-hot categoricals", {
  skip_if_not_installed("torch")

  output_info <- list(
    list(1, "linear"),
    list(4, "softmax")
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    tau = 0.5
  )
  gen$eval()  # Set to eval mode

  z <- torch::torch_randn(c(10, 10))
  output <- gen(z)  # Should use hard=TRUE in eval mode

  output_array <- torch::as_array(output)

  # Categorical columns should be one-hot
  for (i in 1:10) {
    cat_row <- output_array[i, 2:5]
    expect_equal(sum(cat_row), 1)
    expect_equal(sum(cat_row == 1), 1)
  }
})

test_that("TabularGenerator handles mode_specific columns", {
  skip_if_not_installed("torch")

  # mode_specific: 3 mode indicators + 1 value = 4 dimensions
  output_info <- list(
    list(4, "mode_specific"),
    list(2, "softmax")
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    tau = 0.5
  )

  z <- torch::torch_randn(c(10, 10))
  output <- gen(z, hard = FALSE)

  output_array <- torch::as_array(output)

  # Output should have 4 + 2 = 6 columns
  expect_equal(ncol(output_array), 6)

  # First 3 columns (mode indicators) should sum to ~1
  mode_sums <- rowSums(output_array[, 1:3])
  expect_true(all(abs(mode_sums - 1) < 1e-5))

  # 4th column (value) should be in [-1, 1]
  expect_true(all(output_array[, 4] >= -1))
  expect_true(all(output_array[, 4] <= 1))

  # Last 2 columns (softmax) should sum to ~1
  softmax_sums <- rowSums(output_array[, 5:6])
  expect_true(all(abs(softmax_sums - 1) < 1e-5))
})

test_that("gan_trainer accepts output_info parameter", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    output_info = transformer$output_info,
    gumbel_tau = 0.5,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$gumbel_tau, 0.5)
  expect_equal(result$settings$output_info, transformer$output_info)
})

test_that("gan_trainer validates output_info structure", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Invalid output_info - not a list
  expect_error(
    gan_trainer(transformed_data, epochs = 1, output_info = "invalid"),
    "output_info must be a list"
  )

  # Invalid output_info element
  expect_error(
    gan_trainer(transformed_data, epochs = 1, output_info = list(c(1, 2))),
    "must be a list with at least 2 elements"
  )

  # Invalid type
  expect_error(
    gan_trainer(transformed_data, epochs = 1, output_info = list(list(1, "invalid"))),
    "output_info type must be one of"
  )
})

test_that("gan_trainer validates gumbel_tau", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, epochs = 1, gumbel_tau = 0),
    "gumbel_tau must be a positive number"
  )

  expect_error(
    gan_trainer(transformed_data, epochs = 1, gumbel_tau = -1),
    "gumbel_tau must be a positive number"
  )
})

test_that("gan_trainer with output_info uses TabularGenerator", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    output_info = transformer$output_info,
    seed = 123
  )

  # Check that generator has output_info attribute (TabularGenerator specific)
  expect_true(!is.null(result$generator$output_info))
  expect_true(!is.null(result$generator$tau))
})

test_that("sample_synthetic_data works with TabularGenerator", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    output_info = transformer$output_info,
    seed = 123
  )

  # Sample synthetic data
  synthetic <- sample_synthetic_data(trained_gan, transformer, n = 50)

  expect_equal(nrow(synthetic), 50)
  expect_equal(ncol(synthetic), 2)
})

test_that("TabularGenerator with categorical data produces valid samples", {
  skip_if_not_installed("torch")

  # Create data with a categorical column using numeric values
  data <- sample_toydata(n = 100)
  data <- data.frame(data, category = sample(1:3, 100, replace = TRUE))

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = "category")
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 3,
    batch_size = 20,
    output_info = transformer$output_info,
    gumbel_tau = 0.5,
    seed = 123
  )

  # Sample synthetic data
  synthetic <- sample_synthetic_data(trained_gan, transformer, n = 50)

  expect_equal(nrow(synthetic), 50)
  expect_equal(ncol(synthetic), 3)

  # Category column should only contain valid levels (1, 2, or 3)
  expect_true(all(synthetic[, 3] %in% c(1, 2, 3)))
})

test_that("print.trained_RGAN shows Gumbel-Softmax info", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    output_info = transformer$output_info,
    gumbel_tau = 0.3,
    seed = 123
  )

  output <- capture.output(print(result))
  expect_true(any(grepl("Gumbel-Softmax", output)))
  expect_true(any(grepl("tau=0.30", output)))
})

test_that("Gumbel-Softmax with mode_specific normalization works", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    output_info = transformer$output_info,
    gumbel_tau = 0.5,
    seed = 123
  )

  synthetic <- sample_synthetic_data(trained_gan, transformer, n = 50)

  expect_equal(nrow(synthetic), 50)
  expect_equal(ncol(synthetic), 2)
})

test_that("TabularGenerator training mode affects Gumbel-Softmax behavior", {
  skip_if_not_installed("torch")

  output_info <- list(
    list(3, "softmax")
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    tau = 0.5
  )

  torch::torch_manual_seed(42)
  z <- torch::torch_randn(c(10, 10))

  # Training mode: soft samples
  gen$train()
  train_output <- torch::as_array(gen(z))

  # Eval mode: hard samples
  gen$eval()
  eval_output <- torch::as_array(gen(z))

  # In training mode, values should be soft (between 0 and 1)
  expect_true(all(train_output > 0 & train_output < 1))

  # In eval mode, should be hard (only 0s and 1s)
  for (i in 1:nrow(eval_output)) {
    expect_equal(sum(eval_output[i, ] == 1), 1)
  }
})
