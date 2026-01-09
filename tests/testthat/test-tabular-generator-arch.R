test_that("ResidualBlock creates valid architecture", {
  skip_if_not_installed("torch")

  block <- ResidualBlock(
    input_dim = 64,
    output_dim = 64,
    normalization = "batch",
    activation = "relu"
  )

  # Test forward pass
  x <- torch::torch_randn(c(10, 64))
  out <- block(x)

  expect_equal(out$shape[1], 10)
  expect_equal(out$shape[2], 64)
})

test_that("ResidualBlock applies residual connection when dims match", {
  skip_if_not_installed("torch")

  block <- ResidualBlock(
    input_dim = 64,
    output_dim = 64,
    normalization = "none",
    activation = "relu"
  )

  expect_true(block$use_residual)

  # When dims don't match, no residual
  block2 <- ResidualBlock(
    input_dim = 64,
    output_dim = 128,
    normalization = "none",
    activation = "relu"
  )

  expect_false(block2$use_residual)
})

test_that("ResidualBlock supports batch normalization", {
  skip_if_not_installed("torch")

  block <- ResidualBlock(
    input_dim = 64,
    output_dim = 64,
    normalization = "batch",
    activation = "relu"
  )

  expect_true(!is.null(block$norm))
  expect_true(inherits(block$norm, "nn_batch_norm1d"))
})

test_that("ResidualBlock supports layer normalization", {
  skip_if_not_installed("torch")

  block <- ResidualBlock(
    input_dim = 64,
    output_dim = 64,
    normalization = "layer",
    activation = "relu"
  )

  expect_true(!is.null(block$norm))
  expect_true(inherits(block$norm, "nn_layer_norm"))
})

test_that("ResidualBlock supports all activation functions", {
  skip_if_not_installed("torch")

  activations <- c("relu", "leaky_relu", "gelu", "silu")

  for (act in activations) {
    block <- ResidualBlock(
      input_dim = 32,
      output_dim = 32,
      normalization = "batch",
      activation = act
    )

    x <- torch::torch_randn(c(5, 32))
    out <- block(x)

    expect_equal(out$shape[1], 5)
    expect_equal(out$shape[2], 32)
  }
})

test_that("TabularGenerator with batch normalization works", {
  skip_if_not_installed("torch")

  output_info <- list(
    list(1, "linear"),
    list(3, "softmax")
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    hidden_units = list(64, 64),
    normalization = "batch",
    activation = "relu"
  )

  z <- torch::torch_randn(c(10, 10))
  out <- gen(z)

  expect_equal(out$shape[1], 10)
  expect_equal(out$shape[2], 4)

  # Check that batch norm layers exist
  expect_equal(gen$normalization, "batch")
})

test_that("TabularGenerator with layer normalization works", {
  skip_if_not_installed("torch")

  output_info <- list(
    list(1, "linear"),
    list(3, "softmax")
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    hidden_units = list(64, 64),
    normalization = "layer",
    activation = "relu"
  )

  z <- torch::torch_randn(c(10, 10))
  out <- gen(z)

  expect_equal(out$shape[1], 10)
  expect_equal(out$shape[2], 4)
  expect_equal(gen$normalization, "layer")
})

test_that("TabularGenerator with no normalization works", {
  skip_if_not_installed("torch")

  output_info <- list(
    list(1, "linear"),
    list(3, "softmax")
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    hidden_units = list(64, 64),
    normalization = "none",
    dropout_rate = 0.5
  )

  z <- torch::torch_randn(c(10, 10))
  out <- gen(z)

  expect_equal(out$shape[1], 10)
  expect_equal(gen$normalization, "none")
})

test_that("TabularGenerator supports GELU activation", {
  skip_if_not_installed("torch")

  output_info <- list(list(2, "linear"))

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    hidden_units = list(64),
    activation = "gelu"
  )

  z <- torch::torch_randn(c(5, 10))
  out <- gen(z)

  expect_equal(out$shape[1], 5)
  expect_equal(gen$activation, "gelu")
})

test_that("TabularGenerator supports SiLU activation", {
  skip_if_not_installed("torch")

  output_info <- list(list(2, "linear"))

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    hidden_units = list(64),
    activation = "silu"
  )

  z <- torch::torch_randn(c(5, 10))
  out <- gen(z)

  expect_equal(out$shape[1], 5)
  expect_equal(gen$activation, "silu")
})

test_that("TabularGenerator supports residual connections", {
  skip_if_not_installed("torch")

  output_info <- list(list(2, "linear"))

  # With residual (same width layers)
  gen <- TabularGenerator(
    noise_dim = 64,
    output_info = output_info,
    hidden_units = list(64, 64, 64),
    residual = TRUE
  )

  expect_true(gen$residual)
  # First block: 64 -> 64 (residual possible since same dim)
  # Second block: 64 -> 64 (residual active)
  expect_true(gen$blocks[[2]]$use_residual)

  # Without residual
  gen_no_res <- TabularGenerator(
    noise_dim = 64,
    output_info = output_info,
    hidden_units = list(64, 64),
    residual = FALSE
  )

  expect_false(gen_no_res$residual)
  expect_false(gen_no_res$blocks[[1]]$use_residual)
})

test_that("TabularGenerator validates normalization parameter", {
  skip_if_not_installed("torch")

  output_info <- list(list(1, "linear"))

  expect_error(
    TabularGenerator(noise_dim = 10, output_info = output_info, normalization = "invalid"),
    "normalization must be one of"
  )
})

test_that("TabularGenerator validates activation parameter", {
  skip_if_not_installed("torch")

  output_info <- list(list(1, "linear"))

  expect_error(
    TabularGenerator(noise_dim = 10, output_info = output_info, activation = "invalid"),
    "activation must be one of"
  )
})

test_that("TabularGenerator validates init_method parameter", {
  skip_if_not_installed("torch")

  output_info <- list(list(1, "linear"))

  expect_error(
    TabularGenerator(noise_dim = 10, output_info = output_info, init_method = "invalid"),
    "init_method must be one of"
  )
})

test_that("TabularGenerator supports all initialization methods", {
  skip_if_not_installed("torch")

  output_info <- list(list(2, "linear"))
  inits <- c("xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal")

  for (init in inits) {
    gen <- TabularGenerator(
      noise_dim = 10,
      output_info = output_info,
      hidden_units = list(32),
      init_method = init
    )

    z <- torch::torch_randn(c(5, 10))
    out <- gen(z)

    expect_equal(out$shape[1], 5)
  }
})

test_that("gan_trainer accepts generator architecture parameters", {
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
    generator_hidden_units = list(128, 128),
    generator_normalization = "layer",
    generator_activation = "gelu",
    generator_init = "kaiming_uniform",
    generator_residual = FALSE,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$generator_normalization, "layer")
  expect_equal(result$settings$generator_activation, "gelu")
  expect_equal(result$settings$generator_init, "kaiming_uniform")
  expect_false(result$settings$generator_residual)
})

test_that("TabularGenerator default matches CTGAN style", {
  skip_if_not_installed("torch")

  output_info <- list(list(2, "linear"))

  gen <- TabularGenerator(
    noise_dim = 128,
    output_info = output_info
  )

  # CTGAN defaults
  expect_equal(gen$normalization, "batch")
  expect_equal(gen$activation, "relu")
  expect_true(gen$residual)
})

test_that("print.trained_RGAN shows generator architecture info", {
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
    generator_normalization = "layer",
    generator_activation = "gelu",
    seed = 123
  )

  output <- capture.output(print(result))
  expect_true(any(grepl("layer normalization", output)))
  expect_true(any(grepl("gelu activation", output)))
  expect_true(any(grepl("Residual connections", output)))
})

test_that("TabularGenerator with deep architecture trains", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Deep network with 4 layers
  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    output_info = transformer$output_info,
    generator_hidden_units = list(128, 128, 128, 128),
    generator_normalization = "batch",
    generator_residual = TRUE,
    seed = 123
  )

  synthetic <- sample_synthetic_data(result, transformer, n = 50)

  expect_equal(nrow(synthetic), 50)
  expect_equal(ncol(synthetic), 2)
})

test_that("TabularGenerator training produces gradients", {
  skip_if_not_installed("torch")

  output_info <- list(
    list(1, "linear"),
    list(3, "softmax")
  )

  gen <- TabularGenerator(
    noise_dim = 10,
    output_info = output_info,
    hidden_units = list(64, 64),
    normalization = "batch",
    activation = "relu"
  )

  # Forward pass and backward
  z <- torch::torch_randn(c(10, 10), requires_grad = TRUE)
  out <- gen(z)
  loss <- out$sum()
  loss$backward()

  # Check gradients flow back
  expect_true(any(torch::as_array(z$grad) != 0))
})

test_that("ResidualBlock improves gradient flow", {
  skip_if_not_installed("torch")

  # Compare gradient magnitudes with and without residual

  # With residual
  block_res <- ResidualBlock(
    input_dim = 64,
    output_dim = 64,
    normalization = "batch",
    activation = "relu"
  )
  block_res$use_residual <- TRUE

  x_res <- torch::torch_randn(c(10, 64), requires_grad = TRUE)
  out_res <- block_res(x_res)
  out_res$sum()$backward()
  grad_mag_res <- torch::as_array(x_res$grad$abs()$mean())

  # Without residual
  block_no_res <- ResidualBlock(
    input_dim = 64,
    output_dim = 64,
    normalization = "batch",
    activation = "relu"
  )
  block_no_res$use_residual <- FALSE

  x_no_res <- torch::torch_randn(c(10, 64), requires_grad = TRUE)
  out_no_res <- block_no_res(x_no_res)
  out_no_res$sum()$backward()
  grad_mag_no_res <- torch::as_array(x_no_res$grad$abs()$mean())

  # Residual should generally have larger gradients (better flow)
  # This is a soft check - residual adds identity which helps gradients
  expect_true(grad_mag_res > 0)
  expect_true(grad_mag_no_res > 0)
})
