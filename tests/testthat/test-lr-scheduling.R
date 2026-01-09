test_that("gan_trainer accepts lr_schedule parameter", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Test with step schedule
  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    lr_schedule = "step",
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$lr_schedule, "step")
})

test_that("gan_trainer rejects invalid lr_schedule", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, lr_schedule = "invalid"),
    "lr_schedule must be one of"
  )
})

test_that("gan_trainer rejects invalid lr_decay_factor", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, lr_decay_factor = 0),
    "lr_decay_factor must be between 0"
  )

  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, lr_decay_factor = 1.5),
    "lr_decay_factor must be between 0"
  )

  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, lr_decay_factor = -0.1),
    "lr_decay_factor must be between 0"
  )
})

test_that("gan_trainer rejects invalid lr_decay_steps", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, lr_decay_steps = 0),
    "lr_decay_steps must be a positive integer"
  )

  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, lr_decay_steps = -10),
    "lr_decay_steps must be a positive integer"
  )
})

test_that("step schedule reduces learning rate", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Train with step schedule that should trigger at least one decay
  result <- gan_trainer(
    transformed_data,
    epochs = 6,
    batch_size = 20,
    base_lr = 0.001,
    lr_schedule = "step",
    lr_decay_factor = 0.5,
    lr_decay_steps = 3,
    seed = 123
  )

  # After 6 epochs with decay every 3 epochs, we should have decayed twice
  # Final LR should be approximately 0.001 * 0.5^2 = 0.00025
  final_g_lr <- result$generator_optimizer$param_groups[[1]]$lr

  # Check that LR has decreased from initial
  expect_true(final_g_lr < 0.001)
  # Check it's approximately the expected value (0.00025)
  expect_equal(final_g_lr, 0.00025, tolerance = 1e-6)
})

test_that("exponential schedule reduces learning rate each epoch", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 5,
    batch_size = 20,
    base_lr = 0.001,
    lr_schedule = "exponential",
    lr_decay_factor = 0.9,
    seed = 123
  )

  # After 5 epochs with 0.9 decay each epoch: 0.001 * 0.9^4 = 0.0006561
  # (decay is applied at end of epoch, so after epoch 5 we've applied 4 decays)
  final_g_lr <- result$generator_optimizer$param_groups[[1]]$lr

  expect_true(final_g_lr < 0.001)
  expect_equal(final_g_lr, 0.001 * 0.9^4, tolerance = 1e-6)
})

test_that("cosine schedule reduces learning rate", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 10,
    batch_size = 20,
    base_lr = 0.001,
    lr_schedule = "cosine",
    seed = 123
  )

  # After all epochs, cosine schedule should have reduced LR significantly
  final_g_lr <- result$generator_optimizer$param_groups[[1]]$lr

  # Cosine at epoch 10/10: 0.001 * (1 + cos(pi * 10/10)) / 2 = 0.001 * 0 / 2 = 0
  # Should be very close to 0
  expect_true(final_g_lr < 0.0001)
})

test_that("constant schedule does not change learning rate", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 5,
    batch_size = 20,
    base_lr = 0.001,
    lr_schedule = "constant",
    seed = 123
  )

  # LR should remain unchanged
  final_g_lr <- result$generator_optimizer$param_groups[[1]]$lr

  expect_equal(final_g_lr, 0.001, tolerance = 1e-8)
})

test_that("adjust_learning_rate helper function works correctly", {
  skip_if_not_installed("torch")

  # Create a simple optimizer
  net <- torch::nn_linear(2, 1)
  optim <- torch::optim_adam(net$parameters, lr = 0.001)

  # Test step schedule
  RGAN:::adjust_learning_rate(optim, 0.001, 1, 10, "step", 0.5, 5)
  expect_equal(optim$param_groups[[1]]$lr, 0.001)  # No decay yet

  RGAN:::adjust_learning_rate(optim, 0.001, 5, 10, "step", 0.5, 5)
  expect_equal(optim$param_groups[[1]]$lr, 0.0005)  # One decay

  RGAN:::adjust_learning_rate(optim, 0.001, 10, 10, "step", 0.5, 5)
  expect_equal(optim$param_groups[[1]]$lr, 0.00025)  # Two decays

  # Test exponential schedule
  RGAN:::adjust_learning_rate(optim, 0.001, 1, 10, "exponential", 0.9, 5)
  expect_equal(optim$param_groups[[1]]$lr, 0.001)  # First epoch, no decay yet

  RGAN:::adjust_learning_rate(optim, 0.001, 3, 10, "exponential", 0.9, 5)
  expect_equal(optim$param_groups[[1]]$lr, 0.001 * 0.9^2, tolerance = 1e-8)

  # Test cosine schedule
  RGAN:::adjust_learning_rate(optim, 0.001, 5, 10, "cosine", 0.5, 5)
  expected_lr <- 0.001 * (1 + cos(pi * 5 / 10)) / 2
  expect_equal(optim$param_groups[[1]]$lr, expected_lr, tolerance = 1e-8)

  # Test constant schedule
  RGAN:::adjust_learning_rate(optim, 0.001, 5, 10, "constant", 0.5, 5)
  expect_equal(optim$param_groups[[1]]$lr, 0.001)
})

test_that("lr_schedule settings are stored in output", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    lr_schedule = "step",
    lr_decay_factor = 0.5,
    lr_decay_steps = 10,
    seed = 123
  )

  expect_equal(result$settings$lr_schedule, "step")
  expect_equal(result$settings$lr_decay_factor, 0.5)
  expect_equal(result$settings$lr_decay_steps, 10)
})

test_that("print.trained_RGAN shows lr_schedule when not constant", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    lr_schedule = "step",
    lr_decay_factor = 0.5,
    lr_decay_steps = 10,
    seed = 123
  )

  output <- capture.output(print(result))
  expect_true(any(grepl("LR schedule", output)))
  expect_true(any(grepl("step", output)))
})
