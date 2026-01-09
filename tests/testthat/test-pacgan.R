test_that("gan_trainer accepts pac parameter", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Test with pac=2

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    pac = 2,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$pac, 2)
})

test_that("gan_trainer works with pac=10", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # batch_size must be divisible by pac
  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    pac = 10,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$pac, 10)
})

test_that("gan_trainer rejects invalid pac", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # pac must be positive integer
  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, pac = 0),
    "pac must be a positive integer"
  )

  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, pac = -1),
    "pac must be a positive integer"
  )

  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, pac = 2.5),
    "pac must be a positive integer"
  )
})

test_that("gan_trainer rejects batch_size not divisible by pac", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # batch_size=20 is not divisible by pac=3
  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 20, pac = 3),
    "batch_size.*must be divisible by pac"
  )

  # batch_size=25 is not divisible by pac=10
  expect_error(
    gan_trainer(transformed_data, epochs = 1, batch_size = 25, pac = 10),
    "batch_size.*must be divisible by pac"
  )
})

test_that("pack_samples correctly reshapes tensor", {
  skip_if_not_installed("torch")

  # Create a test tensor: 10 samples, 4 features
  data <- torch::torch_randn(c(10, 4))

  # Pack with pac=2: should become 5 samples, 8 features
  packed <- RGAN:::pack_samples(data, 2)

  expect_equal(packed$shape[1], 5)
  expect_equal(packed$shape[2], 8)

  # Pack with pac=5: should become 2 samples, 20 features
  packed5 <- RGAN:::pack_samples(data, 5)

  expect_equal(packed5$shape[1], 2)
  expect_equal(packed5$shape[2], 20)

  # Pack with pac=1: should return unchanged
  packed1 <- RGAN:::pack_samples(data, 1)

  expect_equal(packed1$shape[1], 10)
  expect_equal(packed1$shape[2], 4)
})

test_that("pack_samples preserves data correctly", {
  skip_if_not_installed("torch")

  # Create a simple tensor with known values
  data <- torch::torch_tensor(matrix(1:12, nrow = 4, ncol = 3, byrow = TRUE))
  # data is:
  # [1, 2, 3]
  # [4, 5, 6]
  # [7, 8, 9]
  # [10, 11, 12]

  # Pack with pac=2
  packed <- RGAN:::pack_samples(data, 2)

  # Should be:
  # [1, 2, 3, 4, 5, 6]
  # [7, 8, 9, 10, 11, 12]
  expect_equal(packed$shape[1], 2)
  expect_equal(packed$shape[2], 6)

  packed_array <- torch::as_array(packed)
  expect_equal(packed_array[1, 1], 1)
  expect_equal(packed_array[1, 4], 4)
  expect_equal(packed_array[2, 1], 7)
  expect_equal(packed_array[2, 6], 12)
})

test_that("discriminator with pac has correct input dimension", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # With pac=5, discriminator should accept 2*5=10 input features
  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    pac = 5,
    seed = 123
  )

  # The discriminator should accept input of size data_dim * pac = 2 * 5 = 10
  # Test by creating packed input and passing through discriminator
  test_input <- torch::torch_randn(c(4, 10))  # 4 packed samples, 10 features
  output <- result$discriminator(test_input)

  expect_equal(output$shape[1], 4)
  expect_equal(output$shape[2], 1)
})

test_that("pac=1 produces same results as default", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Train with explicit pac=1
  result1 <- gan_trainer(
    transformed_data,
    epochs = 3,
    batch_size = 20,
    pac = 1,
    track_loss = TRUE,
    seed = 123
  )

  # Train without specifying pac (default)
  result_default <- gan_trainer(
    transformed_data,
    epochs = 3,
    batch_size = 20,
    track_loss = TRUE,
    seed = 123
  )

  # Both should have the same settings
  expect_equal(result1$settings$pac, 1)
  expect_equal(result_default$settings$pac, 1)

  # And produce same losses (due to same seed)
  expect_equal(result1$losses$g_loss, result_default$losses$g_loss)
})

test_that("PacGAN works with WGAN-GP", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Test PacGAN with WGAN-GP
  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    value_function = "wgan-gp",
    gp_lambda = 10,
    pac = 5,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$pac, 5)
  expect_equal(result$settings$value_function, "wgan-gp")
})

test_that("print.trained_RGAN shows pac when > 1", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    pac = 5,
    seed = 123
  )

  output <- capture.output(print(result))
  expect_true(any(grepl("PacGAN", output)))
  expect_true(any(grepl("pac=5", output)))
})

test_that("print.trained_RGAN does not show pac when = 1", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    pac = 1,
    seed = 123
  )

  output <- capture.output(print(result))
  expect_false(any(grepl("PacGAN", output)))
})

test_that("PacGAN works with validation data and early stopping", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 200)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Split into train/validation
  train_data <- transformed_data[1:160, ]
  val_data <- transformed_data[161:200, ]

  # Test PacGAN with validation data
  result <- gan_trainer(
    train_data,
    epochs = 3,
    batch_size = 20,
    pac = 10,
    validation_data = val_data,
    early_stopping = TRUE,
    patience = 5,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$pac, 10)
})

test_that("PacGAN works with TabularGenerator and validation", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 200)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Split into train/validation
  train_data <- transformed_data[1:160, ]
  val_data <- transformed_data[161:200, ]

  # Test PacGAN with TabularGenerator (output_info) and validation
  result <- gan_trainer(
    train_data,
    epochs = 3,
    batch_size = 20,
    pac = 10,
    output_info = transformer$output_info,
    validation_data = val_data,
    early_stopping = TRUE,
    patience = 5,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$pac, 10)

  # Verify we can sample synthetic data
  synthetic <- sample_synthetic_data(result, transformer, n = 50)
  expect_equal(nrow(synthetic), 50)
})
