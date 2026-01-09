test_that("gan_trainer rejects invalid batch_size", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, batch_size = 0, epochs = 1),
    "batch_size must be a positive integer"
  )

  expect_error(
    gan_trainer(transformed_data, batch_size = -10, epochs = 1),
    "batch_size must be a positive integer"
  )
})

test_that("gan_trainer rejects invalid epochs", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, epochs = 0, batch_size = 20),
    "epochs must be a positive integer"
  )

  expect_error(
    gan_trainer(transformed_data, epochs = -5, batch_size = 20),
    "epochs must be a positive integer"
  )
})

test_that("gan_trainer rejects invalid base_lr", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, base_lr = 0, epochs = 1, batch_size = 20),
    "base_lr must be a positive number"
  )

  expect_error(
    gan_trainer(transformed_data, base_lr = -0.001, epochs = 1, batch_size = 20),
    "base_lr must be a positive number"
  )
})

test_that("gan_trainer rejects invalid noise_dim", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, noise_dim = 0, epochs = 1, batch_size = 20),
    "noise_dim must be a positive integer"
  )
})

test_that("gan_trainer warns when batch_size exceeds data size", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 30)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_warning(
    gan_trainer(transformed_data, batch_size = 100, epochs = 1, seed = 123),
    "batch_size.*larger than number of observations"
  )
})

test_that("gan_trainer rejects empty data", {
  skip_if_not_installed("torch")

  empty_data <- matrix(numeric(0), nrow = 0, ncol = 2)

  expect_error(
    gan_trainer(empty_data, epochs = 1, batch_size = 10),
    "data cannot be empty"
  )
})

test_that("gan_trainer rejects data with no columns", {
  skip_if_not_installed("torch")

  # Suppress warning about zero-extent matrix
  bad_data <- suppressWarnings(matrix(numeric(0), nrow = 10, ncol = 0))

  expect_error(
    gan_trainer(bad_data, epochs = 1, batch_size = 5),
    "data must have at least one column"
  )
})

test_that("gan_trainer rejects all-NA data", {
  skip_if_not_installed("torch")

  na_data <- matrix(NA, nrow = 10, ncol = 2)

  expect_error(
    gan_trainer(na_data, epochs = 1, batch_size = 5),
    "data cannot be all NA"
  )
})

test_that("gan_trainer validates plot_dimensions when plot_progress is TRUE", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Invalid plot_dimensions (column doesn't exist)
  expect_error(
    gan_trainer(
      transformed_data,
      epochs = 1,
      batch_size = 20,
      plot_progress = TRUE,
      plot_dimensions = c(1, 5)
    ),
    "plot_dimensions must be between 1 and"
  )

  # Invalid plot_dimensions (wrong length)
  expect_error(
    gan_trainer(
      transformed_data,
      epochs = 1,
      batch_size = 20,
      plot_progress = TRUE,
      plot_dimensions = c(1)
    ),
    "plot_dimensions must be a vector of length 2"
  )
})

test_that("Discriminator uses LeakyReLU activation", {
  skip_if_not_installed("torch")

  d_net <- Discriminator(data_dim = 5, hidden_units = list(32), dropout_rate = 0)

  # Check that LeakyReLU modules are present
  module_names <- names(d_net$seq$modules)
  activation_modules <- grep("Activation", module_names, value = TRUE)

  expect_true(length(activation_modules) > 0)

  # The activation should be leaky_relu, not regular relu
  # We can verify by checking the module class
  activation_module <- d_net$seq$modules[[activation_modules[1]]]
  expect_true(inherits(activation_module, "nn_leaky_relu"))
})
