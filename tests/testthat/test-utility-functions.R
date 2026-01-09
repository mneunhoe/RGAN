test_that("print.trained_RGAN produces output", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = TRUE,
    seed = 123
  )

  # Capture print output
  output <- capture.output(print(trained_gan))

  # Check that key information is present
  expect_true(any(grepl("Trained RGAN Model", output)))
  expect_true(any(grepl("Training Settings", output)))
  expect_true(any(grepl("Value function", output)))
  expect_true(any(grepl("Epochs", output)))
  expect_true(any(grepl("Generator", output)))
  expect_true(any(grepl("Discriminator", output)))
  expect_true(any(grepl("Parameters", output)))
})

test_that("print.trained_RGAN shows losses when tracked", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = TRUE,
    seed = 123
  )

  output <- capture.output(print(trained_gan))

  expect_true(any(grepl("Final Training Losses", output)))
  expect_true(any(grepl("Generator loss", output)))
  expect_true(any(grepl("Discriminator loss", output)))
})

test_that("print.trained_RGAN indicates when losses not tracked", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = FALSE,
    seed = 123
  )

  output <- capture.output(print(trained_gan))

  expect_true(any(grepl("Not tracked", output)))
})

test_that("print.trained_RGAN shows validation metrics when available", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  train_data <- transformed_data[1:80, ]
  val_data <- transformed_data[81:100, ]

  trained_gan <- gan_trainer(
    train_data,
    epochs = 2,
    batch_size = 20,
    validation_data = val_data,
    seed = 123
  )

  output <- capture.output(print(trained_gan))

  expect_true(any(grepl("Validation Metrics", output)))
  expect_true(any(grepl("Discriminator accuracy", output)))
})

test_that("print.trained_RGAN returns object invisibly", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  result <- print(trained_gan)
  expect_identical(result, trained_gan)
})

test_that("plot_losses rejects non-trained_RGAN objects", {
  expect_error(
    plot_losses(list(a = 1)),
    "must be an object of class"
  )
})

test_that("plot_losses requires tracked losses", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = FALSE,
    seed = 123
  )

  expect_error(
    plot_losses(trained_gan),
    "No loss data available"
  )
})

test_that("plot_losses works with tracked losses", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = TRUE,
    seed = 123
  )

  # Should run without error
  expect_no_error(plot_losses(trained_gan))
})

test_that("plot_losses works with smoothing", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 3,
    batch_size = 20,
    track_loss = TRUE,
    seed = 123
  )

  # Should run without error with various smoothing values
  expect_no_error(plot_losses(trained_gan, smooth = 0))
  expect_no_error(plot_losses(trained_gan, smooth = 0.5))
  expect_no_error(plot_losses(trained_gan, smooth = 0.9))
})

test_that("plot_losses returns NULL invisibly", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = TRUE,
    seed = 123
  )

  result <- plot_losses(trained_gan)
  expect_null(result)
})
