test_that("gan_trainer accepts validation_data parameter", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Split data into train/validation
  train_data <- transformed_data[1:80, ]
  val_data <- transformed_data[81:100, ]

  result <- gan_trainer(
    train_data,
    epochs = 3,
    batch_size = 20,
    validation_data = val_data,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_true(!is.null(result$validation_metrics))
  expect_true(length(result$validation_metrics) > 0)
})

test_that("validation_metrics contain expected fields", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  train_data <- transformed_data[1:80, ]
  val_data <- transformed_data[81:100, ]

  result <- gan_trainer(
    train_data,
    epochs = 3,
    batch_size = 20,
    validation_data = val_data,
    seed = 123
  )

  # Check structure of validation metrics
  first_metric <- result$validation_metrics[[1]]
  expect_true("epoch" %in% names(first_metric))
  expect_true("d_accuracy" %in% names(first_metric))
  expect_true("diversity" %in% names(first_metric))

  # d_accuracy should be between 0 and 1
  expect_true(first_metric$d_accuracy >= 0 && first_metric$d_accuracy <= 1)
})

test_that("early_stopping parameter is accepted", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # This should run without error
  result <- gan_trainer(
    transformed_data,
    epochs = 5,
    batch_size = 20,
    early_stopping = TRUE,
    patience = 2,
    track_loss = TRUE,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$early_stopping, TRUE)
  expect_equal(result$settings$patience, 2)
})

test_that("early_stopping with validation_data works together", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  train_data <- transformed_data[1:80, ]
  val_data <- transformed_data[81:100, ]

  result <- gan_trainer(
    train_data,
    epochs = 10,
    batch_size = 20,
    validation_data = val_data,
    early_stopping = TRUE,
    patience = 3,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  # With early stopping enabled, we should have validation metrics
  expect_true(!is.null(result$validation_metrics))
})
