test_that("gan_trainer runs with default settings", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  # Train for just 2 epochs to keep test fast
 result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_true(!is.null(result$generator))
  expect_true(!is.null(result$discriminator))
  expect_true(!is.null(result$settings))
})

test_that("gan_trainer works with wasserstein value function", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    value_function = "wasserstein",
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$value_function, "wasserstein")
})

test_that("gan_trainer works with wgan-gp value function", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    value_function = "wgan-gp",
    gp_lambda = 10,
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$value_function, "wgan-gp")
  expect_equal(result$settings$gp_lambda, 10)
})

test_that("gan_trainer tracks losses when requested", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = TRUE,
    seed = 123
  )

  expect_true(!is.null(result$losses))
  expect_true(length(result$losses$g_loss) > 0)
  expect_true(length(result$losses$d_loss) > 0)
})

test_that("gan_trainer respects seed for reproducibility", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result1 <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = TRUE,
    seed = 42
  )

  result2 <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    track_loss = TRUE,
    seed = 42
  )

  # Losses should be identical with same seed
  expect_equal(result1$losses$g_loss, result2$losses$g_loss)
  expect_equal(result1$losses$d_loss, result2$losses$d_loss)
})

test_that("sample_synthetic_data produces correct output", {
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

  # Sample without transformer (transformed space)
  synth_transformed <- sample_synthetic_data(trained_gan, n_samples = 50)
  expect_equal(nrow(synth_transformed), 50)
  expect_equal(ncol(synth_transformed), 2)

  # Sample with transformer (original space)
  synth_original <- sample_synthetic_data(trained_gan, transformer, n_samples = 50)
  expect_equal(nrow(synth_original), 50)
  expect_equal(ncol(synth_original), 2)
})

test_that("sample_toydata produces expected output", {
  data <- sample_toydata(n = 500, seed = 123)

  expect_equal(nrow(data), 500)
  expect_equal(ncol(data), 2)
  expect_equal(colnames(data), c("x", "y"))
})
