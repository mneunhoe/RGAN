test_that("save_gan creates a file", {
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

  temp_file <- tempfile(fileext = ".rgan")
  on.exit(unlink(temp_file), add = TRUE)

  save_gan(trained_gan, temp_file)

  expect_true(file.exists(temp_file))
})

test_that("load_gan restores a saved model", {
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

  temp_file <- tempfile(fileext = ".rgan")
  on.exit(unlink(temp_file), add = TRUE)

  save_gan(trained_gan, temp_file)
  loaded_gan <- load_gan(temp_file)

  expect_s3_class(loaded_gan, "trained_RGAN")
  expect_true(!is.null(loaded_gan$generator))
  expect_true(!is.null(loaded_gan$discriminator))
  expect_equal(loaded_gan$settings$noise_dim, trained_gan$settings$noise_dim)
  expect_equal(loaded_gan$settings$value_function, trained_gan$settings$value_function)
})

test_that("save/load roundtrip produces identical samples", {
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

  temp_file <- tempfile(fileext = ".rgan")
  on.exit(unlink(temp_file), add = TRUE)

  save_gan(trained_gan, temp_file)
  loaded_gan <- load_gan(temp_file)

  # Generate samples with same noise from both models
  torch::torch_manual_seed(999)
  noise <- torch::torch_randn(c(10, trained_gan$settings$noise_dim))

  trained_gan$generator$eval()
  loaded_gan$generator$eval()

  samples_original <- torch::as_array(trained_gan$generator(noise)$detach())
  samples_loaded <- torch::as_array(loaded_gan$generator(noise)$detach())

  expect_equal(samples_original, samples_loaded, tolerance = 1e-5)
})

test_that("load_gan fails gracefully with invalid file", {
  expect_error(load_gan("nonexistent_file.rgan"), "File not found")
})

test_that("save_gan fails with non-trained_RGAN object", {
  expect_error(save_gan(list(a = 1), "test.rgan"), "must be an object of class")
})

test_that("save/load works with wgan-gp model", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    value_function = "wgan-gp",
    gp_lambda = 5,
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  temp_file <- tempfile(fileext = ".rgan")
  on.exit(unlink(temp_file), add = TRUE)

  save_gan(trained_gan, temp_file)
  loaded_gan <- load_gan(temp_file)

  expect_equal(loaded_gan$settings$value_function, "wgan-gp")
  expect_equal(loaded_gan$settings$gp_lambda, 5)
})
