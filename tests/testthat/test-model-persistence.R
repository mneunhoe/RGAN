test_that("save_gan creates expected files", {
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

  temp_dir <- tempdir()
  base_path <- file.path(temp_dir, "test_gan")
  on.exit({
    unlink(paste0(base_path, "_generator.pt"))
    unlink(paste0(base_path, "_discriminator.pt"))
    unlink(paste0(base_path, "_metadata.rds"))
    unlink(paste0(base_path, "_g_optim.pt"))
    unlink(paste0(base_path, "_d_optim.pt"))
  }, add = TRUE)

  save_gan(trained_gan, base_path)

  expect_true(file.exists(paste0(base_path, "_generator.pt")))
  expect_true(file.exists(paste0(base_path, "_discriminator.pt")))
  expect_true(file.exists(paste0(base_path, "_metadata.rds")))
  expect_true(file.exists(paste0(base_path, "_g_optim.pt")))
  expect_true(file.exists(paste0(base_path, "_d_optim.pt")))
})

test_that("save_gan without optimizers skips optimizer files", {
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

  temp_dir <- tempdir()
  base_path <- file.path(temp_dir, "test_gan_no_optim")
  on.exit({
    unlink(paste0(base_path, "_generator.pt"))
    unlink(paste0(base_path, "_discriminator.pt"))
    unlink(paste0(base_path, "_metadata.rds"))
  }, add = TRUE)

  save_gan(trained_gan, base_path, include_optimizers = FALSE)

  expect_true(file.exists(paste0(base_path, "_generator.pt")))
  expect_true(file.exists(paste0(base_path, "_discriminator.pt")))
  expect_true(file.exists(paste0(base_path, "_metadata.rds")))
  expect_false(file.exists(paste0(base_path, "_g_optim.pt")))
  expect_false(file.exists(paste0(base_path, "_d_optim.pt")))
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

  temp_dir <- tempdir()
  base_path <- file.path(temp_dir, "test_gan_load")
  on.exit({
    unlink(paste0(base_path, "_generator.pt"))
    unlink(paste0(base_path, "_discriminator.pt"))
    unlink(paste0(base_path, "_metadata.rds"))
    unlink(paste0(base_path, "_g_optim.pt"))
    unlink(paste0(base_path, "_d_optim.pt"))
  }, add = TRUE)

  save_gan(trained_gan, base_path)
  loaded_gan <- load_gan(base_path)

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

  temp_dir <- tempdir()
  base_path <- file.path(temp_dir, "test_gan_roundtrip")
  on.exit({
    unlink(paste0(base_path, "_generator.pt"))
    unlink(paste0(base_path, "_discriminator.pt"))
    unlink(paste0(base_path, "_metadata.rds"))
    unlink(paste0(base_path, "_g_optim.pt"))
    unlink(paste0(base_path, "_d_optim.pt"))
  }, add = TRUE)

  save_gan(trained_gan, base_path)
  loaded_gan <- load_gan(base_path)

  # Generate samples with same noise from both models
  torch::torch_manual_seed(999)
  noise <- torch::torch_randn(c(10, trained_gan$settings$noise_dim))

  trained_gan$generator$eval()
  loaded_gan$generator$eval()

  samples_original <- torch::as_array(trained_gan$generator(noise)$detach())
  samples_loaded <- torch::as_array(loaded_gan$generator(noise)$detach())

  expect_equal(samples_original, samples_loaded, tolerance = 1e-5)
})

test_that("load_gan fails gracefully with missing files", {
  expect_error(load_gan("nonexistent_model"), "Generator file not found")
})

test_that("save_gan fails with non-trained_RGAN object", {
  expect_error(save_gan(list(a = 1), "test"), "must be an object of class")
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

  temp_dir <- tempdir()
  base_path <- file.path(temp_dir, "test_gan_wgangp")
  on.exit({
    unlink(paste0(base_path, "_generator.pt"))
    unlink(paste0(base_path, "_discriminator.pt"))
    unlink(paste0(base_path, "_metadata.rds"))
    unlink(paste0(base_path, "_g_optim.pt"))
    unlink(paste0(base_path, "_d_optim.pt"))
  }, add = TRUE)

  save_gan(trained_gan, base_path)
  loaded_gan <- load_gan(base_path)

  expect_equal(loaded_gan$settings$value_function, "wgan-gp")
  expect_equal(loaded_gan$settings$gp_lambda, 5)
})

test_that("load_gan handles path with extension", {
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

  temp_dir <- tempdir()
  base_path <- file.path(temp_dir, "test_gan_ext")
  on.exit({
    unlink(paste0(base_path, "_generator.pt"))
    unlink(paste0(base_path, "_discriminator.pt"))
    unlink(paste0(base_path, "_metadata.rds"))
    unlink(paste0(base_path, "_g_optim.pt"))
    unlink(paste0(base_path, "_d_optim.pt"))
  }, add = TRUE)

  # Save with extension (should be stripped)
  save_gan(trained_gan, paste0(base_path, ".rgan"))

  # Load with extension (should be stripped)
  loaded_gan <- load_gan(paste0(base_path, ".rgan"))

  expect_s3_class(loaded_gan, "trained_RGAN")
})
