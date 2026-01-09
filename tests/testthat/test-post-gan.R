# Tests for post-GAN boosting functionality

test_that("gan_trainer accepts checkpoint parameters", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 6,
    batch_size = 20,
    checkpoint_epochs = 2,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_true(!is.null(result$checkpoints))
  expect_equal(result$checkpoints$epochs, c(2, 4, 6))
  expect_equal(length(result$checkpoints$discriminators), 3)
  expect_equal(length(result$checkpoints$generators), 3)
  expect_false(result$checkpoints$on_disk)
})

test_that("checkpoint_epochs = NULL disables checkpointing", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 4,
    batch_size = 20,
    checkpoint_epochs = NULL,
    seed = 123
  )

  expect_null(result$checkpoints)
})

test_that("disk-based checkpoints are saved correctly", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  temp_dir <- tempdir()
  checkpoint_dir <- file.path(temp_dir, paste0("test_checkpoints_", Sys.getpid()))
  on.exit(unlink(checkpoint_dir, recursive = TRUE), add = TRUE)

  result <- gan_trainer(
    transformed_data,
    epochs = 4,
    batch_size = 20,
    checkpoint_epochs = 2,
    checkpoint_path = checkpoint_dir,
    seed = 123
  )

  expect_true(result$checkpoints$on_disk)
  expect_true(dir.exists(checkpoint_dir))
  expect_true(file.exists(result$checkpoints$discriminators[[1]]))
  expect_true(file.exists(result$checkpoints$generators[[1]]))
})

test_that("compute_discriminator_scores works with in-memory checkpoints", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 4,
    batch_size = 20,
    checkpoint_epochs = 2,
    seed = 123
  )

  # Generate some test samples
  test_samples <- sample_synthetic_data(trained_gan, n_samples = 50)

  scores <- compute_discriminator_scores(
    trained_gan = trained_gan,
    generated_samples = test_samples,
    real_data = transformed_data
  )

  expect_equal(nrow(scores$d_score_fake), 2)  # 2 checkpoints
  expect_equal(ncol(scores$d_score_fake), 50)  # 50 samples
  expect_equal(length(scores$d_score_real), 2)
  expect_equal(scores$epochs, c(2, 4))
})

test_that("compute_discriminator_scores fails without checkpoints", {
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

  test_samples <- sample_synthetic_data(trained_gan, n_samples = 20)

  expect_error(
    compute_discriminator_scores(trained_gan, test_samples, transformed_data),
    "does not contain any checkpoints"
  )
})

test_that("post_gan_boosting runs without errors", {
  skip_if_not_installed("torch")

  # Create synthetic discriminator scores for testing
  n_discriminators <- 5
  n_samples <- 100
  data_dim <- 2

  set.seed(123)
  d_score_fake <- matrix(runif(n_discriminators * n_samples, 0.3, 0.7),
                          nrow = n_discriminators)
  d_score_real <- runif(n_discriminators, 0.6, 0.9)
  B <- matrix(rnorm(n_samples * data_dim), nrow = n_samples)

  # Capture output to suppress progress messages
  result <- suppressMessages(post_gan_boosting(
    d_score_fake = d_score_fake,
    d_score_real = d_score_real,
    B = B,
    real_N = 100,
    steps = 10,
    N_generators = n_discriminators
  ))

  expect_true(!is.null(result$PGB_sample))
  expect_true(!is.null(result$d_score_PGB))
  expect_true(nrow(result$PGB_sample) <= n_samples)
})

test_that("apply_post_gan_boosting runs full workflow", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 6,
    batch_size = 20,
    checkpoint_epochs = 2,
    seed = 123
  )

  # Suppress progress messages
  result <- suppressMessages(apply_post_gan_boosting(
    trained_gan = trained_gan,
    real_data = transformed_data,
    n_candidates = 50,
    steps = 10,
    seed = 456
  ))

  expect_true(!is.null(result$samples))
  expect_true(!is.null(result$scores))
  expect_true(result$n_unique > 0)
  expect_equal(ncol(result$samples), 2)  # Same dimension as original data
})

test_that("apply_post_gan_boosting works with transformer", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  trained_gan <- gan_trainer(
    transformed_data,
    epochs = 6,
    batch_size = 20,
    checkpoint_epochs = 2,
    seed = 123
  )

  # Suppress progress messages
  result <- suppressMessages(apply_post_gan_boosting(
    trained_gan = trained_gan,
    real_data = transformed_data,
    transformer = transformer,
    n_candidates = 50,
    steps = 10,
    seed = 456
  ))

  # With transformer, samples should be inverse transformed
  expect_true(!is.null(result$samples))
  expect_equal(ncol(result$samples), 2)
})

test_that("checkpoint_epochs validation works", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_error(
    gan_trainer(transformed_data, epochs = 4, checkpoint_epochs = -1),
    "checkpoint_epochs must be a positive integer"
  )

  expect_error(
    gan_trainer(transformed_data, epochs = 4, checkpoint_epochs = 2.5),
    "checkpoint_epochs must be a positive integer"
  )
})

test_that("post_gan_boosting with dp=TRUE works", {
  skip_if_not_installed("torch")

  n_discriminators <- 5
  n_samples <- 100
  data_dim <- 2

  set.seed(123)
  d_score_fake <- matrix(runif(n_discriminators * n_samples, 0.3, 0.7),
                          nrow = n_discriminators)
  d_score_real <- runif(n_discriminators, 0.6, 0.9)
  B <- matrix(rnorm(n_samples * data_dim), nrow = n_samples)

  # Suppress progress messages
  result <- suppressMessages(post_gan_boosting(
    d_score_fake = d_score_fake,
    d_score_real = d_score_real,
    B = B,
    real_N = 100,
    steps = 10,
    N_generators = n_discriminators,
    dp = TRUE,
    MW_epsilon = 1.0
  ))

  expect_true(!is.null(result$PGB_sample))
})

test_that("checkpoints work with PacGAN", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  result <- gan_trainer(
    transformed_data,
    epochs = 4,
    batch_size = 20,
    checkpoint_epochs = 2,
    pac = 2,
    seed = 123
  )

  expect_equal(result$settings$pac, 2)
  expect_true(!is.null(result$checkpoints))
  expect_equal(length(result$checkpoints$epochs), 2)
})

test_that("checkpoint_path warning when checkpoint_epochs is NULL", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed_data <- transformer$transform(data)

  expect_warning(
    gan_trainer(
      transformed_data,
      epochs = 2,
      batch_size = 20,
      checkpoint_epochs = NULL,
      checkpoint_path = "/tmp/test"
    ),
    "checkpoint_path provided but checkpoint_epochs is NULL"
  )
})
