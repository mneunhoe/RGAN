## Tests for the `regularization` / `adam_betas` / `discriminator_dropout`
## additions to gan_trainer() (Experiment 1: modern loss, R3GAN).

make_toy <- function(n = 100) {
  data <- sample_toydata(n = n)
  transformer <- data_transformer$new()
  transformer$fit(data)
  transformer$transform(data)
}

# resolve_regularization ------------------------------------------------------

test_that("resolve_regularization preserves the previous defaults", {
  # No regularization, no wgan-gp -> no penalty
  expect_equal(
    resolve_regularization(NULL, gp_lambda = 10, value_function = "original"),
    list(type = "none", gamma = 0, lazy_interval = 1L)
  )

  # wgan-gp with gp_lambda -> the legacy penalty
  expect_equal(
    resolve_regularization(NULL, gp_lambda = 10, value_function = "wgan-gp"),
    list(type = "wgan-gp", gamma = 10, lazy_interval = 1L)
  )

  # gp_lambda = 0 disables it
  expect_equal(
    resolve_regularization(NULL, gp_lambda = 0, value_function = "wgan-gp")$type,
    "none"
  )
})

test_that("resolve_regularization fills in gamma and lazy_interval defaults", {
  res <- resolve_regularization(list(type = "r1r2"))
  expect_equal(res, list(type = "r1r2", gamma = 10, lazy_interval = 1L))

  res <- resolve_regularization(list(type = "r1", gamma = 30, lazy_interval = 4))
  expect_equal(res, list(type = "r1", gamma = 30, lazy_interval = 4L))
})

test_that("resolve_regularization validates its input", {
  expect_error(resolve_regularization(list(gamma = 1)), "type")
  expect_error(resolve_regularization(list(type = "nonsense")))
  expect_error(resolve_regularization(list(type = "r1", gamma = -1)), "non-negative")
  expect_error(resolve_regularization(list(type = "r1", lazy_interval = 0)), "lazy_interval")
  expect_error(resolve_regularization(list(type = "r1", lazy_interval = 2.5)), "lazy_interval")
  expect_error(resolve_regularization("r1"), "must be NULL or a list")
})

test_that("resolve_regularization warns when gp_lambda disagrees", {
  expect_warning(
    resolve_regularization(list(type = "r1", gamma = 10),
                           gp_lambda = 10, gp_lambda_supplied = TRUE),
    "takes precedence"
  )
  expect_warning(
    resolve_regularization(list(type = "wgan-gp", gamma = 5),
                           gp_lambda = 10, gp_lambda_supplied = TRUE),
    "takes precedence"
  )
  # Agreeing values do not warn
  expect_silent(
    resolve_regularization(list(type = "wgan-gp", gamma = 10),
                           gp_lambda = 10, gp_lambda_supplied = TRUE)
  )
  # Not supplied -> no warning
  expect_silent(
    resolve_regularization(list(type = "r1"), gp_lambda = 10,
                           gp_lambda_supplied = FALSE)
  )
})


# Backward compatibility ------------------------------------------------------

test_that("gan_trainer with unchanged defaults still works", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(1000)

  result <- gan_trainer(transformed_toydata, epochs = 2, seed = 123)

  expect_s3_class(result, "trained_RGAN")
  expect_true(is.null(result$settings$regularization) ||
                identical(result$settings$regularization$type, "none"))
  expect_null(result$settings$adam_betas)
  expect_equal(result$settings$discriminator_dropout, 0.5)
  # 1000 rows, batch 50 -> 20 steps per epoch, 2 epochs
  expect_equal(result$settings$n_d_steps, 40L)
})


# New value functions in the trainer -----------------------------------------

test_that("gan_trainer runs rpgan with r1r2 regularization", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "rpgan",
    regularization = list(type = "r1r2", gamma = 10),
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$regularization,
               list(type = "r1r2", gamma = 10, lazy_interval = 1L))
  expect_equal(result$settings$n_d_steps, 10L)  # 100 / 20 = 5 steps, 2 epochs
})

test_that("rpgan uses a linear-output discriminator", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "rpgan",
    epochs = 1,
    batch_size = 50,
    seed = 1
  )

  d_modules <- names(result$discriminator$seq$children)
  expect_false("Sigmoid_Output" %in% d_modules)

  # ... while "original" does use a sigmoid
  result_orig <- gan_trainer(transformed_toydata, epochs = 1,
                             batch_size = 50, seed = 1)
  expect_true("Sigmoid_Output" %in% names(result_orig$discriminator$seq$children))
})

test_that("gan_trainer runs the non-saturating loss with r1", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "non-saturating",
    regularization = list(type = "r1"),
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$regularization$type, "r1")
  expect_equal(result$settings$regularization$gamma, 10)
  expect_equal(result$settings$n_d_steps, 10L)
})

test_that("gan_trainer runs the hinge loss with r2", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "hinge",
    regularization = list(type = "r2", gamma = 5),
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$regularization$type, "r2")
  expect_equal(result$settings$regularization$gamma, 5)
})

test_that("gan_trainer rejects an unknown value_function", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)
  expect_error(
    gan_trainer(transformed_toydata, value_function = "not-a-loss",
                epochs = 1, batch_size = 50),
    "Unknown value_function"
  )
})


# wgan-gp: legacy gp_lambda and the new regularization list agree -------------

test_that("gp_lambda and regularization = wgan-gp give the same settings", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  legacy <- gan_trainer(
    transformed_toydata,
    value_function = "wgan-gp",
    gp_lambda = 10,
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  modern <- gan_trainer(
    transformed_toydata,
    value_function = "wgan-gp",
    regularization = list(type = "wgan-gp", gamma = 10),
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_equal(legacy$settings$regularization,
               list(type = "wgan-gp", gamma = 10, lazy_interval = 1L))
  expect_equal(legacy$settings$regularization, modern$settings$regularization)
  expect_equal(legacy$settings$gp_lambda, modern$settings$gp_lambda)
  expect_equal(legacy$settings$n_d_steps, modern$settings$n_d_steps)
})

test_that("gan_trainer warns when gp_lambda and regularization disagree", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  expect_warning(
    gan_trainer(
      transformed_toydata,
      value_function = "wgan-gp",
      gp_lambda = 5,
      regularization = list(type = "wgan-gp", gamma = 10),
      epochs = 1,
      batch_size = 50,
      seed = 1
    ),
    "takes precedence"
  )
})


# Lazy regularization, PacGAN, betas, dropout ---------------------------------

test_that("lazy_interval runs and is recorded", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "rpgan",
    regularization = list(type = "r1r2", gamma = 10, lazy_interval = 4),
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$regularization$lazy_interval, 4L)
  expect_equal(result$settings$n_d_steps, 10L)
})

test_that("pac = 10 runs together with r1r2", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(200)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "rpgan",
    regularization = list(type = "r1r2", gamma = 10),
    pac = 10,
    epochs = 2,
    batch_size = 50,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")
  expect_equal(result$settings$pac, 10)
  expect_equal(result$settings$n_d_steps, 8L)  # 200 / 50 = 4 steps, 2 epochs
})

test_that("adam_betas is passed to both default optimizers", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "rpgan",
    regularization = list(type = "r1r2", gamma = 10),
    adam_betas = c(0, 0.99),
    epochs = 1,
    batch_size = 50,
    seed = 123
  )

  expect_equal(result$settings$adam_betas, c(0, 0.99))
  expect_equal(result$generator_optimizer$param_groups[[1]]$betas, c(0, 0.99))
  expect_equal(result$discriminator_optimizer$param_groups[[1]]$betas, c(0, 0.99))
})

test_that("adam_betas is validated", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)
  expect_error(
    gan_trainer(transformed_toydata, adam_betas = 0.5, epochs = 1, batch_size = 50),
    "adam_betas"
  )
  expect_error(
    gan_trainer(transformed_toydata, adam_betas = c(0, 1), epochs = 1, batch_size = 50),
    "adam_betas"
  )
})

test_that("discriminator_dropout = 0 removes dropout from the default discriminator", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "rpgan",
    discriminator_dropout = 0,
    epochs = 1,
    batch_size = 50,
    seed = 123
  )

  expect_equal(result$settings$discriminator_dropout, 0)
  expect_equal(result$discriminator$seq$Dropout_1$p, 0)

  expect_error(
    gan_trainer(transformed_toydata, discriminator_dropout = 1,
                epochs = 1, batch_size = 50),
    "discriminator_dropout"
  )
})


# gan_update_step: the penalty actually changes the discriminator loss --------

test_that("r1r2 increases the reported discriminator loss", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  set.seed(1); torch::torch_manual_seed(1)
  no_reg <- gan_trainer(transformed_toydata, value_function = "rpgan",
                        epochs = 1, batch_size = 50, track_loss = TRUE, seed = 7)
  set.seed(1); torch::torch_manual_seed(1)
  with_reg <- gan_trainer(transformed_toydata, value_function = "rpgan",
                          regularization = list(type = "r1r2", gamma = 10),
                          epochs = 1, batch_size = 50, track_loss = TRUE, seed = 7)

  expect_true(all(with_reg$losses$d_loss >= no_reg$losses$d_loss))
  expect_true(all(is.finite(with_reg$losses$d_loss)))
  expect_true(all(is.finite(with_reg$losses$g_loss)))
})

test_that("print.trained_RGAN shows the regularization", {
  skip_if_not_installed("torch")

  transformed_toydata <- make_toy(100)

  result <- gan_trainer(
    transformed_toydata,
    value_function = "rpgan",
    regularization = list(type = "r1r2", gamma = 10, lazy_interval = 4),
    epochs = 1,
    batch_size = 50,
    seed = 1
  )

  out <- utils::capture.output(print(result))
  expect_true(any(grepl("Regularization: r1r2", out)))
  expect_true(any(grepl("gamma=10", out)))
  expect_true(any(grepl("lazy_interval=4", out)))
  expect_true(any(grepl("Discriminator steps", out)))
})
