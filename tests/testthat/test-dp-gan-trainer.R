# Tests for dp_gan_trainer and related functions

# Skip all tests if OpenDP is not available
skip_if_no_opendp <- function() {
  if (!requireNamespace("opendp", quietly = TRUE)) {
    skip("OpenDP package not available")
  }
}

test_that("dp_accountant_poisson initializes correctly", {
  accountant <- dp_accountant_poisson$new(
    sampling_rate = 0.01,
    noise_multiplier = 1.0,
    target_delta = 1e-5
  )

  expect_equal(accountant$steps, 0)
  expect_equal(accountant$sampling_rate, 0.01)
  expect_equal(accountant$noise_multiplier, 1.0)
  expect_equal(accountant$target_delta, 1e-5)
})

test_that("dp_accountant_poisson tracks steps correctly", {
  accountant <- dp_accountant_poisson$new(
    sampling_rate = 0.01,
    noise_multiplier = 1.0,
    target_delta = 1e-5
  )

  accountant$step()
  expect_equal(accountant$steps, 1)

  accountant$step(5)
  expect_equal(accountant$steps, 6)
})

test_that("dp_accountant_poisson computes RDP correctly", {
  accountant <- dp_accountant_poisson$new(
    sampling_rate = 0.01,
    noise_multiplier = 1.0,
    target_delta = 1e-5
  )

  # RDP should be positive for alpha > 1
  rdp_2 <- accountant$compute_rdp(2)
  expect_true(rdp_2 >= 0)

  rdp_10 <- accountant$compute_rdp(10)
  expect_true(rdp_10 >= 0)

  # RDP should increase with alpha
  expect_true(rdp_10 >= rdp_2)
})

test_that("dp_accountant_poisson computes epsilon correctly", {
  accountant <- dp_accountant_poisson$new(
    sampling_rate = 0.01,
    noise_multiplier = 1.0,
    target_delta = 1e-5
  )

  # Zero steps should give near-zero epsilon
  eps_0 <- accountant$get_epsilon()
  expect_true(eps_0 >= 0)

  # After steps, epsilon should increase
  accountant$step(100)
  eps_100 <- accountant$get_epsilon()
  expect_true(eps_100 > eps_0)

  accountant$step(100)
  eps_200 <- accountant$get_epsilon()
  expect_true(eps_200 > eps_100)
})

test_that("dp_accountant_poisson: higher noise_multiplier gives lower epsilon", {
  accountant_low <- dp_accountant_poisson$new(
    sampling_rate = 0.01,
    noise_multiplier = 0.5,
    target_delta = 1e-5
  )
  accountant_high <- dp_accountant_poisson$new(
    sampling_rate = 0.01,
    noise_multiplier = 2.0,
    target_delta = 1e-5
  )

  accountant_low$step(100)
  accountant_high$step(100)

  eps_low <- accountant_low$get_epsilon()
  eps_high <- accountant_high$get_epsilon()

  expect_true(eps_high < eps_low)
})

test_that("dp_accountant_poisson: lower sampling_rate gives lower epsilon", {
  accountant_low <- dp_accountant_poisson$new(
    sampling_rate = 0.001,
    noise_multiplier = 1.0,
    target_delta = 1e-5
  )
  accountant_high <- dp_accountant_poisson$new(
    sampling_rate = 0.1,
    noise_multiplier = 1.0,
    target_delta = 1e-5
  )

  accountant_low$step(100)
  accountant_high$step(100)

  eps_low <- accountant_low$get_epsilon()
  eps_high <- accountant_high$get_epsilon()

  expect_true(eps_low < eps_high)
})

test_that("calibrate_noise_multiplier finds reasonable values", {
  noise <- calibrate_noise_multiplier(
    target_epsilon = 1.0,
    target_delta = 1e-5,
    sampling_rate = 0.01,
    total_steps = 1000
  )

  expect_true(noise > 0)
  expect_true(noise < 100)

  # Verify it achieves approximately the target epsilon
  accountant <- dp_accountant_poisson$new(
    sampling_rate = 0.01,
    noise_multiplier = noise,
    target_delta = 1e-5
  )
  accountant$step(1000)

  achieved_eps <- accountant$get_epsilon()
  expect_true(abs(achieved_eps - 1.0) < 0.1)
})

test_that("calibrate_noise_multiplier: tighter epsilon needs more noise", {
  noise_loose <- calibrate_noise_multiplier(
    target_epsilon = 10.0,
    target_delta = 1e-5,
    sampling_rate = 0.01,
    total_steps = 1000
  )

  noise_tight <- calibrate_noise_multiplier(
    target_epsilon = 0.5,
    target_delta = 1e-5,
    sampling_rate = 0.01,
    total_steps = 1000
  )

  expect_true(noise_tight > noise_loose)
})

test_that("secure_poisson_subsample returns valid indices", {
  skip_if_no_opendp()

  indices <- secure_poisson_subsample(100, 0.1)

  # Should return integer indices

  expect_type(indices, "integer")

  # All indices should be valid
  if (length(indices) > 0) {
    expect_true(all(indices >= 1))
    expect_true(all(indices <= 100))
  }

  # No duplicates
  expect_equal(length(indices), length(unique(indices)))
})

test_that("secure_poisson_subsample respects sampling rate on average", {
  skip_if_no_opendp()

  # Run multiple times and check average
  n_trials <- 100
  n_samples <- 1000
  sampling_rate <- 0.1

  counts <- sapply(1:n_trials, function(i) {
    length(secure_poisson_subsample(n_samples, sampling_rate))
  })

  # Average should be close to expected (with some tolerance)
  expected <- n_samples * sampling_rate
  mean_count <- mean(counts)

  # Within 20% of expected (allowing for randomness)
  expect_true(mean_count > expected * 0.8)
  expect_true(mean_count < expected * 1.2)
})

test_that("sample_secure_gaussian_like produces correct shape", {
  skip_if_no_opendp()

  # Test with different shapes
  tensor_2d <- torch::torch_zeros(10, 5)
  noise_2d <- sample_secure_gaussian_like(tensor_2d, 1.0)
  expect_equal(as.integer(noise_2d$shape), c(10, 5))

  tensor_1d <- torch::torch_zeros(100)
  noise_1d <- sample_secure_gaussian_like(tensor_1d, 1.0)
  expect_equal(as.integer(noise_1d$shape), c(100))
})

test_that("sample_secure_gaussian_like produces non-zero noise", {
  skip_if_no_opendp()

  tensor <- torch::torch_zeros(100, 10)
  noise <- sample_secure_gaussian_like(tensor, 1.0)

  # Noise should have non-trivial variance
  noise_var <- torch::as_array(noise$var())
  expect_true(noise_var > 0.1)
})

test_that("dp_gan_trainer validates inputs correctly", {
  skip_if_no_opendp()

  data <- matrix(rnorm(200), ncol = 2)

  # Invalid batch_size
  expect_error(
    dp_gan_trainer(data, batch_size = -1, epochs = 1),
    "batch_size must be a positive integer"
  )

  # Invalid target_epsilon
  expect_error(
    dp_gan_trainer(data, target_epsilon = -1, epochs = 1),
    "target_epsilon must be a positive number"
  )

  # Invalid target_delta
  expect_error(
    dp_gan_trainer(data, target_delta = 2, epochs = 1),
    "target_delta must be between 0 and 1"
  )

  # Invalid max_grad_norm
  expect_error(
    dp_gan_trainer(data, max_grad_norm = 0, epochs = 1),
    "max_grad_norm must be a positive number"
  )
})

test_that("dp_gan_trainer runs without error on small data", {
  skip_if_no_opendp()

  data <- matrix(rnorm(200), ncol = 2)

  # Very short training run
  result <- dp_gan_trainer(
    data,
    epochs = 2,
    target_epsilon = 10.0,  # Loose privacy for faster test
    verbose = FALSE
  )

  expect_s3_class(result, "trained_RGAN")
  expect_true(!is.null(result$generator))
  expect_true(!is.null(result$discriminator))
  expect_true(!is.null(result$privacy))
  expect_true(result$privacy$final_epsilon > 0)
})

test_that("dp_gan_trainer tracks privacy correctly", {
  skip_if_no_opendp()

  data <- matrix(rnorm(200), ncol = 2)

  result <- dp_gan_trainer(
    data,
    epochs = 5,
    target_epsilon = 10.0,
    verbose = FALSE
  )

  # Should have privacy information
  expect_true(!is.null(result$privacy$final_epsilon))
  expect_true(!is.null(result$privacy$delta))
  expect_true(!is.null(result$privacy$noise_multiplier))
  expect_true(!is.null(result$privacy$max_grad_norm))
  expect_true(!is.null(result$privacy$total_steps))

  # Epsilon should be reasonable
  expect_true(result$privacy$final_epsilon > 0)
  expect_true(result$privacy$final_epsilon <= result$settings$target_epsilon * 1.1)
})

test_that("dp_gan_trainer produces usable generator", {
  skip_if_no_opendp()

  data <- matrix(rnorm(200), ncol = 2)

  result <- dp_gan_trainer(
    data,
    epochs = 3,
    target_epsilon = 10.0,
    verbose = FALSE
  )

  # Generator should produce output of correct dimension
  noise <- torch::torch_randn(c(10, 2))
  synth <- torch::with_no_grad(result$generator(noise))

  expect_equal(as.integer(synth$shape), c(10, 2))
})

test_that("dp_gan_trainer tracks losses when requested", {
  skip_if_no_opendp()

  data <- matrix(rnorm(200), ncol = 2)

  result <- dp_gan_trainer(
    data,
    epochs = 2,
    target_epsilon = 10.0,
    track_loss = TRUE,
    verbose = FALSE
  )

  expect_true(!is.null(result$losses))
  expect_true(length(result$losses$g_loss) > 0)
  expect_true(length(result$losses$d_loss) > 0)
})

test_that("dp_gan_trainer stops when privacy budget exhausted", {
  skip_if_no_opendp()

  data <- matrix(rnorm(200), ncol = 2)

  # Very tight privacy budget
  result <- suppressWarnings(dp_gan_trainer(
    data,
    epochs = 100,  # Many epochs
    target_epsilon = 0.01,  # Very tight
    noise_multiplier = 0.1,  # Will exhaust budget quickly
    verbose = FALSE
  ))

  # Should have stopped early
  expect_true(result$privacy$total_steps < 100 * ceiling(nrow(data) / 50))
})

test_that("generator_step produces reasonable loss", {
  g_net <- Generator(noise_dim = 2, data_dim = 2, dropout_rate = 0.5)
  d_net <- Discriminator(data_dim = 2, dropout_rate = 0.5, sigmoid = TRUE)
  g_optim <- torch::optim_sgd(g_net$parameters, lr = 0.001)

  loss <- generator_step(
    batch_size = 10,
    noise_dim = 2,
    sample_noise = torch::torch_randn,
    device = "cpu",
    g_net = g_net,
    d_net = d_net,
    g_optim = g_optim
  )

  expect_true(is.numeric(loss))
  expect_true(is.finite(loss))
})
