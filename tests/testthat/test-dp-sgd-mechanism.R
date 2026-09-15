# The DP-SGD step must release the quantity the accountant accounts for.
#
# The Sampled Gaussian Mechanism, as formalized (PrivatePGB.sgmOutput), releases
# the Poisson-subsampled SUM of clipped contributions plus Gaussian noise. The
# size of the realised Poisson draw is not part of that release. Dividing the
# result by a constant is post-processing and costs nothing; dividing it by the
# realised batch size would make the released scale depend on a data-dependent
# count, and no part of the accounting pays for that.

skip_if_no_torch <- function() {
  testthat::skip_if_not_installed("torch")
  testthat::skip_if_not(torch::torch_is_installed(), "torch runtime missing")
}

make_fixture <- function(input_dim = 20, hidden = 400, noise_dim = 8) {
  d_net <- torch::nn_sequential(
    torch::nn_linear(input_dim, hidden),
    torch::nn_relu(),
    torch::nn_linear(hidden, 1),
    torch::nn_sigmoid()
  )
  g_net <- torch::nn_sequential(
    torch::nn_linear(noise_dim, input_dim),
    torch::nn_tanh()
  )
  list(
    d_net = d_net,
    g_net = g_net,
    d_optim = torch::optim_sgd(d_net$parameters, lr = 0),
    sample_noise = function(dim) torch::torch_randn(dim),
    noise_dim = noise_dim,
    input_dim = input_dim
  )
}

grad_vector <- function(net) {
  parts <- lapply(net$parameters, function(p) as.numeric(p$grad$view(-1)))
  unlist(parts)
}


test_that("the released noise scale is set by the expected batch size", {
  skip_if_no_torch()

  n_samples <- 200
  sampling_rate <- 0.05          # expected batch 10; realised varies by ~30%
  max_grad_norm <- 1.0
  noise_multiplier <- 1e6        # drown the signal, so grads are pure noise
  expected_batch <- sampling_rate * n_samples

  fx <- make_fixture()
  data <- torch::torch_randn(c(n_samples, fx$input_dim))

  # lr = 0, so the parameters do not move and each step sees the same net.
  observed <- vapply(1:6, function(k) {
    dp_discriminator_step(
      data = data, n_samples = n_samples, sampling_rate = sampling_rate,
      noise_dim = fx$noise_dim, sample_noise = fx$sample_noise,
      device = "cpu", g_net = fx$g_net, d_net = fx$d_net,
      d_optim = fx$d_optim, max_grad_norm = max_grad_norm,
      noise_multiplier = noise_multiplier, secure_rng = FALSE
    )
    stats::sd(grad_vector(fx$d_net))
  }, numeric(1))

  target <- noise_multiplier * max_grad_norm / expected_batch

  # Every step lands on the same scale. Were the divisor the realised batch
  # size, these would scatter by roughly 1 / sqrt(10), about 30 percent.
  for (s in observed) expect_lt(abs(s - target) / target, 0.05)
  expect_lt(stats::sd(observed) / mean(observed), 0.05)
})


test_that("an empty Poisson draw still releases noise and takes a step", {
  skip_if_no_torch()

  fx <- make_fixture()
  expected_batch <- 10
  max_grad_norm <- 1.0
  noise_multiplier <- 3.0

  before <- as.numeric(fx$d_net$parameters[[1]]$view(-1))
  loss <- dp_noise_only_step(
    d_net = fx$d_net, d_optim = torch::optim_sgd(fx$d_net$parameters, lr = 0.1),
    max_grad_norm = max_grad_norm, noise_multiplier = noise_multiplier,
    expected_batch_size = expected_batch, secure_rng = FALSE
  )
  after <- as.numeric(fx$d_net$parameters[[1]]$view(-1))

  expect_equal(loss, 0)
  # The optimizer moved: skipping the step instead would make "no step" an
  # observable event of probability (1 - q)^n, which depends on n.
  expect_false(isTRUE(all.equal(before, after)))

  g <- grad_vector(fx$d_net)
  target <- noise_multiplier * max_grad_norm / expected_batch
  expect_lt(abs(stats::sd(g) - target) / target, 0.06)
  expect_lt(abs(mean(g)), 4 * target / sqrt(length(g)))
})


test_that("a step with an empty draw is routed through the noise-only path", {
  skip_if_no_torch()

  # A sampling rate this small makes the Poisson draw empty with probability
  # essentially one, while keeping the expected batch size positive.
  fx <- make_fixture()
  data <- torch::torch_randn(c(50, fx$input_dim))

  loss <- dp_discriminator_step(
    data = data, n_samples = 50, sampling_rate = 1e-9,
    noise_dim = fx$noise_dim, sample_noise = fx$sample_noise,
    device = "cpu", g_net = fx$g_net, d_net = fx$d_net,
    d_optim = fx$d_optim, max_grad_norm = 1.0,
    noise_multiplier = 1.0, secure_rng = FALSE
  )

  expect_equal(loss, 0)
  # Noise was still written, rather than the step being skipped outright.
  g <- grad_vector(fx$d_net)
  expect_true(all(is.finite(g)))
  expect_gt(stats::sd(g), 0)
})


test_that("a zero expected batch size is rejected rather than divided by", {
  skip_if_no_torch()

  fx <- make_fixture()
  data <- torch::torch_randn(c(50, fx$input_dim))

  expect_error(
    dp_discriminator_step(
      data = data, n_samples = 50, sampling_rate = 0,
      noise_dim = fx$noise_dim, sample_noise = fx$sample_noise,
      device = "cpu", g_net = fx$g_net, d_net = fx$d_net,
      d_optim = fx$d_optim, max_grad_norm = 1.0,
      noise_multiplier = 1.0, secure_rng = FALSE
    ),
    "must be positive"
  )
})
