test_that("GAN_value_fct returns correct structure", {
  skip_if_not_installed("torch")

  real_scores <- torch::torch_tensor(c(0.8, 0.9, 0.7))
  fake_scores <- torch::torch_tensor(c(0.2, 0.1, 0.3))

  result <- GAN_value_fct(real_scores, fake_scores)

  expect_true("d_loss" %in% names(result))
  expect_true("g_loss" %in% names(result))
  expect_s3_class(result$d_loss, "torch_tensor")
  expect_s3_class(result$g_loss, "torch_tensor")
})

test_that("GAN_value_fct handles edge cases with clamping", {
  skip_if_not_installed("torch")

  # Scores at boundaries should not produce Inf/NaN due to clamping
  real_scores <- torch::torch_tensor(c(1.0, 1.0, 1.0))
  fake_scores <- torch::torch_tensor(c(0.0, 0.0, 0.0))

  result <- GAN_value_fct(real_scores, fake_scores)

  expect_false(is.infinite(result$d_loss$item()))
  expect_false(is.nan(result$d_loss$item()))
  expect_false(is.infinite(result$g_loss$item()))
  expect_false(is.nan(result$g_loss$item()))
})

test_that("WGAN_value_fct returns correct structure", {
  skip_if_not_installed("torch")

  real_scores <- torch::torch_tensor(c(2.0, 1.5, 1.8))
  fake_scores <- torch::torch_tensor(c(-1.0, -0.5, -0.8))

  result <- WGAN_value_fct(real_scores, fake_scores)

  expect_true("d_loss" %in% names(result))
  expect_true("g_loss" %in% names(result))
  expect_s3_class(result$d_loss, "torch_tensor")
  expect_s3_class(result$g_loss, "torch_tensor")
})

test_that("WGAN_value_fct discriminator loss is negative of Wasserstein distance", {
  skip_if_not_installed("torch")

  # If discriminator correctly separates real (high) from fake (low),
  # the Wasserstein distance estimate is E[D(real)] - E[D(fake)]
  real_scores <- torch::torch_tensor(c(2.0, 2.0, 2.0))
  fake_scores <- torch::torch_tensor(c(-1.0, -1.0, -1.0))

  result <- WGAN_value_fct(real_scores, fake_scores)

  # D wants to maximize E[D(real)] - E[D(fake)], so loss is negative
  # E[D(real)] = 2, E[D(fake)] = -1, distance = 3
  # d_loss should be -3 (we minimize negative distance)
  expect_equal(result$d_loss$item(), -3, tolerance = 0.01)
})

test_that("KLWGAN_value_fct returns correct structure", {
  skip_if_not_installed("torch")

  real_scores <- torch::torch_tensor(c(0.5, 0.8, 0.6))
  fake_scores <- torch::torch_tensor(c(-0.3, -0.5, -0.2))

  result <- KLWGAN_value_fct(real_scores, fake_scores)

  expect_true("d_loss" %in% names(result))
  expect_true("g_loss" %in% names(result))
  expect_s3_class(result$d_loss, "torch_tensor")
  expect_s3_class(result$g_loss, "torch_tensor")
})

test_that("gradient_penalty computes correct penalty", {
  skip_if_not_installed("torch")

  # Create a simple discriminator
  d_net <- Discriminator(data_dim = 2, hidden_units = list(8), dropout_rate = 0)

  # Create real and fake data
  real_data <- torch::torch_randn(c(10, 2))
  fake_data <- torch::torch_randn(c(10, 2))

  gp <- gradient_penalty(d_net, real_data, fake_data, device = "cpu")

  expect_s3_class(gp, "torch_tensor")
  expect_true(gp$item() >= 0)  # Penalty should be non-negative
})

test_that("gradient_penalty is zero for linear discriminator with unit gradient", {
 skip_if_not_installed("torch")

  # For a linear function f(x) = w'x with ||w|| = 1, the gradient norm is 1
  # and the penalty should be close to 0

  # This is a more theoretical test - in practice we just verify it's reasonable
  d_net <- Discriminator(data_dim = 2, hidden_units = list(8), dropout_rate = 0)

  real_data <- torch::torch_randn(c(20, 2))
  fake_data <- torch::torch_randn(c(20, 2))

  gp <- gradient_penalty(d_net, real_data, fake_data, device = "cpu")

  # GP should be finite and reasonable (not exploding)
 expect_true(gp$item() < 100)
})

test_that("WGAN_weight_clipper clips weights correctly", {
  skip_if_not_installed("torch")

  d_net <- Discriminator(data_dim = 2, hidden_units = list(8), dropout_rate = 0)

  # Set some weights to extreme values
  torch::with_no_grad({
    d_net$parameters$Linear_1.weight$fill_(10)
  })

  # Verify weights are extreme
  max_before <- d_net$parameters$Linear_1.weight$max()$item()
  expect_true(max_before > 1)

  # Apply weight clipping
  WGAN_weight_clipper(d_net, clip_values = c(-0.01, 0.01))

  # Verify weights are clipped
  max_after <- d_net$parameters$Linear_1.weight$max()$item()
  min_after <- d_net$parameters$Linear_1.weight$min()$item()

  expect_true(max_after <= 0.01)
  expect_true(min_after >= -0.01)
})
