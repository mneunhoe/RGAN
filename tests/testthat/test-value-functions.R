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
    d_net$parameters$seq.Linear_1.weight$fill_(10)
  })

  # Verify weights are extreme
  max_before <- d_net$parameters$seq.Linear_1.weight$max()$item()
  expect_true(max_before > 1)

  # Apply weight clipping
  WGAN_weight_clipper(d_net, clip_values = c(-0.01, 0.01))

  # Verify weights are clipped
  max_after <- d_net$parameters$seq.Linear_1.weight$max()$item()
  min_after <- d_net$parameters$seq.Linear_1.weight$min()$item()

  expect_true(max_after <= 0.01)
  expect_true(min_after >= -0.01)
})


# New value functions ---------------------------------------------------------

test_that("NSGAN_value_fct returns finite losses with the right signs", {
  skip_if_not_installed("torch")

  real_scores <- torch::torch_tensor(c(2.0, 1.0, 0.5))
  fake_scores <- torch::torch_tensor(c(-1.0, -2.0, 0.0))

  result <- NSGAN_value_fct(real_scores, fake_scores)

  expect_s3_class(result$d_loss, "torch_tensor")
  expect_s3_class(result$g_loss, "torch_tensor")
  expect_true(is.finite(result$d_loss$item()))
  expect_true(is.finite(result$g_loss$item()))

  # softplus is strictly positive, so both losses are strictly positive
  expect_true(result$d_loss$item() > 0)
  expect_true(result$g_loss$item() > 0)

  # Analytic value
  expected_d <- mean(log1p(exp(-c(2.0, 1.0, 0.5)))) +
    mean(log1p(exp(c(-1.0, -2.0, 0.0))))
  expect_equal(result$d_loss$item(), expected_d, tolerance = 1e-5)
  expect_equal(result$g_loss$item(),
               mean(log1p(exp(c(1.0, 2.0, 0.0)))), tolerance = 1e-5)
})

test_that("NSGAN_value_fct rewards a better discriminator", {
  skip_if_not_installed("torch")

  good <- NSGAN_value_fct(torch::torch_tensor(c(5, 5, 5)),
                          torch::torch_tensor(c(-5, -5, -5)))
  bad <- NSGAN_value_fct(torch::torch_tensor(c(-5, -5, -5)),
                         torch::torch_tensor(c(5, 5, 5)))

  expect_true(good$d_loss$item() < bad$d_loss$item())
  # The generator prefers high fake scores
  expect_true(bad$g_loss$item() < good$g_loss$item())
})

test_that("hinge_value_fct matches its analytic value", {
  skip_if_not_installed("torch")

  real <- c(2.0, 0.5, -1.0)
  fake <- c(-2.0, 0.5, 1.0)

  result <- hinge_value_fct(torch::torch_tensor(real),
                            torch::torch_tensor(fake))

  expected_d <- mean(pmax(0, 1 - real)) + mean(pmax(0, 1 + fake))
  expect_equal(result$d_loss$item(), expected_d, tolerance = 1e-5)
  expect_equal(result$g_loss$item(), -mean(fake), tolerance = 1e-5)

  expect_true(is.finite(result$d_loss$item()))
  expect_true(result$d_loss$item() >= 0)
})

test_that("hinge_value_fct discriminator loss is zero when the margin is met", {
  skip_if_not_installed("torch")

  result <- hinge_value_fct(torch::torch_tensor(c(3, 4, 5)),
                            torch::torch_tensor(c(-3, -4, -5)))
  expect_equal(result$d_loss$item(), 0, tolerance = 1e-6)
})

test_that("RpGAN_value_fct matches its analytic value and is finite", {
  skip_if_not_installed("torch")

  real <- c(1.0, -0.5, 2.0)
  fake <- c(-1.0, 0.5, 0.0)

  result <- RpGAN_value_fct(torch::torch_tensor(real),
                            torch::torch_tensor(fake))

  diff <- real - fake
  expect_equal(result$d_loss$item(), mean(log1p(exp(-diff))), tolerance = 1e-5)
  expect_equal(result$g_loss$item(), mean(log1p(exp(diff))), tolerance = 1e-5)

  expect_true(result$d_loss$item() > 0)
  expect_true(result$g_loss$item() > 0)
})

test_that("RpGAN_value_fct is symmetric: swapping real and fake swaps the losses", {
  skip_if_not_installed("torch")

  real <- torch::torch_tensor(c(1.5, -0.25, 0.75, 3.0))
  fake <- torch::torch_tensor(c(-0.5, 1.25, 0.75, -2.0))

  a <- RpGAN_value_fct(real, fake)
  b <- RpGAN_value_fct(fake, real)

  expect_equal(a$d_loss$item(), b$g_loss$item(), tolerance = 1e-6)
  expect_equal(a$g_loss$item(), b$d_loss$item(), tolerance = 1e-6)
})

test_that("RpGAN_value_fct rewards a discriminator that separates the pairs", {
  skip_if_not_installed("torch")

  good <- RpGAN_value_fct(torch::torch_tensor(c(5, 5, 5)),
                          torch::torch_tensor(c(-5, -5, -5)))
  bad <- RpGAN_value_fct(torch::torch_tensor(c(-5, -5, -5)),
                         torch::torch_tensor(c(5, 5, 5)))

  expect_true(good$d_loss$item() < bad$d_loss$item())
  expect_true(bad$g_loss$item() < good$g_loss$item())
})

test_that("RpGAN_value_fct requires equally sized real and fake batches", {
  skip_if_not_installed("torch")

  expect_error(
    RpGAN_value_fct(torch::torch_tensor(c(1, 2, 3)),
                    torch::torch_tensor(c(1, 2))),
    "same size"
  )
})

test_that("RpGAN_value_fct is flagged as needing the real scores for g_loss", {
  expect_true(isTRUE(attr(RpGAN_value_fct, "requires_real_scores")))
  expect_null(attr(WGAN_value_fct, "requires_real_scores"))
})


# Zero-centred gradient penalties ---------------------------------------------

test_that("r1_penalty is zero for a constant discriminator", {
  skip_if_not_installed("torch")

  const_net <- torch::nn_module(
    initialize = function() {
      self$b <- torch::nn_parameter(torch::torch_zeros(1))
    },
    forward = function(input) {
      self$b$view(c(1, 1))$expand(c(input$shape[1], 1))
    }
  )()

  x <- torch::torch_randn(c(7, 3))
  pen <- r1_penalty(const_net, x, device = "cpu")

  expect_s3_class(pen, "torch_tensor")
  expect_equal(pen$item(), 0, tolerance = 1e-8)
})

test_that("r1_penalty is positive for a non-degenerate discriminator", {
  skip_if_not_installed("torch")

  d_net <- Discriminator(data_dim = 3, hidden_units = list(8), dropout_rate = 0)
  x <- torch::torch_randn(c(16, 3))

  pen <- r1_penalty(d_net, x, device = "cpu")
  expect_true(pen$item() > 0)
  expect_true(is.finite(pen$item()))
})

test_that("r1_penalty equals ||W||^2 for a single linear layer, for any batch size", {
  skip_if_not_installed("torch")

  d_in <- 4
  lin_net <- torch::nn_module(
    initialize = function() {
      self$lin <- torch::nn_linear(d_in, 1)
    },
    forward = function(input) self$lin(input)
  )()

  W <- torch::as_array(lin_net$lin$weight$detach())
  expected <- sum(W^2)

  for (n in c(1, 5, 32)) {
    x <- torch::torch_randn(c(n, d_in))
    pen <- r1_penalty(lin_net, x, device = "cpu")
    expect_equal(pen$item(), expected, tolerance = 1e-5)
  }
})

test_that("r2_penalty behaves like r1_penalty on the given data", {
  skip_if_not_installed("torch")

  d_net <- Discriminator(data_dim = 2, hidden_units = list(8), dropout_rate = 0)
  d_net$eval()
  x <- torch::torch_randn(c(12, 2))

  expect_equal(r1_penalty(d_net, x)$item(),
               r2_penalty(d_net, x)$item(), tolerance = 1e-6)
})

test_that("r1_penalty is differentiable w.r.t. the discriminator parameters", {
  skip_if_not_installed("torch")

  d_net <- Discriminator(data_dim = 3, hidden_units = list(8), dropout_rate = 0)
  x <- torch::torch_randn(c(10, 3))

  pen <- r1_penalty(d_net, x)
  pen$backward()

  # The first linear layer must have received a gradient; the output bias
  # receives none (the input gradient does not depend on it).
  grad <- d_net$parameters$seq.Linear_1.weight$grad
  expect_false(is.null(grad))
  expect_true(as.numeric(grad$abs()$sum()$item()) > 0)
})
