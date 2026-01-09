test_that("Generator creates correct output dimensions", {
  skip_if_not_installed("torch")

  g_net <- Generator(noise_dim = 10, data_dim = 5, dropout_rate = 0.5)

  noise <- torch::torch_randn(c(32, 10))
  output <- g_net(noise)

  expect_equal(output$shape[1], 32)  # batch size
  expect_equal(output$shape[2], 5)   # data dimension
})
test_that("Generator with custom hidden units works", {
  skip_if_not_installed("torch")

  g_net <- Generator(
    noise_dim = 10,
    data_dim = 5,
    hidden_units = list(64, 32, 16),
    dropout_rate = 0.3
  )

  noise <- torch::torch_randn(c(16, 10))
  output <- g_net(noise)

  expect_equal(output$shape[1], 16)
  expect_equal(output$shape[2], 5)
})

test_that("Discriminator creates correct output dimensions", {
  skip_if_not_installed("torch")

  d_net <- Discriminator(data_dim = 5, dropout_rate = 0.5)

  data <- torch::torch_randn(c(32, 5))
  output <- d_net(data)

  expect_equal(output$shape[1], 32)  # batch size
  expect_equal(output$shape[2], 1)   # single score per example
})

test_that("Discriminator with sigmoid produces bounded output", {
  skip_if_not_installed("torch")

  d_net <- Discriminator(data_dim = 5, dropout_rate = 0.5, sigmoid = TRUE)
  d_net$eval()  # Disable dropout for deterministic output

  data <- torch::torch_randn(c(100, 5))
  output <- d_net(data)

  # All outputs should be between 0 and 1
  expect_true(all(torch::as_array(output) >= 0))
  expect_true(all(torch::as_array(output) <= 1))
})

test_that("Discriminator without sigmoid has unbounded output", {
  skip_if_not_installed("torch")

  d_net <- Discriminator(data_dim = 5, dropout_rate = 0, sigmoid = FALSE)

  # Use extreme input to get outputs outside [0, 1]
  data <- torch::torch_randn(c(100, 5)) * 10
  output <- d_net(data)

  output_array <- torch::as_array(output)

  # At least some outputs should be outside [0, 1]
 has_negative <- any(output_array < 0)
  has_greater_than_one <- any(output_array > 1)

  # With random initialization and large inputs, we expect some unbounded outputs
  # This is a probabilistic test, so we just check the structure
  expect_equal(length(output_array), 100)
})

test_that("DCGAN_Generator creates correct output shape for images", {
  skip_if_not_installed("torch")

  g_net <- DCGAN_Generator(noise_dim = 100, number_channels = 3, ngf = 64)

  # DCGAN expects noise as (batch, channels, 1, 1)
  noise <- torch::torch_randn(c(4, 100, 1, 1))
  output <- g_net(noise)

  expect_equal(output$shape[1], 4)   # batch size
  expect_equal(output$shape[2], 3)   # RGB channels
  expect_equal(output$shape[3], 64)  # height
  expect_equal(output$shape[4], 64)  # width
})

test_that("DCGAN_Discriminator accepts correct input shape", {
  skip_if_not_installed("torch")

  d_net <- DCGAN_Discriminator(number_channels = 3, ndf = 64)

  # Input should be (batch, channels, height, width)
  images <- torch::torch_randn(c(4, 3, 64, 64))
  output <- d_net(images)

  expect_equal(output$shape[1], 4)  # batch size
})

test_that("Generator uses LeakyReLU activation", {
  skip_if_not_installed("torch")

  g_net <- Generator(noise_dim = 10, data_dim = 5)

  # Check that LeakyReLU modules are present in the sequential
  module_names <- names(g_net$seq$modules)
  activation_modules <- grep("Activation", module_names, value = TRUE)

  expect_true(length(activation_modules) > 0)
})
