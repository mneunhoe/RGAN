#' @title GAN Value Function
#'
#' @description Implements the original GAN value function as a function to be called in gan_trainer.
#'   The function can serve as a template to implement new value functions in RGAN.
#'
#' @param real_scores The discriminator scores on real examples ($D(x)$)
#' @param fake_scores The discriminator scores on fake examples ($D(G(z))$)
#'
#' @return The function returns a named list with the entries d_loss and g_loss
#' @export
GAN_value_fct <- function(real_scores, fake_scores, epsilon = 1e-7) {
  # Clamp scores to avoid log(0) and log(1) numerical instability
  real_scores <- torch::torch_clamp(real_scores, epsilon, 1 - epsilon)
  fake_scores <- torch::torch_clamp(fake_scores, epsilon, 1 - epsilon)

  d_loss <-
    torch::torch_log(real_scores) + torch::torch_log(1 - fake_scores)
  d_loss <- -d_loss$mean()


  g_loss <- torch::torch_log(1 - fake_scores)

  g_loss <- g_loss$mean()

  return(list(d_loss = d_loss,
              g_loss = g_loss))

}

#' @title WGAN Value Function
#'
#' @description Implements the Wasserstein GAN (WGAN) value function as a function to be called in gan_trainer.
#'   Note that for this to work properly you also need to implement a weight clipper (or other procedure) to constrain the Discriminator weights.
#'
#' @param real_scores The discriminator scores on real examples ($D(x)$)
#' @param fake_scores The discriminator scores on fake examples ($D(G(z))$)
#'
#' @return The function returns a named list with the entries d_loss and g_loss
#' @export
WGAN_value_fct <- function(real_scores, fake_scores) {
  d_loss <-
    torch::torch_mean(real_scores) - torch::torch_mean(fake_scores)
  d_loss <- -d_loss$mean()


  g_loss <- torch::torch_mean(fake_scores)

  g_loss <- -g_loss$mean()

  return(list(d_loss = d_loss,
              g_loss = g_loss))

}

#' @title KLWGAN Value Function
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param real_scores The discriminator scores on real examples ($D(x)$)
#' @param fake_scores The discriminator scores on fake examples ($D(G(z))$)
#'
#' @return The function returns a named list with the entries d_loss and g_loss
#' @export
KLWGAN_value_fct <- function(real_scores, fake_scores) {
  d_loss <-
    kl_real(real_scores) + kl_fake(fake_scores)
  d_loss <- d_loss$mean()


  g_loss <-  kl_gen(fake_scores)

  g_loss <- g_loss$mean()

  return(list(d_loss = d_loss,
              g_loss = g_loss))

}


#' @title WGAN Weight Clipper
#'
#' @description A function that clips the weights of a Discriminator (for WGAN training).
#'
#' @param d_net A torch::nn_module (typically a discriminator/critic) for which the weights should be clipped
#' @param clip_values A vector with the lower and upper bound for weight values. Any value outside this range will be set to the closer value.
#'
#' @return The function modifies the torch::nn_module weights in place
#' @export
WGAN_weight_clipper <- function(d_net, clip_values = c(-0.01, 0.01)) {
  for (parameter in names(d_net$parameters)) {
    d_net$parameters[[parameter]]$data()$clip_(clip_values[1], clip_values[2])
  }
}


#' @title Gradient Penalty for WGAN-GP
#'
#' @description Computes the gradient penalty for WGAN-GP training as described in
#'   Gulrajani et al. (2017) "Improved Training of Wasserstein GANs".
#'   The gradient penalty enforces the Lipschitz constraint on the discriminator
#'   by penalizing gradients that deviate from norm 1 on interpolated samples.
#'
#' @param d_net The discriminator network (torch::nn_module)
#' @param real_data Real data samples (torch_tensor)
#' @param fake_data Generated fake data samples (torch_tensor)
#' @param device The device to use ("cpu", "cuda", or "mps")
#'
#' @return The gradient penalty loss (torch_tensor)
#' @export
gradient_penalty <- function(d_net, real_data, fake_data, device = "cpu") {
  batch_size <- real_data$shape[1]


  # Sample random interpolation coefficients
  alpha <- torch::torch_rand(batch_size, 1, device = device)

  # Create interpolated samples between real and fake data
  interpolates <- (alpha * real_data + (1 - alpha) * fake_data)$requires_grad_(TRUE)

  # Get discriminator scores on interpolated samples
  d_interpolates <- d_net(interpolates)

  # Compute gradients of discriminator output w.r.t. interpolated inputs
  gradients <- torch::autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = torch::torch_ones_like(d_interpolates, device = device),
    create_graph = TRUE,
    retain_graph = TRUE
  )[[1]]

  # Compute gradient norm
  gradients <- gradients$view(c(batch_size, -1))
  gradient_norm <- gradients$norm(2, dim = 2)

  # Compute penalty: (||grad|| - 1)^2

  gradient_penalty <- ((gradient_norm - 1) ^ 2)$mean()

  return(gradient_penalty)
}
