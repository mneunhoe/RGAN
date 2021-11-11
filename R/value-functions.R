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
GAN_value_fct <- function(real_scores, fake_scores) {
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
