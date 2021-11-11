#' @title Uniform Random numbers between values a and b
#'
#' @description Provides a function to sample torch tensors from an arbitrary uniform distribution.
#'
#' @param shape Vector of dimensions of resulting tensor
#' @param a Lower bound of uniform distribution to sample from
#' @param b Upper bound of uniform distribution to sample from
#'
#' @return A sample from the specified uniform distribution in a tensor with the specified shape
#' @export
torch_rand_ab <- function(shape, a = -1, b = 1, ...) {
  (a-b) * torch::torch_rand(shape, ...) + b
}

#' @title KL WGAN loss on real examples
#'
#' @description Utility function for the kl WGAN loss for real examples as described in [https://arxiv.org/abs/1910.09779](https://arxiv.org/abs/1910.09779)
#'   and implemented in python in [https://github.com/ermongroup/f-wgan](https://github.com/ermongroup/f-wgan).
#'
#' @param dis_real Discriminator scores on real examples ($D(x)$)
#'
#' @return The loss
kl_real <- function(dis_real) {
  loss_real <- torch::torch_mean(torch::nnf_relu(1 - dis_real))

  return(loss_real)
}

#' @title KL WGAN loss on fake examples
#'
#' @description  Utility function for the kl WGAN loss for fake examples as described in [https://arxiv.org/abs/1910.09779](https://arxiv.org/abs/1910.09779)
#'   and implemented in python in [https://github.com/ermongroup/f-wgan](https://github.com/ermongroup/f-wgan).
#'
#' @param dis_fake Discriminator scores on fake examples ($D(G(z))$)
#'
#' @return The loss
kl_fake <- function(dis_fake) {
  dis_fake_norm = torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch::torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss_fake = torch::torch_mean(torch::nnf_relu(1. + dis_fake))

  return(loss_fake)
}

#' @title KL WGAN loss for Generator training
#'
#' @description Utility function for the kl WGAN loss for Generator training as described in [https://arxiv.org/abs/1910.09779](https://arxiv.org/abs/1910.09779)
#'   and implemented in python in [https://github.com/ermongroup/f-wgan](https://github.com/ermongroup/f-wgan).
#'
#' @param dis_fake Discriminator scores on fake examples ($D(G(z))$)
#'
#' @return The loss
kl_gen <- function(dis_fake) {
  dis_fake_norm = torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch::torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss = -torch::torch_mean(dis_fake)
  return(loss)
}
