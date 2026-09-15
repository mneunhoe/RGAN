#' @title GAN Value Function
#'
#' @description Implements the original GAN value function as a function to be called in gan_trainer.
#'   The function can serve as a template to implement new value functions in RGAN.
#'
#' @param real_scores The discriminator scores on real examples ($D(x)$)
#' @param fake_scores The discriminator scores on fake examples ($D(G(z))$)
#' @param epsilon Small constant for numerical stability to avoid log(0). Defaults to 1e-7.
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


#' @title Non-Saturating GAN Value Function
#'
#' @description Implements the non-saturating GAN loss of Goodfellow et al. (2014)
#'   directly on discriminator *logits*, using the numerically stable softplus
#'   formulation. Use it together with a discriminator that has a linear output
#'   layer (`sigmoid = FALSE`), which is the RGAN default for every value
#'   function other than `"original"`.
#'
#'   \deqn{L_D = E[softplus(-D(x))] + E[softplus(D(G(z)))]}
#'   \deqn{L_G = E[softplus(-D(G(z)))]}
#'
#' @param real_scores The discriminator scores (logits) on real examples (\eqn{D(x)})
#' @param fake_scores The discriminator scores (logits) on fake examples (\eqn{D(G(z))})
#'
#' @references Goodfellow, I. et al. (2014). Generative Adversarial Nets.
#'   *Advances in Neural Information Processing Systems 27*.
#'
#' @return The function returns a named list with the entries d_loss and g_loss
#' @export
NSGAN_value_fct <- function(real_scores, fake_scores) {
  d_loss <- torch::nnf_softplus(-real_scores)$mean() +
    torch::nnf_softplus(fake_scores)$mean()

  g_loss <- torch::nnf_softplus(-fake_scores)$mean()

  return(list(d_loss = d_loss,
              g_loss = g_loss))
}


#' @title Hinge GAN Value Function
#'
#' @description Implements the hinge loss for GAN training (Lim & Ye 2017;
#'   Tran et al. 2017), as used by SAGAN and BigGAN. Expects a discriminator
#'   with a linear output layer (`sigmoid = FALSE`).
#'
#'   \deqn{L_D = E[relu(1 - D(x))] + E[relu(1 + D(G(z)))]}
#'   \deqn{L_G = -E[D(G(z))]}
#'
#' @param real_scores The discriminator scores (logits) on real examples (\eqn{D(x)})
#' @param fake_scores The discriminator scores (logits) on fake examples (\eqn{D(G(z))})
#'
#' @references Lim, J. H. & Ye, J. C. (2017). Geometric GAN. arXiv:1705.02894.
#'
#' @return The function returns a named list with the entries d_loss and g_loss
#' @export
hinge_value_fct <- function(real_scores, fake_scores) {
  d_loss <- torch::nnf_relu(1 - real_scores)$mean() +
    torch::nnf_relu(1 + fake_scores)$mean()

  g_loss <- -fake_scores$mean()

  return(list(d_loss = d_loss,
              g_loss = g_loss))
}


#' @title Relativistic Pairing GAN (RpGAN) Value Function
#'
#' @description Implements the relativistic pairing GAN loss
#'   (Jolicoeur-Martineau 2019), which is the loss used by R3GAN
#'   (Huang et al., NeurIPS 2024). Instead of scoring real and fake examples
#'   independently, the loss only depends on the *difference* between the score
#'   of a real example and the score of a fake example:
#'
#'   \deqn{L_D = E[softplus(-(D(x) - D(G(z))))]}
#'   \deqn{L_G = E[softplus(D(x) - D(G(z)))]}
#'
#'   Together with zero-centred gradient penalties (see [r1_penalty()] and
#'   [r2_penalty()]) this objective is well-behaved and, in the R3GAN paper,
#'   removes mode collapse without any of the usual GAN tricks.
#'
#'   **Pairing is by batch index**: element `i` of `real_scores` is paired with
#'   element `i` of `fake_scores`. The two score tensors must therefore have the
#'   same length. This is always the case in [gan_update_step()], where the real
#'   batch and the fake batch have the same size (and, with `pac > 1`, the same
#'   number of packs).
#'
#'   Expects a discriminator with a linear output layer (`sigmoid = FALSE`).
#'
#' @param real_scores The discriminator scores (logits) on real examples (\eqn{D(x)})
#' @param fake_scores The discriminator scores (logits) on fake examples (\eqn{D(G(z))}),
#'   of the same length as `real_scores`
#'
#' @references
#'   Jolicoeur-Martineau, A. (2019). The relativistic discriminator: a key
#'   element missing from standard GAN. *ICLR 2019*. arXiv:1807.00734.
#'
#'   Huang, Y., Gokaslan, A., Kuleshov, V. & Tompkin, J. (2024). The GAN is
#'   dead; long live the GAN! A Modern GAN Baseline. *NeurIPS 2024*.
#'   arXiv:2501.05441.
#'
#' @return The function returns a named list with the entries d_loss and g_loss
#' @export
RpGAN_value_fct <- function(real_scores, fake_scores) {
  if (!identical(as.integer(real_scores$numel()),
                 as.integer(fake_scores$numel()))) {
    stop(
      "RpGAN_value_fct pairs real and fake scores by batch index, so the real and fake batches must have the same size."
    )
  }

  diff <- real_scores - fake_scores

  d_loss <- torch::nnf_softplus(-diff)$mean()
  g_loss <- torch::nnf_softplus(diff)$mean()

  return(list(d_loss = d_loss,
              g_loss = g_loss))
}
# RpGAN is the only built-in value function whose *generator* loss depends on
# the real scores. gan_trainer() uses this attribute to recompute the real
# scores (under the updated discriminator) before the generator step.
attr(RpGAN_value_fct, "requires_real_scores") <- TRUE


#' @title Zero-Centred Gradient Penalty on Real Data (R1)
#'
#' @description Computes the R1 gradient penalty of Mescheder et al. (2018):
#'   the mean over the batch of the squared L2 norm of the gradient of the
#'   discriminator output with respect to its *real* input,
#'   \eqn{E[ \| \nabla_x D(x) \|^2 ]}.
#'
#'   Unlike the WGAN-GP penalty (see [gradient_penalty()]), which pushes the
#'   gradient norm towards 1 on interpolates, this penalty is *zero-centred*:
#'   it pushes the gradient norm towards 0 on the data manifold. R1 together
#'   with R2 (see [r2_penalty()]) and the relativistic pairing loss
#'   ([RpGAN_value_fct()]) is the R3GAN recipe.
#'
#'   The trainer multiplies the returned value by `gamma / 2`.
#'
#'   The gradient is computed with `create_graph = TRUE`, so the penalty is
#'   itself differentiable with respect to the discriminator parameters. Note
#'   that the discriminator's *output bias* receives no gradient from this
#'   penalty (the input-gradient does not depend on it); that is expected and
#'   harmless.
#'
#' @param d_net The discriminator network (torch::nn_module)
#' @param real_data Real data samples (torch_tensor). With PacGAN this should be
#'   the packed tensor that is actually fed to the discriminator.
#' @param device The device to use ("cpu", "cuda", or "mps")
#'
#' @references Mescheder, L., Geiger, A. & Nowozin, S. (2018). Which Training
#'   Methods for GANs do actually Converge? *ICML 2018*. arXiv:1801.04406.
#'
#' @return The penalty (torch_tensor, scalar)
#' @export
r1_penalty <- function(d_net, real_data, device = "cpu") {
  x <- real_data$detach()$clone()$requires_grad_(TRUE)

  out <- d_net(x)

  gradients <- torch::autograd_grad(
    outputs = out,
    inputs = x,
    grad_outputs = torch::torch_ones_like(out, device = device),
    create_graph = TRUE,
    retain_graph = TRUE,
    allow_unused = TRUE
  )[[1]]

  # A discriminator whose output does not depend on its input (e.g. a constant
  # net) produces no gradient at all; the penalty is then exactly zero.
  # (With allow_unused = TRUE torch returns an *undefined* tensor in that case.
  # It throws on most operations but reports zero elements, which is the
  # portable way to detect it.)
  n_grad <- tryCatch(as.numeric(gradients$numel()), error = function(e) 0)
  undefined <- is.null(gradients) || !inherits(gradients, "torch_tensor") ||
    n_grad == 0

  if (undefined) {
    return(torch::torch_zeros(1, device = device)$sum())
  }

  gradients <- gradients$view(c(x$shape[1], -1))

  gradients$pow(2)$sum(dim = 2)$mean()
}


#' @title Zero-Centred Gradient Penalty on Fake Data (R2)
#'
#' @description Computes the R2 gradient penalty, the mirror image of
#'   [r1_penalty()] evaluated on generated samples:
#'   \eqn{E[ \| \nabla_x D(G(z)) \|^2 ]}.
#'
#'   The trainer multiplies the returned value by `gamma / 2`. The fake data is
#'   detached inside the function, so no gradient flows back to the generator.
#'   As for R1, the discriminator's output bias receives no gradient from this
#'   penalty.
#'
#' @param d_net The discriminator network (torch::nn_module)
#' @param fake_data Generated samples (torch_tensor). With PacGAN this should be
#'   the packed tensor that is actually fed to the discriminator.
#' @param device The device to use ("cpu", "cuda", or "mps")
#'
#' @references Huang, Y., Gokaslan, A., Kuleshov, V. & Tompkin, J. (2024). The
#'   GAN is dead; long live the GAN! A Modern GAN Baseline. *NeurIPS 2024*.
#'   arXiv:2501.05441.
#'
#' @return The penalty (torch_tensor, scalar)
#' @export
r2_penalty <- function(d_net, fake_data, device = "cpu") {
  r1_penalty(d_net, fake_data, device = device)
}
