#' @title GAN_value_fct
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
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

#' @title WGAN_value_fct
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
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

#' @title FWGAN_value_fct
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
#' @export
FWGAN_value_fct <- function(real_scores, fake_scores) {
  d_loss <-
    kl_real(real_scores) + kl_fake(fake_scores)
  d_loss <- d_loss$mean()


  g_loss <-  kl_gen(fake_scores)

  g_loss <- g_loss$mean()

  return(list(d_loss = d_loss,
              g_loss = g_loss))

}



#' @title WGAN_weight_clipper
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
#' @export
WGAN_weight_clipper <- function(d_net) {
  for (parameter in names(d_net$parameters)) {
    d_net$parameters[[parameter]]$data()$clip_(-0.01, 0.01)
  }
}
