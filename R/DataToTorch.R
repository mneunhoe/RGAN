#' @title DataToTorch
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
#' @export
DataToTorch <- function(transformed_data, device = "cpu") {
  torch::torch_tensor(transformed_data)$to(device = device)
}


