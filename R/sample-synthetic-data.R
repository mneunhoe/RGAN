#' @title Sample Synthetic Data
#'
#' @description Provides a function that makes it easy to sample synthetic data from a Generator
#'
#' @param g_net A torch::nn_module with a Generator
#' @param z A noise vector
#' @param device The device on which synthetic data should be sampled (cpu or cuda)
#' @param eval_dropout Should dropout be applied during inference
#'
#' @return Function to sample from Generator given a noise vector z
#' @export
sample_synthetic_data <-
  function(g_net,
           z,
           device,
           eval_dropout = FALSE) {
    # Pass the noise through the Generator to create fake data

    if (eval_dropout) {
      fake_data <-  g_net(z)
    } else {
      g_net$eval()
      fake_data <-  g_net(z)
      g_net$train()
    }
    # Create an R array/matrix from the torch_tensor
    synth_data <- torch::as_array(fake_data$detach()$cpu())
    return(synth_data)
  }
