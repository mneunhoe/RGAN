#' @title Sample Synthetic Data with explicit noise input
#'
#' @description Provides a function that makes it easy to sample synthetic data from a Generator
#'
#' @param g_net A torch::nn_module with a Generator
#' @param z A noise vector
#' @param device The device on which synthetic data should be sampled (cpu or cuda)
#' @param eval_dropout Should dropout be applied during inference
#'
#' @return Synthetic data
#' @export
expert_sample_synthetic_data <-
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


#' @title Sample Synthetic Data from a trained RGAN
#'
#' @description Provides a function that makes it easy to sample synthetic data from a Generator
#'
#' @param trained_gan A trained RGAN object of class "trained_RGAN"
#' @param transformer The transformer object used to pre-process the data
#'
#' @return Function to sample from a
#' @export
#' @examples
#' \dontrun{
#' # Before running the first time the torch backend needs to be installed
#' torch::install_torch()
#' # Load data
#' data <- sample_toydata()
#' # Build new transformer
#' transformer <- data_transformer$new()
#' # Fit transformer to data
#' transformer$fit(data)
#' # Transform data and store as new object
#' transformed_data <-  transformer$transform(data)
#' # Train the default GAN
#' trained_gan <- gan_trainer(transformed_data)
#' # Sample synthetic data from the trained GAN
#' synthetic_data <- sample_synthetic_data(trained_gan, transformer)
#' # Plot the results
#' GAN_update_plot(data = data,
#' synth_data = synthetic_data,
#' main = "Real and Synthetic Data after Training")
#' }
sample_synthetic_data <-
  function(trained_gan, transformer = NULL) {
    z <- trained_gan$settings$sample_noise(c(
      trained_gan$settings$synthetic_examples,
      trained_gan$settings$noise_dim
    ))$to(device = trained_gan$settings$device)


    if (trained_gan$settings$eval_dropout) {
      fake_data <-  trained_gan$generator(z)
    } else {
      trained_gan$generator$eval()
      fake_data <- trained_gan$generator(z)
      trained_gan$generator$train()
    }
    # Create an R array/matrix from the torch_tensor
    synth_data <- torch::as_array(fake_data$detach()$cpu())
    if (!is.null(transformer)) {
      synth_data <- transformer$inverse_transform(synth_data)
    }
    return(synth_data)
  }
