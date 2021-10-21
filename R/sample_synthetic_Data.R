#' @title sample_synthetic_data
#'
#' @description Provides a class to transform data for RGAN.
#'   Method `$new()` initializes a new transformer, method `$fit(data)` learns
#'   the parameters for the transformation from data (e.g. means and sds).
#'   Methods `$transform()` and `$inverse_transform()` can be used to transform
#'   and back transform a data set based on the learned parameters.
#'   Currently, DataTransformer supports z-transformation (a.k.a. normalization)
#'   for numerical features/variables and one hot encoding for categorical
#'   features/variables. In your call to fit you just need to indicate which
#'   columns contain discrete features.
#'
#' @return Function to sample from Generator
#' @export
sample_synthetic_data <-
  function(g_net,
           z,
           device
  ) {
    # Pass the noise through the Generator to create fake data
    fake_data <-  g_net(z)

    # Create an R array/matrix from the torch_tensor
    synth_data <- torch::as_array(fake_data$detach()$cpu())
    return(synth_data)
  }
