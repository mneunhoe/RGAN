#' @title Save a Trained GAN
#'
#' @description Saves a trained GAN model to disk, including the generator,
#'   discriminator, optimizers, and all training settings. The model can be
#'   restored later using \code{\link{load_gan}}.
#'
#' @param trained_gan A trained GAN object of class "trained_RGAN" returned by \code{\link{gan_trainer}}
#' @param path The file path where the model should be saved (should end in .rds or .rgan)
#' @param include_optimizers Whether to include optimizer states for resuming training. Defaults to TRUE.
#'
#' @return Invisibly returns the path where the model was saved
#' @export
#'
#' @examples
#' \dontrun{
#' # Train a GAN
#' data <- sample_toydata()
#' transformer <- data_transformer$new()
#' transformer$fit(data)
#' transformed_data <- transformer$transform(data)
#' trained_gan <- gan_trainer(transformed_data, epochs = 10)
#'
#' # Save the trained GAN
#' save_gan(trained_gan, "my_gan_model.rgan")
#'
#' # Load it back later
#' loaded_gan <- load_gan("my_gan_model.rgan")
#' }
save_gan <- function(trained_gan, path, include_optimizers = TRUE) {
  if (!inherits(trained_gan, "trained_RGAN")) {
    stop("trained_gan must be an object of class 'trained_RGAN' returned by gan_trainer()")
  }

  # Create a list to store all model components
  model_data <- list(
    # Store network state dicts (weights)
    generator_state = trained_gan$generator$state_dict(),
    discriminator_state = trained_gan$discriminator$state_dict(),

    # Store settings needed to reconstruct the networks
    settings = trained_gan$settings,

    # Store losses if available
    losses = trained_gan$losses,

    # Metadata
    metadata = list(
      rgan_version = as.character(utils::packageVersion("RGAN")),
      torch_version = as.character(utils::packageVersion("torch")),
      saved_at = Sys.time(),
      include_optimizers = include_optimizers
    )
  )

  # Optionally include optimizer states for resuming training
 if (include_optimizers) {
    model_data$generator_optimizer_state <- trained_gan$generator_optimizer$state_dict()
    model_data$discriminator_optimizer_state <- trained_gan$discriminator_optimizer$state_dict()
  }

  # Save using torch's serialization for tensor compatibility
 torch::torch_save(model_data, path)

  message(sprintf("GAN model saved to: %s", path))
  invisible(path)
}


#' @title Load a Trained GAN
#'
#' @description Loads a trained GAN model that was previously saved using
#'   \code{\link{save_gan}}. The loaded model can be used for sampling
#'   synthetic data or continued training.
#'
#' @param path The file path to the saved model
#' @param device The device to load the model onto ("cpu", "cuda", or "mps").
#'   Defaults to "cpu". Use this to move a model trained on GPU to CPU or vice versa.
#'
#' @return A trained GAN object of class "trained_RGAN" that can be used with
#'   \code{\link{sample_synthetic_data}} or passed to \code{\link{gan_trainer}}
#'   for continued training.
#' @export
#'
#' @examples
#' \dontrun{
#' # Load a previously saved GAN
#' loaded_gan <- load_gan("my_gan_model.rgan")
#'
#' # Use it to generate synthetic data
#' transformer <- data_transformer$new()
#' # (fit transformer to original data or load it separately)
#' synthetic_data <- sample_synthetic_data(loaded_gan, transformer, n_samples = 100)
#'
#' # Or continue training
#' continued_gan <- gan_trainer(
#'   transformed_data,
#'   generator = loaded_gan$generator,
#'   discriminator = loaded_gan$discriminator,
#'   generator_optimizer = loaded_gan$generator_optimizer,
#'   discriminator_optimizer = loaded_gan$discriminator_optimizer,
#'   epochs = 50
#' )
#' }
load_gan <- function(path, device = "cpu") {
  if (!file.exists(path)) {
    stop(sprintf("File not found: %s", path))
  }

  # Load the saved model data
 model_data <- torch::torch_load(path)

  # Check for required components
  required <- c("generator_state", "discriminator_state", "settings", "metadata")
  missing <- setdiff(required, names(model_data))
  if (length(missing) > 0) {
    stop(sprintf("Invalid model file. Missing components: %s", paste(missing, collapse = ", ")))
  }

  settings <- model_data$settings

  # Reconstruct the generator network
  if (settings$data_type == "tabular") {
    # For tabular data, we need to infer data_dim from the state dict
    # The output layer weight shape tells us the data dimension
    output_weight <- model_data$generator_state$Output.weight
    data_dim <- output_weight$shape[1]
    noise_dim <- settings$noise_dim

    g_net <- Generator(
      noise_dim = noise_dim,
      data_dim = data_dim,
      dropout_rate = 0.5
    )$to(device = device)
  } else {
    # For image data, use DCGAN architecture
    g_net <- DCGAN_Generator(
      noise_dim = settings$noise_dim,
      dropout_rate = 0.5
    )$to(device = device)
  }

  # Load generator weights
  g_net$load_state_dict(model_data$generator_state)

  # Reconstruct the discriminator network
  if (settings$data_type == "tabular") {
    # Determine if sigmoid was used based on value function
    use_sigmoid <- identical(settings$value_function, "original")

    d_net <- Discriminator(
      data_dim = data_dim,
      dropout_rate = 0.5,
      sigmoid = use_sigmoid
    )$to(device = device)
  } else {
    use_sigmoid <- identical(settings$value_function, "original")
    d_net <- DCGAN_Discriminator(
      dropout_rate = 0.5,
      sigmoid = use_sigmoid
    )$to(device = device)
  }

  # Load discriminator weights
  d_net$load_state_dict(model_data$discriminator_state)

  # Reconstruct optimizers
  g_optim <- torch::optim_adam(g_net$parameters, lr = settings$base_lr)
  d_optim <- torch::optim_adam(d_net$parameters, lr = settings$base_lr * settings$ttur_factor)

  # Load optimizer states if available
  if (model_data$metadata$include_optimizers &&
      !is.null(model_data$generator_optimizer_state)) {
    g_optim$load_state_dict(model_data$generator_optimizer_state)
    d_optim$load_state_dict(model_data$discriminator_optimizer_state)
  }

  # Update device in settings
  settings$device <- device

  # Reconstruct the trained_RGAN object
  output <- list(
    generator = g_net,
    discriminator = d_net,
    generator_optimizer = g_optim,
    discriminator_optimizer = d_optim,
    losses = model_data$losses,
    settings = settings
  )
  class(output) <- "trained_RGAN"

  message(sprintf(
    "GAN model loaded from: %s (saved with RGAN v%s)",
    path,
    model_data$metadata$rgan_version
  ))

  return(output)
}
