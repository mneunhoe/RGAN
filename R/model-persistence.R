#' @title Save a Trained GAN
#'
#' @description Saves a trained GAN model to disk, including the generator,
#'   discriminator, optimizers, and all training settings. The model can be
#'   restored later using \code{\link{load_gan}}.
#'
#'   The function creates multiple files with the given path as base name:
#'   \itemize{
#'     \item \code{path_generator.pt} - Generator network weights
#'     \item \code{path_discriminator.pt} - Discriminator network weights
#'     \item \code{path_metadata.rds} - Settings, losses, and metadata
#'     \item \code{path_g_optim.pt} - Generator optimizer state (if include_optimizers=TRUE
#'     \item \code{path_d_optim.pt} - Discriminator optimizer state (if include_optimizers=TRUE)
#'   }
#'
#' @param trained_gan A trained GAN object of class "trained_RGAN" returned by \code{\link{gan_trainer}}
#' @param path The base file path for saving (without extension). Files will be created with suffixes.
#' @param include_optimizers Whether to include optimizer states for resuming training. Defaults to TRUE.
#'
#' @return Invisibly returns the base path where the model was saved
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
#' save_gan(trained_gan, "my_gan_model")
#'
#' # Load it back later
#' loaded_gan <- load_gan("my_gan_model")
#' }
save_gan <- function(trained_gan, path, include_optimizers = TRUE) {
  if (!inherits(trained_gan, "trained_RGAN")) {
    stop("trained_gan must be an object of class 'trained_RGAN' returned by gan_trainer()")
  }

  # Remove any file extension from path to use as base name
  path <- sub("\\.[^.]*$", "", path)

  # Save generator and discriminator networks directly
  torch::torch_save(trained_gan$generator, paste0(path, "_generator.pt"))
  torch::torch_save(trained_gan$discriminator, paste0(path, "_discriminator.pt"))

  # Optionally save optimizer states
  if (include_optimizers) {
    torch::torch_save(trained_gan$generator_optimizer$state_dict(), paste0(path, "_g_optim.pt"))
    torch::torch_save(trained_gan$discriminator_optimizer$state_dict(), paste0(path, "_d_optim.pt"))
  }

  # Save R metadata (settings, losses, etc.) separately
  metadata <- list(
    settings = trained_gan$settings,
    losses = trained_gan$losses,
    validation_metrics = trained_gan$validation_metrics,
    rgan_version = as.character(utils::packageVersion("RGAN")),
    torch_version = as.character(utils::packageVersion("torch")),
    saved_at = Sys.time(),
    include_optimizers = include_optimizers
  )
  saveRDS(metadata, paste0(path, "_metadata.rds"))

  message(sprintf("GAN model saved to: %s_*.pt/rds", path))
  invisible(path)
}


#' @title Load a Trained GAN
#'
#' @description Loads a trained GAN model that was previously saved using
#'   \code{\link{save_gan}}. The loaded model can be used for sampling
#'   synthetic data or continued training.
#'
#' @param path The base file path to the saved model (without extension, same as used in save_gan)
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
#' loaded_gan <- load_gan("my_gan_model")
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
  # Remove any file extension from path to use as base name
  path <- sub("\\.[^.]*$", "", path)

  # Check for required files
  generator_path <- paste0(path, "_generator.pt")
  discriminator_path <- paste0(path, "_discriminator.pt")
  metadata_path <- paste0(path, "_metadata.rds")

  if (!file.exists(generator_path)) {
    stop(sprintf("Generator file not found: %s", generator_path))
  }
  if (!file.exists(discriminator_path)) {
    stop(sprintf("Discriminator file not found: %s", discriminator_path))
  }
  if (!file.exists(metadata_path)) {
    stop(sprintf("Metadata file not found: %s", metadata_path))
  }

  # Load the networks
  g_net <- torch::torch_load(generator_path)
  d_net <- torch::torch_load(discriminator_path)

  # Move to specified device
  g_net <- g_net$to(device = device)
  d_net <- d_net$to(device = device)

  # Load metadata
  metadata <- readRDS(metadata_path)
  settings <- metadata$settings

  # Reconstruct optimizers
  g_optim <- torch::optim_adam(g_net$parameters, lr = settings$base_lr)
  d_optim <- torch::optim_adam(d_net$parameters, lr = settings$base_lr * settings$ttur_factor)

  # Load optimizer states if available
  g_optim_path <- paste0(path, "_g_optim.pt")
  d_optim_path <- paste0(path, "_d_optim.pt")

  if (metadata$include_optimizers && file.exists(g_optim_path) && file.exists(d_optim_path)) {
    g_optim$load_state_dict(torch::torch_load(g_optim_path))
    d_optim$load_state_dict(torch::torch_load(d_optim_path))
  }

  # Update device in settings
  settings$device <- device

  # Reconstruct the trained_RGAN object
  output <- list(
    generator = g_net,
    discriminator = d_net,
    generator_optimizer = g_optim,
    discriminator_optimizer = d_optim,
    losses = metadata$losses,
    validation_metrics = metadata$validation_metrics,
    settings = settings
  )
  class(output) <- "trained_RGAN"

  message(sprintf(
    "GAN model loaded from: %s (saved with RGAN v%s)",
    path,
    metadata$rgan_version
  ))

  return(output)
}
