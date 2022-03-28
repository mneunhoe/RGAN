#' @title gan_trainer
#'
#' @description Provides a function to quickly train a GAN model.
#'
#' @param data Input a data set. Needs to be a matrix, array, torch::torch_tensor or torch::dataset.
#' @param noise_dim The dimensions of the GAN noise vector z. Defaults to 2.
#' @param noise_distribution The noise distribution. Expects a function that samples from a distribution and returns a torch_tensor. For convenience "normal" and "uniform" will automatically set a function. Defaults to "normal".
#' @param value_function The value function for GAN training. Expects a function that takes discriminator scores of real and fake data as input and returns a list with the discriminator loss and generator loss. For reference see: . For convenience three loss functions "original", "wasserstein" and "f-wgan" are already implemented. Defaults to "original".
#' @param data_type "tabular" or "image", controls the data type, defaults to "tabular".
#' @param generator The generator network. Expects a neural network provided as torch::nn_module. Default is NULL which will create a simple fully connected neural network.
#' @param generator_optimizer The optimizer for the generator network. Expects a torch::optim_xxx function, e.g. torch::optim_adam(). Default is NULL which will setup `torch::optim_adam(g_net$parameters, lr = base_lr)`.
#' @param discriminator The discriminator network. Expects a neural network provided as torch::nn_module. Default is NULL which will create a simple fully connected neural network.
#' @param discriminator_optimizer The optimizer for the generator network. Expects a torch::optim_xxx function, e.g. torch::optim_adam(). Default is NULL which will setup `torch::optim_adam(g_net$parameters, lr = base_lr * ttur_factor)`.
#' @param base_lr The base learning rate for the optimizers. Default is 0.0001. Only used if no optimizer is explicitly passed to the trainer.
#' @param ttur_factor A multiplier for the learning rate of the discriminator, to implement the two time scale update rule.
#' @param weight_clipper The wasserstein GAN puts some constraints on the weights of the discriminator, therefore weights are clipped during training.
#' @param batch_size The number of training samples selected into the mini batch for training. Defaults to 50.
#' @param epochs The number of training epochs. Defaults to 150.
#' @param plot_progress Monitor training progress with plots. Defaults to FALSE.
#' @param plot_interval Number of training steps between plots. Input number of steps or "epoch". Defaults to "epoch".
#' @param eval_dropout Should dropout be applied during the sampling of synthetic data? Defaults to FALSE.
#' @param synthetic_examples Number of synthetic examples that should be generated. Defaults to 500. For image data e.g. 16 would be more reasonable.
#' @param plot_dimensions If you monitor training progress with a plot which dimensions of the data do you want to look at? Defaults to c(1, 2), i.e. the first two columns of the tabular data.
#' @param device Input on which device (e.g. "cpu" or "cuda") training should be done. Defaults to "cpu".
#'
#' @return gan_trainer trains the neural networks and returns an object of class trained_RGAN that contains the last generator, discriminator and the respective optimizers, as well as the settings.
#' @export
#'
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
gan_trainer <-
  function(data,
           noise_dim = 2,
           noise_distribution = "normal",
           value_function = "original",
           data_type = "tabular",
           generator = NULL,
           generator_optimizer = NULL,
           discriminator = NULL,
           discriminator_optimizer = NULL,
           base_lr = 0.0001,
           ttur_factor = 4,
           weight_clipper = NULL,
           batch_size = 50,
           epochs = 150,
           plot_progress = FALSE,
           plot_interval = "epoch",
           eval_dropout = FALSE,
           synthetic_examples = 500,
           plot_dimensions = c(1, 2),
           device = "cpu") {
# Check if data is in the correct format ---------------------------------------
    !(any(
      c("dataset", "matrix", "array", "torch_tensor") %in% class(data)
    ))

    if (!(any(
      c("dataset", "matrix", "array", "torch_tensor") %in% class(data)
    ))) {
      stop(
        "Data needs to be in correct format. \ntorch::dataset, matrix, array or torch::torch_tensor are permitted."
      )
    }

# Calculate the number of steps per epoch --------------------------------------
    if ((any(c("array", "matrix") %in% class(data)))) {
      data <- torch::torch_tensor(data)$to(device = "cpu")
      data_dim <- ncol(data)
      steps <- nrow(data) %/% batch_size
    }

    if("image_folder" %in% class(data)) {
      steps <- length(data$imgs[[1]]) %/% batch_size
    }

# Set the plotting interval ----------------------------------------------------
    plot_interval <- ifelse(plot_interval == "epoch", steps, plot_interval)

# Set up the neural networks if none are provided ------------------------------
    if (is.null(generator)) {
      g_net <-
        Generator(noise_dim = noise_dim,
                  data_dim = data_dim,
                  dropout_rate = 0.5)$to(device = device)
    } else {
      g_net <- generator
    }

    if (is.null(generator_optimizer)) {
      g_optim <- torch::optim_adam(g_net$parameters, lr = base_lr)
    } else {
      g_optim <- generator_optimizer
    }

    if (is.null(discriminator)) {
      if(value_function != "original") {
      d_net <-
        Discriminator(data_dim = data_dim, dropout_rate = 0.5)$to(device = device)
      } else {
        d_net <-
          Discriminator(data_dim = data_dim, dropout_rate = 0.5, sigmoid = TRUE)$to(device = device)
      }
    } else {
      d_net <- discriminator
    }

    if (is.null(discriminator_optimizer)) {
      d_optim <- torch::optim_adam(d_net$parameters, lr = base_lr * ttur_factor)
    } else {
      d_optim <- discriminator_optimizer
    }


# Define the noise distribution for the generator ------------------------------
    if (inherits(noise_distribution, "function")) {
      sample_noise <- noise_distribution
    } else {
      if (noise_distribution == "normal") {
        sample_noise <- torch::torch_randn
      }

      if (noise_distribution == "uniform") {
        sample_noise <- torch_rand_ab
      }
    }

# Define the value function ----------------------------------------------------
    if (inherits(value_function, "function")) {
      value_fct <- value_function
    } else {
      if (is.null(value_function) | value_function == "original") {
        value_fct <- GAN_value_fct

        weight_clipper <- function(d_net) {

        }
      }

      if (value_function == "wasserstein") {
        value_fct <- WGAN_value_fct

        weight_clipper <- WGAN_weight_clipper

      }
      if (value_function == "f-wgan") {
        value_fct <- KLWGAN_value_fct

        weight_clipper <- function(d_net) {

        }

      }


    }





# Sample a fixed noise vector to observe training progress ---------------------
    fixed_z <-
      sample_noise(c(synthetic_examples, noise_dim))$to(device = device)

# Initialize progress bar ------------------------------------------------------
    cli::cli_progress_bar("Training the GAN", total = epochs * steps)
# Start GAN training loop ------------------------------------------------------
    for (i in 1:(epochs * steps)) {
      gan_update_step(
        data,
        batch_size,
        noise_dim,
        sample_noise,
        device,
        g_net,
        d_net,
        g_optim,
        d_optim,
        value_fct,
        weight_clipper
      )

      cli::cli_progress_update()

        if (plot_progress & i %% plot_interval == 0) {
# Create synthetic data for our plot.
          synth_data <-
            expert_sample_synthetic_data(g_net, fixed_z, device, eval_dropout = eval_dropout)

          if (data_type == "tabular") {
            GAN_update_plot(
              data = data,
              dimensions = plot_dimensions,
              synth_data = synth_data,
              epoch = i %/% steps
            )
          }

          if (data_type == "image") {
            GAN_update_plot_image(synth_data = synth_data)
          }
        }

    }

    output <-  list(
      generator = g_net,
      discriminator = d_net,
      generator_optimizer = g_optim,
      discriminator_optimizer = d_optim,
      settings = list(noise_dim = noise_dim,
                      noise_distribution = noise_distribution,
                      sample_noise = sample_noise,
                      value_function = value_function,
                      data_type = data_type,
                      base_lr = base_lr,
                      ttur_factor = ttur_factor,
                      weight_clipper = weight_clipper,
                      batch_size = batch_size,
                      epochs = epochs,
                      plot_progress = plot_progress,
                      plot_interval = plot_interval,
                      eval_dropout = eval_dropout,
                      synthetic_examples = synthetic_examples,
                      plot_dimensions = plot_dimensions,
                      device = device)
    )
    class(output) <- "trained_RGAN"
    return(
     output
    )

  }


#' @title gan_update_step
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param data Input a data set. Needs to be a matrix, array, torch::torch_tensor or torch::dataset.
#' @param batch_size The number of training samples selected into the mini batch for training. Defaults to 50.
#' @param noise_dim The dimensions of the GAN noise vector z. Defaults to 2.
#' @param sample_noise A function to sample noise to a torch::tensor
#' @param device Input on which device (e.g. "cpu" or "cuda") training should be done. Defaults to "cpu".
#' @param g_net The generator network. Expects a neural network provided as torch::nn_module. Default is NULL which will create a simple fully connected neural network.
#' @param g_optim The optimizer for the generator network. Expects a torch::optim_xxx function, e.g. torch::optim_adam(). Default is NULL which will setup `torch::optim_adam(g_net$parameters, lr = base_lr)`.
#' @param d_net The discriminator network. Expects a neural network provided as torch::nn_module. Default is NULL which will create a simple fully connected neural network.
#' @param d_optim The optimizer for the generator network. Expects a torch::optim_xxx function, e.g. torch::optim_adam(). Default is NULL which will setup `torch::optim_adam(g_net$parameters, lr = base_lr * ttur_factor)`.
#' @param value_function The value function for GAN training. Expects a function that takes discriminator scores of real and fake data as input and returns a list with the discriminator loss and generator loss. For reference see: . For convenience three loss functions "original", "wasserstein" and "f-wgan" are already implemented. Defaults to "original".
#' @param weight_clipper The wasserstein GAN puts some constraints on the weights of the discriminator, therefore weights are clipped during training.
#'
#' @return A function
#' @export
gan_update_step <-
  function(data,
           batch_size,
           noise_dim,
           sample_noise,
           device = "cpu",
           g_net,
           d_net,
           g_optim,
           d_optim,
           value_function,
           weight_clipper) {
    # Get a fresh batch of data ------------------------------------------------
    real_data <- get_batch(data, batch_size, device)

    # Get a fresh noise sample -------------------------------------------------
    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)
    # Produce fake data from noise ---------------------------------------------
    fake_data <- torch::with_no_grad(g_net(input = z))
    # Compute the discriminator scores on real and fake data -------------------
    dis_real <- d_net(real_data)
    dis_fake <- d_net(fake_data)
    # Calculate the discriminator loss
    d_loss <- value_function(dis_real, dis_fake)[["d_loss"]]
    # Clip weights according to weight_clipper ---------------------------------
    weight_clipper(d_net)
    # What follows is one update step for the discriminator net-----------------
    # First set all previous gradients to zero
    d_optim$zero_grad()

    # Pass the loss backward through the net
    d_loss$backward()

    # Take one step of the optimizer
    d_optim$step()

    # Update the generator -----------------------------------------------------
    # Get a fresh noise sample -------------------------------------------------
    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)
    # Produce fake data --------------------------------------------------------
    fake_data <- g_net(z)

    # Calculate discriminator score for fake data ------------------------------
    dis_fake <- d_net(fake_data)
    # Get generator loss based on scores ---------------------------------------
    g_loss <- value_function(dis_real, dis_fake)[["g_loss"]]
    # What follows is one update step for the generator net --------------------
    # First set all previous gradients to zero
    g_optim$zero_grad()

    # Pass the loss backward through the net
    g_loss$backward()

    # Take one step of the optimizer
    g_optim$step()
  }


#' @title GAN_update_plot
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param data Real data to be plotted
#' @param dimensions Which columns of the data should be plotted
#' @param synth_data The synthetic data to be plotted
#' @param epoch The epoch during training for the plot title
#' @param main An optional plot title
#'
#' @return A function
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
GAN_update_plot <-
  function(data,
           dimensions = c(1, 2),
           synth_data,
           epoch,
           main = NULL) {

    if("torch_tensor" %in% class(data)) {
      data <- torch::as_array(data$cpu())
    }
    # Now we plot the training data.
    plot(
      data[, dimensions],
      bty = "n",
      col = viridis::viridis(2, alpha = 0.7)[1],
      #xlim = c(-50, 50),
      pch = 19,
      xlab = ifelse(
        !is.null(colnames(data)),
        colnames(data)[dimensions[1]],
        paste0("Var ", dimensions[1])
      ),
      ylab = ifelse(
        !is.null(colnames(data)),
        colnames(data)[dimensions[2]],
        paste0("Var ", dimensions[2])
      ),
      main = ifelse(is.null(main), paste0("Epoch: ", epoch), main),
      las = 1
    )
    # And we add the synthetic data on top.
    graphics::points(
      synth_data[, dimensions],
      bty = "n",
      col = viridis::viridis(2, alpha = 0.7)[2],
      pch = 19
    )
    # Finally a legend to understand the plot.
    graphics::legend(
      "topleft",
      bty = "n",
      pch = 19,
      col = viridis::viridis(2),
      legend = c("Real", "Synthetic")
    )
  }


#' @title GAN_update_plot_image
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param mfrow The dimensions of the grid of images to be plotted
#' @param synth_data The synthetic data (images) to be plotted
#'
#' @return A function
#' @export
GAN_update_plot_image <-
  function(mfrow = c(4, 4),
           synth_data) {

    synth_data <- (synth_data + 1) / 2
    synth_data <- aperm(synth_data, c(1,3,4,2))
    oldpar <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(oldpar))
    graphics::par(mfrow = mfrow, mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
    lapply(lapply(lapply(1:(dim(synth_data)[1]), function(x) synth_data[x,,,]), grDevices::as.raster), plot)

  }

get_batch <- function(dataset, batch_size, device = "cpu") {
  if ("dataset" %in% class(dataset)) {
    dataloader <-
      torch::dataloader(dataset,
                        batch_size,
                        shuffle = TRUE,
                        num_workers = 0)
    torch::dataloader_next(torch::dataloader_make_iter(dataloader))$x$to(device = device)
  } else {
    dataset[sample(nrow(dataset), size = batch_size)]$to(device = device)
  }
}
