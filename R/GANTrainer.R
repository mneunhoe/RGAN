#' @title GANTrainer
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
#' @export
GANTrainer <-
  function(data,
           noise_dim = 2,
           noise_distribution = "normal",
           value_function = "original",
           data_type = "tabular",
           generator = NULL,
           generator_optimizer = NULL,
           discriminator = NULL,
           discriminator_optimizer = NULL,
           weight_clipper = NULL,
           batch_size = 50,
           epochs = 150,
           plot = FALSE,
           plot_every = "epoch",
           eval_dropout = FALSE,
           synthetic_examples = 500,
           plot_dimensions = c(1, 2),
           device = "cpu") {



    ! (any(
      c("dataset", "matrix", "array", "torch_tensor") %in% class(data)
    ))

    if (!(any(
      c("dataset", "matrix", "array", "torch_tensor") %in% class(data)
    ))) {
      stop(
        "Data needs to be in correct format. \ntorch::dataset, matrix, array or torch::torch_tensor are permitted."
      )
    }

    if ((any(c("array", "matrix") %in% class(data)))) {
      data <- torch::torch_tensor(data)$to(device = "cpu")
      data_dim <- ncol(data)
      steps <- nrow(data) %/% batch_size
    }

    if("image_folder" %in% class(data)) {
      steps <- length(data$imgs[[1]]) %/% batch_size
    }


    plot_every <- ifelse(plot_every == "epoch", steps, plot_every)

    if (is.null(generator)) {
      g_net <-
        Generator(noise_dim = noise_dim,
                  data_dim = data_dim,
                  dropout_rate = 0.5)$to(device = device)
    } else {
      g_net <- generator
    }

    if (is.null(generator_optimizer)) {
      g_optim <- torch::optim_adam(g_net$parameters, lr = 0.0001)
    } else {
      g_optim <- generator_optimizer
    }

    if (is.null(discriminator)) {
      d_net <-
        Discriminator(data_dim = data_dim, dropout_rate = 0.5)$to(device = device)
    } else {
      d_net <- discriminator
    }

    if (is.null(discriminator_optimizer)) {
      d_optim <- torch::optim_adam(d_net$parameters, lr = 0.0001 * 4)
    } else {
      d_optim <- discriminator_optimizer
    }



    if (class(noise_distribution) == "function") {
      sample_noise <- noise_distribution
    } else {
      if (noise_distribution == "normal") {
        sample_noise <- torch::torch_randn
      }

      if (noise_distribution == "uniform") {
        sample_noise <- torch_rand_ab
      }
    }

    if (class(value_function) == "function") {
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
        value_fct <- FWGAN_value_fct

        weight_clipper <- function(d_net) {

        }

      }


    }






    fixed_z <-
      sample_noise(c(synthetic_examples, noise_dim))$to(device = device)

    for (i in 1:(epochs * steps)) {
      GAN_update_step(
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

      # This concludes one update step of the GAN. We will now repeat this many times.

      ###########################
      # Monitor Training Progress
      ###########################

      # During training we want to observe whether the GAN is learning anything useful.
      # Here we will create a simple message to the console and a plot after each epoch. That is when i %% steps == 0.


      if (i %% plot_every == 0) {
        # Print the current epoch to the console.
        cat("Update Step: ", i, "\n")


        if (plot) {
          # Create synthetic data for our plot. This synthetic data will always use the same noise sample -- fixed_z -- so it is easier for us to monitor training progress.
          synth_data <-
            sample_synthetic_data(g_net, fixed_z, device, eval_dropout = eval_dropout)

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
    }

    return(
      list(
        generator = g_net,
        discriminator = d_net,
        generator_optimizer = g_optim,
        discriminator_optimizer = d_optim
      )
    )

  }


#' @title GAN_update_step
#'
#' @description Provides a function to send the output of a DataTransformer to
#'   a torch tensor, so that it can be accessed during GAN training.
#'
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
#' @export
GAN_update_step <-
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
    # For each training iteration we need a fresh (mini-)batch from our data.

    # Then we subset the data set (x is the torch version of the data) to our fresh batch.
    real_data <- get_batch(data, batch_size, device)

    ###########################
    # Update the Discriminator
    ###########################

    # In a GAN we also need a noise sample for each training iteration.
    # torch_randn creates a torch object filled with draws from a standard normal distribution

    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)


    # Now our Generator net produces fake data based on the noise sample.
    # Since we want to update the Discriminator, we do not need to calculate the gradients of the Generator net.
    fake_data <- torch::with_no_grad(g_net(input = z))

    # The Discriminator net now computes the scores for fake and real data
    dis_real <- d_net(real_data)
    dis_fake <- d_net(fake_data)

    # We combine these scores to give our discriminator loss
    # d_loss <- kl_real(dis_real) + kl_fake(dis_fake)
    # d_loss <- d_loss$mean()

    # Gan loss
    # d_loss <- torch_log(dis_real) + torch_log(1-dis_fake)
    # d_loss <- -d_loss$mean()

    # # WGAN loss
    #

    d_loss <- value_function(dis_real, dis_fake)[["d_loss"]]

    # d_loss <-
    #   torch::torch_mean(dis_real) - torch::torch_mean(dis_fake)
    # d_loss <- -d_loss$mean()

    # Clip
    weight_clipper(d_net)
    # if (value_function == "wasserstein") {
    #   for (parameter in names(d_net$parameters)) {
    #     d_net$parameters[[parameter]]$data()$clip_(-0.01, 0.01)
    #   }
    #
    # }

    # What follows is one update step for the Discriminator net

    # First set all previous gradients to zero
    d_optim$zero_grad()

    # Pass the loss backward through the net
    d_loss$backward()

    # Take one step of the optimizer
    d_optim$step()

    ###########################
    # Update the Generator
    ###########################

    # To update the Generator we will use a fresh noise sample.
    # torch_randn creates a torch object filled with draws from a standard normal distribution

    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)


    # Now we can produce new fake data
    fake_data <- g_net(z)

    # The Discriminator now scores the new fake data
    dis_fake <- d_net(fake_data)

    # Now we can calculate the Generator loss
    # g_loss = kl_gen(dis_fake)

    # g_loss <- torch_log(1-dis_fake)
    #
    # g_loss = g_loss$mean()

    # WGAN loss

    # g_loss <- torch::torch_mean(dis_fake)
    # #
    # g_loss <- -g_loss$mean()

    g_loss <- value_function(dis_real, dis_fake)[["g_loss"]]
    # And take an update step of the Generator

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
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
#' @export
GAN_update_plot <-
  function(data,
           dimensions = c(1, 2),
           synth_data,
           epoch) {
    # Now we plot the training data.
    plot(
      torch::as_array(data$cpu())[, dimensions],
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
      main = paste0("Epoch: ", epoch),
      las = 1
    )
    # And we add the synthetic data on top.
    points(
      synth_data[, dimensions],
      bty = "n",
      col = viridis::viridis(2, alpha = 0.7)[2],
      pch = 19
    )
    # Finally a legend to understand the plot.
    legend(
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
#' @param transformed_data Input a data set after DataTransformer
#' @param device Input on which device (e.g. "cpu" or "cuda") will you be training?
#'
#' @return A function
#' @export
GAN_update_plot_image <-
  function(mfrow = c(4, 4),
           synth_data) {

    synth_data <- (synth_data + 1) / 2
    synth_data <- aperm(synth_data, c(1,3,4,2))
    par(mfrow = mfrow, mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
    lapply(lapply(lapply(1:(dim(synth_data)[1]), function(x) synth_data[x,,,]), as.raster), plot)

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
