
<!-- README.md is generated from README.Rmd. Please edit that file -->

# RGAN

<!-- badges: start -->
[![CRAN status](https://www.r-pkg.org/badges/version/RGAN)](https://CRAN.R-project.org/package=RGAN)
[![R-CMD-check](https://github.com/mneunhoe/RGAN/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/mneunhoe/RGAN/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

The goal of RGAN is to facilitate training and experimentation with
Generative Adversarial Nets (GAN) in R. RGAN provides tools for
generating synthetic tabular data with state-of-the-art techniques
including differential privacy and post-GAN boosting.

## Features

- **Easy GAN Training**: Train GANs on tabular data with a single function call
- **Differentially Private Training**: Train GANs with formal privacy guarantees using DP-SGD
- **Post-GAN Boosting**: Improve synthetic data quality using discriminator ensembles
- **Flexible Data Transformation**: Support for continuous, categorical, and mixed data types with mode-specific normalization
- **Multiple Architectures**: Built-in support for various generator and discriminator architectures
- **Loss Functions**: WGAN-GP, standard GAN, KL-WGAN, and more

## Installation

You can install the released version of RGAN from
[CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("RGAN")
```

And the development version from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("mneunhoe/RGAN")
```

## Example

This is a basic example which shows you how to train a GAN and observe
training progress on toy data.

Before running RGAN for the first time you need to make sure that torch
is properly installed:

``` r
install.packages("torch")
#> Installing package into '/private/var/folders/z8/wk0vgp996m74v0g_x797qzf00000gn/T/RtmppUYncE/temp_libpath7cc3448fcda2'
#> (as 'lib' is unspecified)
#> 
#> The downloaded binary packages are in
#>  /var/folders/z8/wk0vgp996m74v0g_x797qzf00000gn/T//Rtmpa6rd9N/downloaded_packages
library(torch)
```

Then you can get started to train a GAN on toy data (or potentially your
own data).

``` r
library(RGAN)

# Sample some toy data to play with.
data <- sample_toydata()

# Transform (here standardize) the data to facilitate learning.
# First, create a new data transformer.
transformer <- data_transformer$new()

# Fit the transformer to your data.
transformer$fit(data)

# Use the fitted transformer to transform your data.
transformed_data <- transformer$transform(data)

# Have a look at the transformed data.
par(mfrow = c(3, 2))
plot(
  transformed_data,
  bty = "n",
  col = viridis::viridis(2, alpha = 0.7)[1],
  pch = 19,
  xlab = "Var 1",
  ylab = "Var 2",
  main = "The Real Data",
  las = 1
)

# Set the device you want to train on.
# First, we check whether a compatible GPU is available for computation.
use_cuda <- torch::cuda_is_available()

# If so we would use it to speed up training (especially for models with image data).
device <- ifelse(use_cuda, "cuda", "cpu")

# Now train the GAN and observe some intermediate results.
res <-
  gan_trainer(
    transformed_data,
    eval_dropout = TRUE,
    plot_progress = TRUE,
    plot_interval = 600,
    device = device
  )
#> Training the GAN ■■                                 3% | ETA:  1m
#> Training the GAN ■■                                 5% | ETA:  1m
#> Training the GAN ■■■■                              10% | ETA:  1m
#> Training the GAN ■■■■■■                            16% | ETA: 48s
#> Training the GAN ■■■■■■■                           21% | ETA: 45s
#> Training the GAN ■■■■■■■■■                         26% | ETA: 42s
#> Training the GAN ■■■■■■■■■■                        32% | ETA: 39s
#> Training the GAN ■■■■■■■■■■■■                      37% | ETA: 36s
#> Training the GAN ■■■■■■■■■■■■■■                    42% | ETA: 32s
#> Training the GAN ■■■■■■■■■■■■■■■                   48% | ETA: 30s
#> Training the GAN ■■■■■■■■■■■■■■■■■                 53% | ETA: 27s
#> Training the GAN ■■■■■■■■■■■■■■■■■■                58% | ETA: 24s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■              63% | ETA: 21s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■             68% | ETA: 19s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■■■           73% | ETA: 16s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■■■■          78% | ETA: 13s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■■■■■■        83% | ETA: 10s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■■■■■■■       88% | ETA:  7s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■     93% | ETA:  4s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■    98% | ETA:  1s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■  100% | ETA:  0s
```

<img src="man/figures/README-RGAN-example-1.png" width="100%" />

After training you can work with the resulting GAN to sample synthetic
data, or potentially keep training for further steps.

If you want to sample synthetic data from your GAN you need to provide a
GAN Generator and a noise vector (that needs to be a torch tensor and
should come from the same distribution that you used during training).
For example, we could look at the difference of synthetic data generated
with and without dropout during generation/inference (using the same
noise vector).

``` r
par(mfrow = c(1, 2))

# Set the noise vector.
noise_vector <- torch::torch_randn(c(nrow(transformed_data), 2))$to(device = device)

# Generate synthetic data from the trained generator with dropout during generation.
synth_data_dropout <- expert_sample_synthetic_data(res$generator, noise_vector,eval_dropout = TRUE)

# Plot data and synthetic data
GAN_update_plot(data = transformed_data, synth_data = synth_data_dropout, main = "With dropout")

synth_data_no_dropout <- expert_sample_synthetic_data(res$generator, noise_vector,eval_dropout = F)

GAN_update_plot(data = transformed_data, synth_data = synth_data_no_dropout, main = "Without dropout")
```

<img src="man/figures/README-sampling-data-1.png" width="100%" />

If you want to continue training you can pass the generator,
discriminator as well as the respective optimizers to gan_trainer like
that:

``` r
res_cont <- gan_trainer(transformed_data,
                   generator = res$generator,
                   discriminator = res$discriminator,
                   generator_optimizer = res$generator_optimizer,
                   discriminator_optimizer = res$discriminator_optimizer,
                   epochs = 10
                   )
#> Training the GAN ■■■■■■■■■■■■■■■■                  50% | ETA:  2s
#> Training the GAN ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■  100% | ETA:  0s
```

## Differentially Private Training

RGAN supports training GANs with formal differential privacy guarantees
using DP-SGD:

``` r
# Train with differential privacy
dp_gan <- dp_gan_trainer(
  transformed_data,
  epochs = 50,
  epsilon = 1.0,           # Privacy budget
  delta = 1e-5,            # Privacy parameter
  noise_multiplier = 1.1,  # Noise for DP-SGD
  max_grad_norm = 1.0      # Gradient clipping bound
)
```

See `vignette("dp-gan-training")` for details on privacy parameters and
best practices.

## Post-GAN Boosting

Improve synthetic data quality using discriminator ensembles captured
during training:

``` r
# Train with checkpointing
trained_gan <- gan_trainer(
  transformed_data,
  checkpoint_epochs = 10  # Save checkpoints every 10 epochs
)

# Apply post-GAN boosting
boosted <- apply_post_gan_boosting(
  trained_gan,
  real_data = transformed_data,
  transformer = transformer
)
```

See `vignette("post-gan-boosting")` for the full workflow and DP
boosting options.

## Vignettes

RGAN includes detailed vignettes covering advanced usage:

- **[Training a State-of-the-Art GAN on the Adult
  Dataset](https://mneunhoe.github.io/RGAN/articles/adult-sota-gan.html)**:
  Complete example with mixed data types, WGAN-GP, and PacGAN
- **[Training GANs with Differential
  Privacy](https://mneunhoe.github.io/RGAN/articles/dp-gan-training.html)**:
  Privacy-preserving GAN training with DP-SGD
- **[Improving Synthetic Data Quality with Post-GAN
  Boosting](https://mneunhoe.github.io/RGAN/articles/post-gan-boosting.html)**:
  Using discriminator ensembles to select high-quality samples

Browse all vignettes with `browseVignettes("RGAN")`.

## Citation

If you use RGAN in your research, please cite:

    @software{neunhoeffer2024rgan,
      author = {Neunhoeffer, Marcel},
      title = {RGAN: Generative Adversarial Nets in R},
      year = {2024},
      url = {https://github.com/mneunhoe/RGAN}
    }

For the post-GAN boosting methodology:

    @inproceedings{neunhoeffer2021private,
      title={Private Post-GAN Boosting},
      author={Neunhoeffer, Marcel and Wu, Zhiwei Steven and Dwork, Cynthia},
      booktitle={International Conference on Learning Representations},
      year={2021}
    }
