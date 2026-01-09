# RGAN 0.2.0

## Major New Features

### Differentially Private GAN Training

* New `dp_gan_trainer()` function for training GANs with formal differential privacy
  guarantees using DP-SGD (Differentially Private Stochastic Gradient Descent)
* Privacy mechanisms include:

  - Poisson subsampling for privacy amplification
  - Per-sample gradient clipping to bound sensitivity
  - Calibrated Gaussian noise addition
  - Rényi Differential Privacy (RDP) accounting for tight composition
* New `secure_rng` parameter to choose between cryptographically secure RNG (OpenDP)

  for production use or fast torch RNG for development/testing
* Privacy budget pre-calculation with `compute_max_steps()` for efficient training
* New `calibrate_noise_multiplier()` to find optimal noise for target epsilon
* New vignette: "Training GANs with Differential Privacy"

### Post-GAN Boosting

* New `apply_post_gan_boosting()` wrapper for easy post-processing of GAN samples
* New `compute_discriminator_scores()` for evaluating samples across checkpoints
* Checkpoint support in both `gan_trainer()` and `dp_gan_trainer()`:
  - `checkpoint_epochs` parameter to save model states at intervals
  - `checkpoint_path` for disk-based storage of large training runs
* Support for differentially private post-GAN boosting with exponential mechanism
* End-to-end privacy guarantees when combining DP training with DP boosting
* New vignette: "Post-GAN Boosting for Improved Synthetic Data Quality"

## Improvements

### Documentation

* Comprehensive documentation for `data_transformer` class including:
  - Detailed explanation of standard vs mode-specific (GMM) normalization
  - Examples for mixed continuous/categorical data
  - Integration examples with RGAN workflow
* Added CTGAN reference (Xu et al., 2019) for mode-specific normalization

### Training

* New `track_loss` parameter in `gan_trainer()` and `dp_gan_trainer()` to
  record training losses for analysis
* Loss tracking available via `trained_gan$losses$g_loss` and `$d_loss`

## Bug Fixes
* Fixed `sample_synthetic_data()` compatibility with `dp_gan_trainer()` output
* Fixed TabularGenerator method documentation
* Fixed GAN_value_fct missing epsilon parameter documentation

## Dependencies

* OpenDP package now suggested (optional) for cryptographically secure DP training

# RGAN 0.1.1

* Minor CRAN compliance fixes
* Added `track_loss` parameter to `gan_trainer()` and `gan_update_step()`

# RGAN 0.1.0

* Added a `NEWS.md` file to track changes to the package.
* This is the initial release of RGAN
