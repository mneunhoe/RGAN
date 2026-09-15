# RGAN 0.2.1

## Privacy correctness

This release fixes three defects in `post_gan_boosting(dp = TRUE)` that caused the
realised privacy loss to differ from the requested `MW_epsilon`. Version 0.2.0 was
distributed on GitHub only. **A run made with 0.2.0 did not receive the budget it
asked for if any of the following applied:**

* **two or more discriminator checkpoints produced identical quality scores.**
  Selection sampled a score *value* and mapped it back to the first index holding
  that value, so every tied discriminator after the first was given probability
  zero. When a tie is present on one dataset and broken on an adjacent one the
  ratio of output probabilities is unbounded, so no finite epsilon held. Selection
  now samples the discriminator index directly, which is what the exponential
  mechanism is.
* **PacGAN packing was used (`pac > 1`).** With packing each discriminator score
  covers `pac` records, so `d_score_real` averages `n / pac` values and its
  sensitivity is `pac / n`. The mechanism was calibrated with `real_N = n`, which
  understated the loss by a factor of `pac`.
  `compute_discriminator_scores()` now reports `n_real_units`, and
  `apply_post_gan_boosting()` calibrates to it.
* **the GAN used a value function other than `"original"`.** Those discriminators
  have no sigmoid output layer, so their scores are unbounded and the `1 / n`
  sensitivity does not hold. `dp = TRUE` now rejects scores outside `[0, 1]` with
  an error instead of proceeding.

## Other fixes

* The exponential mechanism weights are now formed with the log-sum-exp shift.
  Previously the exponent `epsilon0 * real_N * U / 2` could underflow for every
  discriminator, leaving `p_U` exactly uniform: the algorithm selected at random
  while still spending the full budget, with no warning.
* `apply_post_gan_boosting(dp = TRUE)` now warns when the checkpoints carry no
  privacy record. `dp = TRUE` privatises discriminator *selection* only; if the
  GAN itself was trained with `gan_trainer()` the checkpoints already encode the
  data and the output is not differentially private end to end. Total budget is the
  training epsilon plus `MW_epsilon`.
* `post_gan_boosting()` now validates that `N_generators`, `nrow(d_score_fake)`,
  `length(d_score_real)` and `nrow(B)` agree. A mismatched `N_generators` silently
  recycled the selection indicator into the averaged discriminator when one count
  divided the other.

## New features

* `compute_discriminator_scores()` and `apply_post_gan_boosting()` gain
  `score_transform`, which squashes unbounded critic outputs into `[0, 1]` with a
  sigmoid. The transform is monotone, so it preserves the ranking of samples, and
  it makes WGAN and KLWGAN discriminators usable with `dp = TRUE`.
* `post_gan_boosting()` additionally returns `D_bar`, the mixed discriminator
  strategy: the fraction of rounds in which each discriminator was selected.

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
