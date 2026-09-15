# RGAN 0.2.2

## The mechanism must match the accountant

`dp_discriminator_step()` released a quantity the privacy analysis does not
cover. **Any 0.2.0, 0.2.1 or earlier-0.2.2 run is affected**, independently of
the accountant defect below.

* The noisy gradient sum was divided by the *realised* Poisson batch size. The
  Sampled Gaussian Mechanism bound covers the noisy sum; dividing it by a
  constant is post-processing and is free, but the realised batch size is not a
  constant. It is a data-dependent count -- under add/remove-one adjacency a
  dataset with one more record can yield one more sampled record -- and it set
  both the location and the scale of what was released. Nothing in the
  accounting paid for it. The divisor is now the expected batch size,
  `sampling_rate * n_samples`, which is what Opacus and TensorFlow Privacy use.
* An empty Poisson draw returned early without taking an optimizer step. That
  made "no step" an observable event of probability `(1 - q)^n`, again
  depending on the number of records. The mechanism now releases noise and
  steps, as its formalized counterpart does; see `dp_noise_only_step()`.
* `dp_discriminator_step()` now rejects `sampling_rate * n_samples <= 0` rather
  than dividing by it.

Gradients are now scaled by the expected rather than the realised batch size,
so the effective learning rate changes slightly; it is unbiased for the same
quantity, and learning rates tuned on 0.2.1 may need revisiting.

## Correspondence with a machine-checked proof

The accounting is now tied to a Lean 4 formalization that depends only on
Mathlib, is free of `sorry`, and uses only Lean's three standard axioms.

* At integer orders the moment computed by `log_moment_sgm_int()` is Lean's
  `sgmMoment`, `rdp_sgm()` is `sgmRDP`, and the expression `get_epsilon()`
  minimises is the epsilon of `approxDP_transcript_cks`. The new test file
  `test-lean-correspondence.R` checks each of these against a direct
  transcription of the Lean definition.
* The subsampling reduction of Mironov, Talwar and Zhang -- that the mechanism's
  own outputs on adjacent datasets are no further apart than the two-point
  mixture pair -- is `renyiDiv_sgmMechanism_le`, now proved rather than assumed.
  This is what makes the per-step bound apply to the mechanism rather than only
  to the canonical pair.
* Two gaps between proof and package are documented in `rdp_sgm()`: the proof is
  carried out for a scalar release, and it covers integer orders only.
* The accountant gains a `certified_only` argument, restricting the order grid
  to the integer orders that carry a proof. It reports a larger epsilon -- by up
  to 11 percent when the optimal order is small -- in exchange for a fully
  checked number.

## Privacy correctness

`dp_gan_trainer()` reported an epsilon that was far too small in the regime
DP-SGD normally runs in. **Any 0.2.0 or 0.2.1 run with a sampling rate below
0.01 - that is, `batch_size / nrow(data) < 0.01`, which is the usual case -
consumed substantially more privacy budget than it reported.** Re-run the
accounting for anything you have published.

* The RDP of the Sampled Gaussian Mechanism is now computed with the bound of
  Mironov, Talwar and Zhang (2019), evaluated in log space by the new internal
  `rdp_sgm()`. The previous accountant used a hand-written two-branch formula.
  Its `q < 0.01` branch returned `q^2 * alpha / (2 sigma^2)`, which understates
  the true value by up to four orders of magnitude; its `q >= 0.01` branch
  overstated it by up to a factor of three; and the bound jumped by a factor of
  10,000 across the branch boundary at `q = 0.01`.
* `get_epsilon()` now uses the conversion of Canonne, Kamath and Steinke (2020),
  which is uniformly tighter than the classical
  `eps = rdp + log(1 / delta) / (alpha - 1)`.
* The order grid searched by `get_epsilon()` is now fine between 1 and 11 and
  coarsens above that, and is configurable through the accountant's new `orders`
  field. Non-integer orders are supported through a series expansion
  (`log_moment_sgm_frac()`), since the binomial sum only terminates at integer
  orders. That series stops on a bound for its whole remaining tail rather than
  on the size of the last term: the terms decay only polynomially, as
  `i^-(alpha + 2)`, so a term-size threshold would need about `1e10` terms at
  `alpha = 1.25` and was in practice never met. It is also evaluated in
  vectorised blocks, and now raises an error instead of silently returning an
  under-converged sum. Because a fractional order carries no proof, `rdp_sgm()`
  additionally brackets it at run time between the certified values at
  `floor(alpha)` and `ceiling(alpha)`, which must contain it since the Renyi
  divergence is non-decreasing in its order. They are not a refinement: restricting the search to integers inflates
  the reported epsilon by up to 11 percent, measured across a sweep of realistic
  settings. The penalty is concentrated where the optimal order is small - at
  `alpha* = 2` consecutive integers are 50 percent apart in relative terms -
  and it is negligible (below 0.1 percent) once the optimum exceeds about 12.
  The default grid lands within 0.2 percent of the optimum over the same sweep.
* `calibrate_noise_multiplier()` could return a noise multiplier whose epsilon
  *exceeded* the target by up to `tolerance`, because it returned early on
  `abs(eps - target) < tolerance` regardless of sign. It is now one-sided: the
  multiplier it returns always satisfies `epsilon <= target_epsilon`. It also
  errors, rather than silently returning `max_noise`, when the target cannot be
  reached within the search range.
* `compute_max_steps()` inherits the corrected accountant, so training now stops
  at the right step. Budgets computed with 0.2.0 or 0.2.1 allowed far too many
  steps.

## Testing

* New `test-dp-accountant.R` checks `rdp_sgm()` against direct numerical
  integration of the density ratio - an independent evaluation path, so it
  tests the algebra rather than the code against itself - plus continuity in the
  sampling rate, the exact boundary cases `q = 0` and `q = 1`, monotonicity of
  epsilon in every parameter, and one-sided calibration. The property tests
  alone would not have caught the 0.2.0 defect: that formula satisfied all of
  them. The reference cross-check is what does.
* The fractional-order series is pinned against the exact binomial sum at
  integer orders, where the two must agree to machine precision, and against
  numerical integration at fractional orders.

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
