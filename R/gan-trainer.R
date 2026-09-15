#' @title gan_trainer
#'
#' @description Provides a function to quickly train a GAN model.
#'
#' @param data Input a data set. Needs to be a matrix, array, torch::torch_tensor or torch::dataset.
#' @param noise_dim The dimensions of the GAN noise vector z. Defaults to 2.
#' @param noise_distribution The noise distribution. Expects a function that samples from a distribution and returns a torch_tensor. For convenience "normal" and "uniform" will automatically set a function. Defaults to "normal".
#' @param value_function The value function for GAN training. Expects a function that takes discriminator scores of real and fake data as input and returns a list with the discriminator loss and generator loss. For convenience the loss functions "original", "non-saturating", "hinge", "wasserstein", "wgan-gp", "f-wgan" and "rpgan" are already implemented. Defaults to "original".
#'   Only "original" uses a sigmoid output layer on the default discriminator; every other built-in
#'   value function (including "non-saturating", "hinge" and "rpgan") works on logits.
#' @param gp_lambda The gradient penalty coefficient for WGAN-GP training. Only used when value_function is "wgan-gp" and `regularization` is NULL. Defaults to 10. Superseded by `regularization`.
#' @param regularization Optional discriminator regularization, given as a list with the fields
#'   `type` (one of "none", "wgan-gp", "r1", "r2", "r1r2"), `gamma` (the penalty coefficient,
#'   default 10) and `lazy_interval` (an integer >= 1, default 1: the penalty is only applied on
#'   every `lazy_interval`-th discriminator step and is then scaled by `lazy_interval`, the lazy
#'   regularization of StyleGAN2). "r1" penalizes the squared gradient norm of the discriminator
#'   with respect to real inputs, "r2" the same with respect to fake inputs, "r1r2" both; the
#'   penalty added to the discriminator loss is `gamma / 2` times the penalty (see [r1_penalty()]).
#'   Defaults to NULL, which preserves the previous behaviour: a WGAN-GP penalty with coefficient
#'   `gp_lambda` if `value_function == "wgan-gp"` and `gp_lambda > 0`, and no penalty otherwise.
#'   If `regularization` is supplied it takes precedence over `gp_lambda`, and a warning is issued
#'   if an explicitly supplied `gp_lambda` disagrees with it.
#' @param adam_betas Optional numeric vector of length 2 with the Adam beta coefficients used for
#'   both default optimizers, e.g. `c(0, 0.99)` as in R3GAN. Defaults to NULL, which keeps torch's
#'   own default of `c(0.9, 0.999)`. Only used if no optimizer is explicitly passed to the trainer.
#' @param discriminator_dropout Dropout rate of the default discriminator. Defaults to 0.5. Set to 0
#'   to train without dropout (as R3GAN does). Only used if `discriminator` is NULL.
#' @param instance_noise Optional adaptive instance noise, given as a list. Defaults to NULL
#'   (no instance noise; the training path is then bit-identical to the previous behaviour).
#'   The mechanism is Diffusion-GAN (Wang et al., ICLR 2023, arXiv:2206.02262), simplified to
#'   plain Gaussian noise without the diffusion mixture: the trainer maintains a noise ceiling
#'   `T` in `[sigma_min, sigma_max]`, and on every discriminator step draws one
#'   `sigma ~ U(sigma_min, T)` for the whole batch and adds `N(0, sigma^2)` noise to both the
#'   real and the fake batch *before* PacGAN packing and before the discriminator. The R1/R2/GP
#'   penalties see exactly the same noised (and packed) tensors. The generator step adds noise
#'   with the same `sigma` to its fake batch, and the gradient flows through the noise addition.
#'   Fields (with defaults):
#'   \describe{
#'     \item{`sigma_max`}{Upper bound of the noise ceiling. Default 0.5.}
#'     \item{`sigma_min`}{Lower bound of the noise ceiling. Default 0.}
#'     \item{`adaptive`}{Adapt `T` during training? Default TRUE. `T` starts at `sigma_min` when
#'       adaptive and at `sigma_max` otherwise.}
#'     \item{`target_rd`}{Target for the discriminator-overfitting statistic `r_d`, a number in
#'       `[-1, 1]`. Default 0.6. How `r_d` is measured depends on what the discriminator output
#'       means under the chosen value function (see [resolve_rd_mode()]):
#'       for relativistic (paired) losses such as `"rpgan"`, and for the Wasserstein/critic
#'       family (`"wasserstein"`, `"wgan-gp"`, `"f-wgan"`), the absolute level of `D` is
#'       unconstrained -- only `D(real) - D(fake)` enters the loss -- so
#'       `r_d = mean(sign(D(real_i) - D(fake_i)))` over the paired batch, that is the fraction of
#'       correctly ranked pairs mapped to `[-1, 1]`;
#'       for `"original"`, whose default discriminator has a sigmoid output,
#'       `r_d = mean(sign(D(real) - 0.5))`;
#'       for the zero-anchored logit losses `"non-saturating"` and `"hinge"`,
#'       `r_d = mean(sign(D(real)))`, the original ADA statistic of Karras et al. (2020).
#'       A user-supplied value function uses the paired definition when it carries the attribute
#'       `requires_real_scores = TRUE` and `mean(sign(D(real)))` otherwise. In every case the
#'       scores are those of the noised and augmented batch the discriminator was just trained
#'       on.}
#'     \item{`per_type`}{Use a per-column noise scale derived from `output_info`? Default TRUE:
#'       categorical blocks (`"softmax"`, `"cossin"`) and the mode-indicator sub-block of a
#'       `"mode_specific"` block get a multiplier of 0.5, numeric columns 1. This is a simple
#'       per-type schedule inspired by CDTD (Mueller et al., ICLR 2025), not CDTD's learned
#'       schedule. If `output_info` is NULL every column counts as numeric.}
#'     \item{`condition_d`}{Append the current `sigma` as an extra input column to the
#'       discriminator? Default FALSE. With TRUE the default discriminator's input dimension
#'       becomes `data_dim * pac + 1`; a custom `discriminator` must then be built with that
#'       extra input itself.}
#'     \item{`update_every`}{Number of discriminator steps between adaptations of `T`.
#'       Default 4. Every `update_every` steps `T` moves by `sigma_max / 100` up if
#'       `r_d > target_rd` and down otherwise, clamped to `[sigma_min, sigma_max]`.}
#'   }
#' @param augmentation Optional adaptive differentiable augmentation of the discriminator inputs,
#'   given as a list. Defaults to NULL (no augmentation). Augmentation is applied to both the real
#'   and the fake batch, after the instance noise and before packing, and the gradient flows
#'   through it for the fake batch, as ADA requires (Karras et al., NeurIPS 2020). Fields (with
#'   defaults):
#'   \describe{
#'     \item{`p_max`}{Upper bound of the augmentation probability. Default 0.5.}
#'     \item{`adaptive`}{Adapt `p` during training? Default TRUE. `p` starts at 0 when adaptive
#'       and at `p_max` otherwise.}
#'     \item{`target_rd`}{Target for the same `r_d` statistic used by `instance_noise`, whose
#'       definition depends on the value function; see the `target_rd` field of
#'       `instance_noise` above and [resolve_rd_mode()]. Default 0.6.}
#'     \item{`ops`}{Subset of `c("mask", "mixup", "swap")`, see [augment_batch()].
#'       Default all three.}
#'     \item{`update_every`}{Number of discriminator steps between adaptations of `p`.
#'       Default 4; the step size is `p_max / 100`, clamped to `[0, p_max]`.}
#'   }
#' @param conditioning Optional any-order conditioning of the generator and the
#'   discriminator, given as a list. Defaults to NULL (unconditional training;
#'   the training path is then bit-identical to the previous behaviour).
#'   Requires `output_info` and a matrix / tensor `data`.
#'   This is Experiment 5 of the 2026 experiment plan: CTGAN's conditional
#'   vector with training-by-sampling (Xu et al., NeurIPS 2019) and the
#'   TabMT / TabularARGN random-mask any-order variant (report item T3) with
#'   classifier-free-guidance style condition dropout (report item L7).
#'
#'   The condition of a row is `[mask, masked values]`: a 0/1 mask of length
#'   `n_blocks`, one entry per column block of the transformed matrix (the
#'   blocks of [output_info_blocks()], so a `"mode_specific"` column
#'   contributes a mode-indicator block and a value block, and both are always
#'   observed together), followed by the row's transformed values with the
#'   columns of every unobserved block set to zero (length `data_dim`). The
#'   generator input is `cat(z, condition)` and the discriminator input is
#'   `cat(row, condition)` per sample, appended *before* PacGAN packing. The
#'   fake rows of a step use the same conditions as the real rows, as in CTGAN,
#'   so the discriminator compares `(real, cond)` against `(fake, cond)`.
#'   Fields (with defaults):
#'   \describe{
#'     \item{`type`}{`"single_column"` (CTGAN: the mask observes exactly one
#'       categorical block, and the real batch is sampled *by* that condition)
#'       or `"random_mask"` (a random observed subset per row, whose values are
#'       the real row's own values). Default `"random_mask"`.}
#'     \item{`mask_prob`}{`"uniform"` (default; a per-row Bernoulli mask whose
#'       rate is drawn `U(0, 1)` per row, so every masking rate is trained) or
#'       a fixed numeric rate. Only used by `"random_mask"`.}
#'     \item{`cond_dropout`}{Probability that a row's condition is replaced by
#'       the all-unobserved mask, in both the discriminator and the generator
#'       step. Default 0.1. This is what makes unconditional sampling from a
#'       conditionally trained model well defined.}
#'     \item{`log_frequency`}{Draw categories with probability proportional to
#'       `log(count + 1)` instead of their frequency, CTGAN's device against
#'       rare-level collapse. Default TRUE.}
#'     \item{`resample_rare`}{Only for `"random_mask"`: `TRUE` (or a fraction)
#'       makes a fraction of the rows of every batch be re-drawn by a
#'       log-frequency category of one of their observed categorical blocks, so
#'       that rare levels are conditioned on far more often than their
#'       empirical frequency. The *real row* is re-drawn (not only the
#'       condition values), so the `(real row, condition)` pair the
#'       discriminator sees stays consistent. Default FALSE;
#'       `resample_rare_frac` defaults to 0.5.}
#'     \item{`consistency_weight`}{Weight of the consistency loss added to the
#'       generator loss (see [conditioning_consistency_loss()]): cross-entropy
#'       on observed categorical blocks, mean squared error on observed numeric
#'       and `"cossin"` blocks, averaged over the observed cells. Default 1.}
#'   }
#'   A custom `generator` must accept the concatenated
#'   `noise_dim + n_blocks + data_dim` input, and a custom `discriminator` the
#'   `pac * (data_dim + n_blocks + data_dim)` input (for a
#'   [TokenDiscriminator()] set `extra_inputs = n_blocks + data_dim`).
#' @param distillation Optional Diffusion2GAN-style pair-regression distillation from a
#'   flow-matching teacher, given as a list. Defaults to NULL (no distillation; the training
#'   path is then bit-identical to the previous behaviour). This is Experiment 6 of the 2026
#'   experiment plan (report item L5), the recipe of Kang et al. (ECCV 2024, arXiv:2405.05967).
#'
#'   The teacher's ODE is a deterministic map from noise `z` to a sample `x_hat`. Every
#'   generator step draws a minibatch of those pairs and adds
#'   `lambda * weighted_mean_loss(G(z_i), x_hat_i)` to the generator loss. The adversarial part
#'   is untouched: it still uses a *fresh* noise draw and the real data batch, so the
#'   discriminator never sees the teacher's output as "real". The pairs only pull the
#'   generator's noise-to-data map towards the teacher's, which is what lets a one-step
#'   student inherit the teacher's coverage.
#'
#'   Fields (with defaults):
#'   \describe{
#'     \item{`pairs`}{A list `list(z = matrix, x = matrix)` from [flow_pairs()]. If NULL the
#'       pairs are generated from `teacher`.}
#'     \item{`teacher`}{A trained teacher of class `"trained_RFlow"`. Used to generate `pairs`
#'       when they are not supplied, and to refresh them when `refresh_every` is set. A bare
#'       `"trained_RFlow"` object passed as `distillation` is shorthand for
#'       `list(teacher = <that object>)`.}
#'     \item{`n_pairs`}{Number of pairs to generate from `teacher`. Default `4 * nrow(data)`.}
#'     \item{`n_steps`}{ODE steps per pair batch. Default 50.}
#'     \item{`solver`}{`"heun"` (default) or `"euler"`.}
#'     \item{`refresh_every`}{Regenerate the pairs from `teacher` every this many epochs.
#'       Default NULL (one fixed pair set for the whole run).}
#'     \item{`lambda`}{Weight of the regression term. Default 10.}
#'     \item{`loss`}{`"mse"` (default) or `"huber"`; `huber_delta` (default 1) sets the
#'       transition point.}
#'     \item{`per_type_weight`}{Weight the per-column residual by [flow_loss_weights()], the
#'       CDTD-style calibration that stops a wide one-hot block from dominating a single
#'       numeric column. Default TRUE.}
#'     \item{`adversarial`}{Default TRUE. With FALSE the discriminator step is skipped
#'       entirely (`n_d_steps` stays 0) and the generator is trained on the regression term
#'       alone: the pure-regression ablation, arm F3 of the plan.}
#'     \item{`noise_match`}{Default TRUE: the regression term feeds the teacher's own `z` to
#'       the generator, which requires `noise_dim == ncol(pairs$z)` and errors otherwise.
#'       FALSE draws fresh noise instead, which breaks the pairing and leaves only a
#'       distribution-level pull towards the teacher's samples; it exists as an ablation.}
#'   }
#'   See [distill_to_gan()] for a wrapper that builds the pairs and sets `noise_dim` for you.
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
#' @param track_loss Store the training losses as additional output. Defaults to FALSE.
#' @param plot_loss Monitor the losses during training with plots. Defaults to FALSE.
#' @param device Input on which device (e.g. "cpu", "cuda", or "mps") training should be done. Defaults to "cpu".
#' @param seed Optional seed for reproducibility. Sets both R's random seed and torch's random seed. Defaults to NULL (no seed).
#' @param validation_data Optional validation data for monitoring training. Should be in the same format as training data.
#' @param early_stopping Enable early stopping based on validation metrics. Defaults to FALSE.
#' @param patience Number of epochs without improvement before stopping. Only used if early_stopping is TRUE. Defaults to 10.
#' @param lr_schedule Learning rate schedule type. One of "constant" (default), "step", "exponential", or "cosine".
#'   "step" reduces LR by lr_decay_factor every lr_decay_steps epochs.
#'   "exponential" applies lr_decay_factor decay each epoch.
#'   "cosine" uses cosine annealing from base_lr to 0 over all epochs.
#' @param lr_decay_factor Multiplicative factor for learning rate decay. Used with "step" and "exponential" schedules. Defaults to 0.1.
#' @param lr_decay_steps Number of epochs between learning rate reductions for "step" schedule. Defaults to 50.
#' @param pac Number of samples to pack together for PacGAN (reduces mode collapse). The discriminator
#'   sees `pac` samples concatenated together, helping it detect lack of diversity. Must divide batch_size
#'   evenly. Defaults to 1 (standard GAN, no packing). Common values are 8 or 10.
#' @param output_info Optional output structure from data_transformer$output_info. When provided,
#'   enables Gumbel-Softmax for categorical columns, improving gradient flow for discrete variables.
#'   Each element should be a list with (dimension, type) where type is "linear", "mode_specific", or "softmax".
#' @param gumbel_tau Temperature for Gumbel-Softmax. Lower values (e.g., 0.2) produce more discrete
#'   outputs. Only used when output_info is provided. Defaults to 0.2.
#' @param generator_hidden_units List of hidden layer sizes for TabularGenerator. Defaults to
#'   list(256, 256) as used in CTGAN. Only used when output_info is provided.
#' @param generator_normalization Normalization type for TabularGenerator: "batch" (default, CTGAN-style),
#'   "layer", or "none". Only used when output_info is provided.
#' @param generator_activation Activation function for TabularGenerator: "relu" (default), "leaky_relu",
#'   "gelu", or "silu". Only used when output_info is provided.
#' @param generator_init Weight initialization for TabularGenerator: "xavier_uniform" (default),
#'   "xavier_normal", "kaiming_uniform", or "kaiming_normal". Only used when output_info is provided.
#' @param generator_residual Enable residual connections in TabularGenerator. Defaults to TRUE.
#'   Only used when output_info is provided.
#' @param checkpoint_epochs Interval for saving model checkpoints (in epochs). If NULL (default),
#'   no checkpoints are saved. For example, checkpoint_epochs = 10 saves checkpoints at epochs
#'   10, 20, 30, etc. Checkpoints enable post-GAN boosting for improved sample quality.
#' @param checkpoint_path Optional path for disk-based checkpoint persistence. If NULL (default),
#'   checkpoints are stored in memory only. If provided, checkpoints are saved to disk, enabling
#'   post-GAN boosting for large training runs with many checkpoints.
#'
#' @return gan_trainer trains the neural networks and returns an object of class trained_RGAN that contains the last generator, discriminator and the respective optimizers, as well as the settings. With `conditioning` the returned object additionally carries `conditioning_meta`, the per-block metadata (`blocks`, block-to-column mapping, categorical block ids, category values, counts and log-frequency tables) that [sample_synthetic_data()] and [impute_synthetic()] need to build conditions at sampling time. The settings include the resolved `regularization`, `instance_noise`, `augmentation` and `conditioning` lists, the final noise ceiling `noise_T_final` and augmentation probability `aug_p_final`, and `n_d_steps`, the total number of discriminator updates performed (useful to equalise training budgets across arms), `n_g_steps`, the number of generator updates, and the resolved `distillation` recipe (without the pair matrices themselves). With `track_loss = TRUE` the trajectories of the noise ceiling and the augmentation probability are returned as `losses$noise_T` and `losses$aug_p`, and the pair-regression term as `losses$distill_loss`.
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
           gp_lambda = 10,
           regularization = NULL,
           adam_betas = NULL,
           discriminator_dropout = 0.5,
           instance_noise = NULL,
           augmentation = NULL,
           conditioning = NULL,
           distillation = NULL,
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
           track_loss = FALSE,
           plot_loss = FALSE,
           device = "cpu",
           seed = NULL,
           validation_data = NULL,
           early_stopping = FALSE,
           patience = 10,
           lr_schedule = "constant",
           lr_decay_factor = 0.1,
           lr_decay_steps = 50,
           pac = 1,
           output_info = NULL,
           gumbel_tau = 0.2,
           generator_hidden_units = list(256, 256),
           generator_normalization = "batch",
           generator_activation = "relu",
           generator_init = "xavier_uniform",
           generator_residual = TRUE,
           checkpoint_epochs = NULL,
           checkpoint_path = NULL) {
# Remember whether the caller supplied gp_lambda explicitly --------------------
    gp_lambda_supplied <- !missing(gp_lambda)

# Set random seeds for reproducibility -----------------------------------------
    if (!is.null(seed)) {
      set.seed(seed)
      torch::torch_manual_seed(seed)
    }

# Input validation -------------------------------------------------------------
    if (batch_size <= 0) {
      stop("batch_size must be a positive integer")
    }
    if (epochs <= 0) {
      stop("epochs must be a positive integer")
    }
    if (base_lr <= 0) {
      stop("base_lr must be a positive number")
    }
    if (noise_dim <= 0) {
      stop("noise_dim must be a positive integer")
    }

    # Validate learning rate schedule parameters
    valid_schedules <- c("constant", "step", "exponential", "cosine")
    if (!lr_schedule %in% valid_schedules) {
      stop(sprintf(
        "lr_schedule must be one of: %s",
        paste(valid_schedules, collapse = ", ")
      ))
    }
    if (lr_decay_factor <= 0 || lr_decay_factor > 1) {
      stop("lr_decay_factor must be between 0 (exclusive) and 1 (inclusive)")
    }
    if (lr_decay_steps <= 0) {
      stop("lr_decay_steps must be a positive integer")
    }
    if (pac < 1 || pac != as.integer(pac)) {
      stop("pac must be a positive integer")
    }
    if (batch_size %% pac != 0) {
      stop(sprintf("batch_size (%d) must be divisible by pac (%d)", batch_size, pac))
    }
    if (gumbel_tau <= 0) {
      stop("gumbel_tau must be a positive number")
    }
    if (!is.numeric(discriminator_dropout) ||
        length(discriminator_dropout) != 1 ||
        discriminator_dropout < 0 || discriminator_dropout >= 1) {
      stop("discriminator_dropout must be a single number in [0, 1)")
    }
    if (!is.null(adam_betas)) {
      if (!is.numeric(adam_betas) || length(adam_betas) != 2 ||
          any(adam_betas < 0) || any(adam_betas >= 1)) {
        stop("adam_betas must be a numeric vector of length 2 with values in [0, 1)")
      }
    }

    # Resolve the discriminator regularization ---------------------------------
    regularization <- resolve_regularization(
      regularization = regularization,
      gp_lambda = gp_lambda,
      gp_lambda_supplied = gp_lambda_supplied,
      value_function = value_function
    )

    # Resolve instance noise and augmentation ----------------------------------
    # Both default to NULL; in that case every code path below is exactly the
    # one that existed before, including the random number draws, so training
    # is bit-identical to the previous behaviour.
    instance_noise <- resolve_instance_noise(instance_noise)
    augmentation <- resolve_augmentation(augmentation)
    conditioning <- resolve_conditioning(conditioning)

    # Validate output_info structure if provided
    if (!is.null(output_info)) {
      if (!is.list(output_info)) {
        stop("output_info must be a list")
      }
      for (i in seq_along(output_info)) {
        info <- output_info[[i]]
        if (!is.list(info) || length(info) < 2) {
          stop("Each element of output_info must be a list with at least 2 elements (dimension, type)")
        }
        valid_types <- c("linear", "softmax", "mode_specific", "quantile", "cossin")
        if (!info[[2]] %in% valid_types) {
          stop(sprintf("output_info type must be one of: %s", paste(valid_types, collapse = ", ")))
        }
      }
    }

    # Validate checkpoint parameters
    if (!is.null(checkpoint_epochs)) {
      if (!is.numeric(checkpoint_epochs) || checkpoint_epochs <= 0 ||
          checkpoint_epochs != as.integer(checkpoint_epochs)) {
        stop("checkpoint_epochs must be a positive integer")
      }
    }
    if (!is.null(checkpoint_path) && is.null(checkpoint_epochs)) {
      warning("checkpoint_path provided but checkpoint_epochs is NULL. No checkpoints will be saved.")
    }

    # Validate device availability
    if (device == "cuda" && !torch::cuda_is_available()) {
      warning("CUDA device requested but not available. Falling back to CPU.")
      device <- "cpu"
    }
    if (device == "mps" && !torch::backends_mps_is_available()) {
      warning("MPS device requested but not available. Falling back to CPU.")
      device <- "cpu"
    }

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
      # Check for empty data
      if (nrow(data) == 0) {
        stop("data cannot be empty")
      }
      if (ncol(data) == 0) {
        stop("data must have at least one column")
      }
      # Check for all-NA data
      if (all(is.na(data))) {
        stop("data cannot be all NA values")
      }

      data <- torch::torch_tensor(data)$to(device = "cpu")
      data_dim <- ncol(data)
      steps <- nrow(data) %/% batch_size

      # Warn if batch_size is larger than data
      if (batch_size > nrow(data)) {
        warning(sprintf(
          "batch_size (%d) is larger than number of observations (%d). Using full data as single batch.",
          batch_size, nrow(data)
        ))
        steps <- 1
      }

      # Validate plot_dimensions
      if (plot_progress) {
        if (length(plot_dimensions) != 2) {
          stop("plot_dimensions must be a vector of length 2")
        }
        if (any(plot_dimensions < 1) || any(plot_dimensions > data_dim)) {
          stop(sprintf(
            "plot_dimensions must be between 1 and %d (number of columns in data)",
            data_dim
          ))
        }
      }
    }

    if("image_folder" %in% class(data)) {
      steps <- length(data$imgs[[1]]) %/% batch_size
    }

# Prepare validation data if provided ------------------------------------------
    if (!is.null(validation_data)) {
      if ((any(c("array", "matrix") %in% class(validation_data)))) {
        validation_data <- torch::torch_tensor(validation_data)$to(device = "cpu")
      }
    }

# Initialize early stopping variables ------------------------------------------
    best_val_metric <- Inf
    epochs_without_improvement <- 0
    best_generator_state <- NULL
    best_discriminator_state <- NULL
    validation_metrics <- list()

# Initialize checkpoint storage ------------------------------------------------
    checkpoints <- NULL
    if (!is.null(checkpoint_epochs)) {
      checkpoints <- list(
        epochs = integer(0),
        discriminators = list(),
        generators = list(),
        on_disk = !is.null(checkpoint_path)
      )
      # Create checkpoint directory if needed
      if (!is.null(checkpoint_path) && !dir.exists(checkpoint_path)) {
        dir.create(checkpoint_path, recursive = TRUE)
      }
    }

# Set the plotting interval ----------------------------------------------------
    plot_interval <- ifelse(plot_interval == "epoch", steps, plot_interval)

# Any-order conditioning -------------------------------------------------------
    # The condition of a row is [mask over the output_info blocks, masked
    # values]; see R/conditioning.R. Everything below is skipped when
    # conditioning is NULL, so that path stays bit-identical.
    conditioning_meta <- NULL
    cond_dim <- 0L
    data_matrix <- NULL
    if (!is.null(conditioning)) {
      if (is.null(output_info)) {
        stop("conditioning requires output_info (the transformer's output_info)")
      }
      if (!inherits(data, "torch_tensor")) {
        stop("conditioning requires `data` to be a matrix, array or torch_tensor")
      }
      data_matrix <- torch::as_array(data$cpu())
      conditioning_meta <- build_conditioning_meta(output_info, data_matrix)
      cond_dim <- as.integer(conditioning_meta$cond_dim)
      if (identical(conditioning$type, "single_column") &&
          length(conditioning_meta$cat_blocks) == 0) {
        stop("conditioning$type = \"single_column\" requires at least one categorical (\"softmax\" or \"cossin\") column in output_info")
      }
    }

# Pair-regression distillation -------------------------------------------------
    # Resolved here because it needs data_dim and nrow(data). Everything below
    # is skipped when `distillation` is NULL, including every random draw, so
    # that path stays bit-identical.
    distill <- NULL
    if (!is.null(distillation) && !isFALSE(distillation)) {
      if (!inherits(data, "torch_tensor")) {
        stop("distillation requires `data` to be a matrix, array or torch_tensor")
      }
      distill <- resolve_distillation(
        distillation,
        noise_dim = noise_dim,
        data_dim = data_dim,
        n_rows = nrow(data),
        output_info = output_info,
        device = device
      )
    }

# Set up the neural networks if none are provided ------------------------------
    if (is.null(generator)) {
      if (!is.null(output_info)) {
        # Use TabularGenerator with Gumbel-Softmax when output_info is provided
        g_net <-
          TabularGenerator(noise_dim = noise_dim,
                           output_info = output_info,
                           hidden_units = generator_hidden_units,
                           dropout_rate = 0.5,
                           tau = gumbel_tau,
                           normalization = generator_normalization,
                           activation = generator_activation,
                           init_method = generator_init,
                           residual = generator_residual,
                           cond_dim = cond_dim)$to(device = device)
      } else {
        g_net <-
          Generator(noise_dim = noise_dim,
                    data_dim = data_dim,
                    dropout_rate = 0.5)$to(device = device)
      }
    } else {
      g_net <- generator
    }

    if (is.null(generator_optimizer)) {
      if (is.null(adam_betas)) {
        g_optim <- torch::optim_adam(g_net$parameters, lr = base_lr)
      } else {
        g_optim <- torch::optim_adam(g_net$parameters, lr = base_lr,
                                     betas = adam_betas)
      }
    } else {
      g_optim <- generator_optimizer
    }

    if (is.null(discriminator)) {
      # For PacGAN, discriminator input is data_dim * pac (packed samples).
      # With instance_noise$condition_d the current sigma is appended as one
      # extra input column (Diffusion-GAN's noise-level conditioning).
      # With `conditioning` the condition is appended to every sample before
      # packing, so the packed width is pac * (data_dim + cond_dim).
      d_input_dim <- (data_dim + cond_dim) * pac +
        if (isTRUE(instance_noise$condition_d)) 1L else 0L
      # Only the "original" value function needs a sigmoid output layer; every
      # other built-in value function (wasserstein, wgan-gp, f-wgan,
      # non-saturating, hinge, rpgan) and every user supplied function works on
      # logits.
      use_sigmoid <- is.character(value_function) &&
        length(value_function) == 1 && value_function == "original"
      d_net <-
        Discriminator(data_dim = d_input_dim,
                      dropout_rate = discriminator_dropout,
                      sigmoid = use_sigmoid)$to(device = device)
    } else {
      d_net <- discriminator
    }

    if (is.null(discriminator_optimizer)) {
      if (is.null(adam_betas)) {
        d_optim <- torch::optim_adam(d_net$parameters, lr = base_lr * ttur_factor)
      } else {
        d_optim <- torch::optim_adam(d_net$parameters, lr = base_lr * ttur_factor,
                                     betas = adam_betas)
      }
    } else {
      d_optim <- discriminator_optimizer
    }

    # Store initial learning rates for scheduling
    g_initial_lr <- g_optim$param_groups[[1]]$lr
    d_initial_lr <- d_optim$param_groups[[1]]$lr

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
    value_fct <- NULL
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
      if (value_function == "wgan-gp") {
        value_fct <- WGAN_value_fct

        # No weight clipping needed for WGAN-GP
        weight_clipper <- function(d_net) {

        }

      }
      if (value_function == "f-wgan") {
        value_fct <- KLWGAN_value_fct

        weight_clipper <- function(d_net) {

        }

      }
      if (value_function == "non-saturating") {
        value_fct <- NSGAN_value_fct

        weight_clipper <- function(d_net) {

        }

      }
      if (value_function == "hinge") {
        value_fct <- hinge_value_fct

        weight_clipper <- function(d_net) {

        }

      }
      if (value_function == "rpgan") {
        value_fct <- RpGAN_value_fct

        weight_clipper <- function(d_net) {

        }

      }

      if (is.null(value_fct)) {
        stop(
          sprintf(
            "Unknown value_function '%s'. Use one of: %s, or pass a function.",
            value_function,
            paste(
              c("original", "non-saturating", "hinge", "wasserstein",
                "wgan-gp", "f-wgan", "rpgan"),
              collapse = ", "
            )
          )
        )
      }
    }

    # Does the generator loss depend on the real scores? Only the relativistic
    # pairing loss does; in that case the real scores have to be recomputed
    # under the updated discriminator before the generator step.
    paired_loss <- isTRUE(attr(value_fct, "requires_real_scores"))

# Instance noise / augmentation state ------------------------------------------
    # The adaptive controllers keep their state in one environment that
    # gan_update_step() mutates, so that the trainer can read the current
    # noise ceiling T and augmentation probability p after every step.
    na_state <- NULL
    noise_scale <- NULL
    if (!is.null(instance_noise) || !is.null(augmentation)) {
      # How the discriminator-overfitting statistic r_d is measured depends on
      # what the discriminator output means under the chosen value function.
      # See resolve_rd_mode() for the rule.
      rd_mode <- resolve_rd_mode(value_function, paired_loss)
      na_state <- new.env(parent = emptyenv())
      na_state$T <- if (is.null(instance_noise)) {
        0
      } else if (isTRUE(instance_noise$adaptive)) {
        instance_noise$sigma_min
      } else {
        instance_noise$sigma_max
      }
      na_state$p <- if (is.null(augmentation)) {
        0
      } else if (isTRUE(augmentation$adaptive)) {
        0
      } else {
        augmentation$p_max
      }
      na_state$rd_mode <- rd_mode
      if (!is.null(instance_noise)) {
        noise_scale <- torch::torch_tensor(
          matrix(instance_noise_scale(output_info, data_dim,
                                      per_type = instance_noise$per_type),
                 nrow = 1)
        )$to(device = device)
      }
    }
    noise_T_trace <- NULL
    aug_p_trace <- NULL





# Sample a fixed noise vector to observe training progress ---------------------
    fixed_z <-
      sample_noise(c(synthetic_examples, noise_dim))$to(device = device)
    # With conditioning the progress plots use the all-unobserved condition,
    # the unconditional mode that cond_dropout trains.
    if (cond_dim > 0) {
      fixed_z <- torch::torch_cat(
        list(fixed_z,
             torch::torch_zeros(c(synthetic_examples, cond_dim),
                                device = fixed_z$device)$to(dtype = fixed_z$dtype)),
        dim = 2
      )
    }

# Initialize progress bar ------------------------------------------------------
    cli::cli_progress_bar("Training the GAN", total = epochs * steps)


    losses <- NULL

# Count the discriminator updates actually performed ---------------------------
    n_d_steps <- 0L
    n_g_steps <- 0L
    distill_trace <- NULL


# Start GAN training loop ------------------------------------------------------
    for (i in 1:(epochs * steps)) {

      # Refresh the distillation pairs from the teacher ----------------------
      if (!is.null(distill) && !is.null(distill$refresh_every) &&
          i > 1L && (i %% steps) == 1L &&
          (((i %/% steps)) %% distill$refresh_every) == 0L) {
        new_pairs <- flow_pairs(distill$teacher, n_pairs = distill$n_pairs,
                                n_steps = distill$n_steps,
                                solver = distill$solver)
        distill$pairs <- validate_flow_pairs(new_pairs, data_dim)
        distill$z_t <- torch::torch_tensor(
          distill$pairs$z, dtype = torch::torch_float())$to(device = device)
        distill$x_t <- torch::torch_tensor(
          distill$pairs$x, dtype = torch::torch_float())$to(device = device)
      }

      step_loss <- gan_update_step(
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
        weight_clipper,
        gp_lambda = if (identical(regularization$type, "wgan-gp")) regularization$gamma else 0,
        track_loss,
        pac,
        regularization = regularization,
        step = i,
        paired_loss = paired_loss,
        instance_noise = instance_noise,
        augmentation = augmentation,
        na_state = na_state,
        noise_scale = noise_scale,
        output_info = output_info,
        conditioning = conditioning,
        conditioning_meta = conditioning_meta,
        data_matrix = data_matrix,
        distillation = distill
      )

      # The pure-regression ablation performs no discriminator update at all.
      if (is.null(distill) || isTRUE(distill$adversarial)) {
        n_d_steps <- n_d_steps + 1L
      }
      n_g_steps <- n_g_steps + 1L

      if(track_loss & length(losses) == 0){
        losses <- step_loss
      }
      if(track_loss & length(losses) > 0){
        losses$g_loss <- c(losses$g_loss, step_loss$g_loss)
        losses$d_loss <- c(losses$d_loss, step_loss$d_loss)
      }
      if (track_loss && !is.null(step_loss$distill_loss)) {
        distill_trace <- c(distill_trace, step_loss$distill_loss)
      }

      # Record the controller trajectories on every controller update ---------
      if (!is.null(instance_noise) && (i %% instance_noise$update_every) == 0) {
        noise_T_trace <- c(noise_T_trace, na_state$T)
      }
      if (!is.null(augmentation) && (i %% augmentation$update_every) == 0) {
        aug_p_trace <- c(aug_p_trace, na_state$p)
      }

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

# Validation and early stopping at the end of each epoch -----------------------
        if (i %% steps == 0) {
          current_epoch <- i %/% steps

          if (!is.null(validation_data) || early_stopping) {
            # Compute validation metrics
            g_net$eval()
            d_net$eval()

            # Generate synthetic samples for validation
            # Ensure sample count is divisible by pac for PacGAN
            val_sample_count <- min(500, nrow(data))
            val_sample_count <- val_sample_count - (val_sample_count %% pac)
            if (val_sample_count < pac) val_sample_count <- pac

            val_noise <- sample_noise(c(val_sample_count, noise_dim))$to(device = device)
            if (cond_dim > 0) {
              val_noise <- torch::torch_cat(
                list(val_noise,
                     torch::torch_zeros(c(val_sample_count, cond_dim),
                                        device = val_noise$device)$to(dtype = val_noise$dtype)),
                dim = 2
              )
            }
            val_synth <- torch::with_no_grad(g_net(val_noise))

            # Compute discriminator accuracy on validation data
            if (!is.null(validation_data)) {
              # Ensure batch size is divisible by pac for PacGAN
              val_size <- min(batch_size, nrow(validation_data))
              val_size <- val_size - (val_size %% pac)  # Make divisible by pac
              if (val_size < pac) val_size <- pac

              val_batch <- validation_data[sample(nrow(validation_data),
                                                   size = val_size)]$to(device = device)

              # Pack samples for PacGAN if needed
              if (pac > 1) {
                val_synth_size <- min(val_size, nrow(val_synth))
                val_synth_size <- val_synth_size - (val_synth_size %% pac)
                if (val_synth_size < pac) val_synth_size <- pac
                val_batch_packed <- val_batch
                val_synth_packed <- val_synth[1:val_synth_size]
              } else {
                val_batch_packed <- val_batch
                val_synth_packed <- val_synth[1:min(batch_size, nrow(val_synth))]
              }
              # The all-unobserved condition is appended per sample, before
              # packing, exactly as in the training step.
              if (cond_dim > 0) {
                val_batch_packed <- attach_condition(
                  val_batch_packed,
                  torch::torch_zeros(c(val_batch_packed$shape[1], cond_dim),
                                     device = val_batch_packed$device)$to(dtype = val_batch_packed$dtype))
                val_synth_packed <- attach_condition(
                  val_synth_packed,
                  torch::torch_zeros(c(val_synth_packed$shape[1], cond_dim),
                                     device = val_synth_packed$device)$to(dtype = val_synth_packed$dtype))
              }
              if (pac > 1) {
                val_batch_packed <- pack_samples(val_batch_packed, pac)
                val_synth_packed <- pack_samples(val_synth_packed, pac)
              }

              # With noise-level conditioning the discriminator expects the
              # extra sigma column; validation data is clean, so sigma = 0.
              if (isTRUE(instance_noise$condition_d)) {
                val_batch_packed <- append_sigma_column(val_batch_packed, 0)
                val_synth_packed <- append_sigma_column(val_synth_packed, 0)
              }

              val_real_scores <- torch::with_no_grad(d_net(val_batch_packed))
              val_fake_scores <- torch::with_no_grad(d_net(val_synth_packed))

              # Discriminator accuracy: real should be high, fake should be low
              if (value_function == "original") {
                d_acc_real <- (val_real_scores > 0.5)$float()$mean()$item()
                d_acc_fake <- (val_fake_scores < 0.5)$float()$mean()$item()
              } else {
                # For WGAN variants, use sign of scores
                d_acc_real <- (val_real_scores > 0)$float()$mean()$item()
                d_acc_fake <- (val_fake_scores < 0)$float()$mean()$item()
              }
              d_accuracy <- (d_acc_real + d_acc_fake) / 2

              # Compute generator diversity (average pairwise distance)
              val_synth_array <- torch::as_array(val_synth$cpu())
              if (nrow(val_synth_array) > 1) {
                sample_idx <- sample(nrow(val_synth_array), min(100, nrow(val_synth_array)))
                diversity <- mean(stats::dist(val_synth_array[sample_idx, , drop = FALSE]))
              } else {
                diversity <- 0
              }

              # Store validation metrics
              validation_metrics[[length(validation_metrics) + 1]] <- list(
                epoch = current_epoch,
                d_accuracy = d_accuracy,
                diversity = diversity
              )

              # Validation metric: we want balanced discriminator (accuracy ~0.5)
              # and high diversity. Lower is better.
              val_metric <- abs(d_accuracy - 0.5) - 0.01 * diversity
            } else {
              # Without validation data, use training loss variance as proxy
              if (track_loss && length(losses$d_loss) >= steps) {
                recent_d_loss <- tail(losses$d_loss, steps)
                val_metric <- stats::sd(recent_d_loss)
              } else {
                val_metric <- Inf
              }
            }

            # Early stopping check
            if (early_stopping) {
              if (val_metric < best_val_metric) {
                best_val_metric <- val_metric
                epochs_without_improvement <- 0
                # Save best model state
                best_generator_state <- g_net$state_dict()
                best_discriminator_state <- d_net$state_dict()
              } else {
                epochs_without_improvement <- epochs_without_improvement + 1
              }

              if (epochs_without_improvement >= patience) {
                cli::cli_alert_info(sprintf(
                  "Early stopping at epoch %d (no improvement for %d epochs)",
                  current_epoch, patience
                ))
                # Restore best model
                if (!is.null(best_generator_state)) {
                  g_net$load_state_dict(best_generator_state)
                  d_net$load_state_dict(best_discriminator_state)
                }
                break
              }
            }

            g_net$train()
            d_net$train()
          }

          # Adjust learning rates at epoch boundary
          if (lr_schedule != "constant") {
            adjust_learning_rate(
              g_optim, g_initial_lr, current_epoch, epochs,
              lr_schedule, lr_decay_factor, lr_decay_steps
            )
            adjust_learning_rate(
              d_optim, d_initial_lr, current_epoch, epochs,
              lr_schedule, lr_decay_factor, lr_decay_steps
            )
          }

          # Save checkpoints at specified intervals
          if (!is.null(checkpoint_epochs) && current_epoch %% checkpoint_epochs == 0) {
            if (!is.null(checkpoint_path)) {
              # Disk-based storage
              d_path <- file.path(checkpoint_path, sprintf("discriminator_epoch_%04d.pt", current_epoch))
              g_path <- file.path(checkpoint_path, sprintf("generator_epoch_%04d.pt", current_epoch))
              torch::torch_save(d_net$state_dict(), d_path)
              torch::torch_save(g_net$state_dict(), g_path)
              checkpoints$epochs <- c(checkpoints$epochs, current_epoch)
              checkpoints$discriminators[[length(checkpoints$discriminators) + 1]] <- d_path
              checkpoints$generators[[length(checkpoints$generators) + 1]] <- g_path
            } else {
              # In-memory storage (clone to avoid reference issues)
              checkpoints$epochs <- c(checkpoints$epochs, current_epoch)
              d_state <- lapply(d_net$state_dict(), function(x) x$clone())
              g_state <- lapply(g_net$state_dict(), function(x) x$clone())
              checkpoints$discriminators[[length(checkpoints$discriminators) + 1]] <- d_state
              checkpoints$generators[[length(checkpoints$generators) + 1]] <- g_state
            }
          }
        }

    }

    # Attach the instance-noise / augmentation trajectories --------------------
    if (track_loss) {
      if (!is.null(noise_T_trace)) {
        if (is.null(losses)) losses <- list()
        losses$noise_T <- noise_T_trace
      }
      if (!is.null(aug_p_trace)) {
        if (is.null(losses)) losses <- list()
        losses$aug_p <- aug_p_trace
      }
      if (!is.null(distill_trace)) {
        if (is.null(losses)) losses <- list()
        losses$distill_loss <- distill_trace
      }
    }

    output <-  list(
      generator = g_net,
      discriminator = d_net,
      generator_optimizer = g_optim,
      discriminator_optimizer = d_optim,
      losses = losses,
      validation_metrics = if (length(validation_metrics) > 0) validation_metrics else NULL,
      checkpoints = checkpoints,
      conditioning_meta = conditioning_meta,
      settings = list(noise_dim = noise_dim,
                      noise_distribution = noise_distribution,
                      sample_noise = sample_noise,
                      value_function = value_function,
                      gp_lambda = gp_lambda,
                      regularization = regularization,
                      adam_betas = adam_betas,
                      discriminator_dropout = discriminator_dropout,
                      instance_noise = instance_noise,
                      augmentation = augmentation,
                      conditioning = conditioning,
                      cond_dim = cond_dim,
                      # The recipe, without the (potentially very large) pair
                      # matrices, the teacher module and the device tensors.
                      distillation = if (is.null(distill)) NULL else
                        distill[c("lambda", "loss", "huber_delta", "adversarial",
                                  "noise_match", "per_type_weight", "n_pairs",
                                  "n_steps", "solver", "refresh_every", "seed")],
                      noise_T_final = if (is.null(instance_noise)) NULL else na_state$T,
                      aug_p_final = if (is.null(augmentation)) NULL else na_state$p,
                      n_d_steps = n_d_steps,
                      n_g_steps = n_g_steps,
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
                      device = device,
                      early_stopping = early_stopping,
                      patience = patience,
                      lr_schedule = lr_schedule,
                      lr_decay_factor = lr_decay_factor,
                      lr_decay_steps = lr_decay_steps,
                      pac = pac,
                      output_info = output_info,
                      gumbel_tau = gumbel_tau,
                      generator_hidden_units = generator_hidden_units,
                      generator_normalization = generator_normalization,
                      generator_activation = generator_activation,
                      generator_init = generator_init,
                      generator_residual = generator_residual,
                      checkpoint_epochs = checkpoint_epochs,
                      checkpoint_path = checkpoint_path)
    )
    class(output) <- "trained_RGAN"
    return(
     output
    )

  }


#' @title gan_update_step
#'
#' @description Provides a function to perform a single GAN training update step,
#'   including discriminator and generator updates.
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
#' @param value_function The value function for GAN training. Expects a function that takes discriminator scores of real and fake data as input and returns a list with the discriminator loss and generator loss. For convenience four loss functions "original", "wasserstein", "wgan-gp", and "f-wgan" are already implemented. Defaults to "original".
#' @param weight_clipper The wasserstein GAN puts some constraints on the weights of the discriminator, therefore weights are clipped during training.
#' @param gp_lambda The gradient penalty coefficient for WGAN-GP. Set to 0 to disable. Defaults to 0.
#' @param track_loss Store the training losses as additional output. Defaults to FALSE.
#' @param pac Number of samples to pack together for PacGAN. Defaults to 1 (no packing).
#' @param regularization Optional resolved regularization list with the fields `type`
#'   (one of "none", "wgan-gp", "r1", "r2", "r1r2"), `gamma` and `lazy_interval`, as produced by
#'   `gan_trainer()`. Defaults to NULL (no zero-centred penalty; the `gp_lambda` path is used).
#' @param step The index of the current discriminator step (1-based). Used to decide whether a
#'   lazily applied penalty is due on this step. Defaults to 1.
#' @param paired_loss Set to TRUE if the generator loss depends on the real scores (as it does for
#'   the relativistic pairing loss). In that case the real scores are recomputed, without gradient,
#'   under the updated discriminator before the generator step. Defaults to FALSE, which reuses
#'   the detached real scores from the discriminator step.
#' @param instance_noise Optional resolved instance-noise list, as produced by
#'   `resolve_instance_noise()` inside [gan_trainer()]. Defaults to NULL (no instance noise).
#' @param augmentation Optional resolved augmentation list, as produced by
#'   `resolve_augmentation()` inside [gan_trainer()]. Defaults to NULL (no augmentation).
#' @param na_state An environment holding the mutable state of the two adaptive controllers
#'   (`T`, `p` and `sigmoid_output`). Required when `instance_noise` or `augmentation` is given;
#'   the function updates it in place. Defaults to NULL.
#' @param noise_scale Optional torch tensor of shape (1, data_dim) with the per-column instance
#'   noise multipliers, as produced by `instance_noise_scale()`. Defaults to NULL (multiplier 1).
#' @param output_info Optional `output_info` list, used to determine the column blocks for the
#'   augmentation operations. Defaults to NULL.
#' @param conditioning Optional resolved conditioning list, as produced by
#'   `resolve_conditioning()` inside [gan_trainer()]. Defaults to NULL (unconditional training).
#' @param conditioning_meta The conditioning metadata from `build_conditioning_meta()`.
#'   Required when `conditioning` is given.
#' @param data_matrix The training data as a plain numeric matrix. Required when `conditioning`
#'   is given; it is the source of the condition values.
#' @param distillation Optional resolved distillation list, as produced by
#'   `resolve_distillation()` inside [gan_trainer()]. When given, a minibatch of teacher pairs
#'   `(z, x_hat)` is drawn every step and `lambda * loss(G(z), x_hat)` is added to the
#'   generator loss; with `adversarial = FALSE` the whole discriminator step is skipped.
#'   Defaults to NULL (no distillation).
#' @return A list with generator and discriminator losses if track_loss is TRUE, otherwise NULL.
#'   With `distillation` the list also carries `distill_loss`, the unweighted (pre-`lambda`)
#'   regression term of this step.
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
           weight_clipper,
           gp_lambda = 0,
           track_loss = FALSE,
           pac = 1,
           regularization = NULL,
           step = 1,
           paired_loss = FALSE,
           instance_noise = NULL,
           augmentation = NULL,
           na_state = NULL,
           noise_scale = NULL,
           output_info = NULL,
           conditioning = NULL,
           conditioning_meta = NULL,
           data_matrix = NULL,
           distillation = NULL) {
    # The pure-regression ablation (distillation$adversarial = FALSE) skips the
    # whole discriminator step, so the generator sees only the teacher pairs.
    do_adv <- is.null(distillation) || isTRUE(distillation$adversarial)
    # Get a fresh batch of data ------------------------------------------------
    # With conditioning the real batch and its condition are drawn together:
    # for "single_column" the batch is sampled *by* the condition (CTGAN's
    # training-by-sampling), for "random_mask" the condition is read off the
    # sampled rows. The condition is built from the clean data, before any
    # instance noise or augmentation.
    cond <- NULL
    cond_mask <- NULL
    cond_values <- NULL
    d_loss <- NULL
    sigma <- NULL
    # >>> adversarial path: everything down to the matching `} else {` below is
    # the ordinary GAN step and runs unless distillation$adversarial is FALSE.
    # The body is deliberately left at its original indentation so that the
    # diff against the pre-distillation trainer stays readable.
    if (do_adv) {
    if (!is.null(conditioning)) {
      if (is.null(conditioning_meta) || is.null(data_matrix)) {
        stop("conditioning requires conditioning_meta and data_matrix")
      }
      cstep <- build_step_condition(data, data_matrix, batch_size,
                                    conditioning, conditioning_meta, device)
      real_data <- cstep$real_data
      cond <- cstep$cond
      cond_mask <- cstep$mask
      cond_values <- cstep$values
    } else {
      real_data <- get_batch(data, batch_size, device)
    }

    # Get a fresh noise sample -------------------------------------------------
    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)
    # Produce fake data from noise ---------------------------------------------
    fake_data <- torch::with_no_grad(g_net(input = attach_condition(z, cond)))

    # Instance noise (Diffusion-GAN, simplified) -------------------------------
    # One sigma per batch, drawn from U(sigma_min, T). The same sigma is reused
    # for the generator step below, and the noise is added *before* packing, so
    # that the discriminator, the penalties and the generator all see exactly
    # the same noised tensors.
    if (!is.null(instance_noise)) {
      sigma <- instance_noise$sigma_min +
        (na_state$T - instance_noise$sigma_min) * stats::runif(1)
      if (sigma > 0) {
        real_data <- add_instance_noise(real_data, sigma, noise_scale)
        fake_data <- add_instance_noise(fake_data, sigma, noise_scale)
      }
    }

    # Adaptive augmentation (ADA-style, differentiable) ------------------------
    if (!is.null(augmentation) && na_state$p > 0) {
      real_data <- augment_batch(real_data, na_state$p, augmentation$ops,
                                 output_info)
      fake_data <- augment_batch(fake_data, na_state$p, augmentation$ops,
                                 output_info)
    }

    # Append the condition per sample, then pack for PacGAN --------------------
    # The condition goes in before packing, so a packed sample slot is
    # [row, condition]; that is the layout TokenDiscriminator's `extra_inputs`
    # expects, and for the default MLP discriminator it only widens the input.
    real_data_packed <- attach_condition(real_data, cond)
    fake_data_packed <- attach_condition(fake_data, cond)
    if (pac > 1) {
      real_data_packed <- pack_samples(real_data_packed, pac)
      fake_data_packed <- pack_samples(fake_data_packed, pac)
    }

    # Noise-level conditioning of the discriminator ----------------------------
    if (!is.null(instance_noise) && isTRUE(instance_noise$condition_d)) {
      real_data_packed <- append_sigma_column(real_data_packed, sigma)
      fake_data_packed <- append_sigma_column(fake_data_packed, sigma)
    }

    # Compute the discriminator scores on real and fake data -------------------
    dis_real <- d_net(real_data_packed)
    dis_fake <- d_net(fake_data_packed)
    # Keep detached copies of the scores the discriminator step saw; the
    # adaptive controllers below need them after the backward pass has
    # consumed the graph.
    if (!is.null(na_state)) {
      dis_real_d <- dis_real$detach()
      dis_fake_d <- dis_fake$detach()
    }
    # Calculate the discriminator loss
    d_loss <- value_function(dis_real, dis_fake)[["d_loss"]]

    # Discriminator regularization ---------------------------------------------
    # Penalties are computed on the packed tensors, i.e. exactly the inputs the
    # discriminator is called with.
    reg_type <- if (is.null(regularization)) "none" else regularization$type
    lazy_interval <- if (is.null(regularization)) 1L else regularization$lazy_interval
    gamma <- if (is.null(regularization)) 0 else regularization$gamma
    # Lazy regularization (StyleGAN2): apply the penalty only every
    # lazy_interval-th step, scaled by lazy_interval. With the default
    # lazy_interval = 1 the penalty is applied on every step.
    penalty_due <- (step %% lazy_interval) == 0

    if (gp_lambda > 0 && penalty_due) {
      gp <- gradient_penalty(d_net, real_data_packed, fake_data_packed$detach(), device)
      d_loss <- d_loss + lazy_interval * gp_lambda * gp
    }

    if (reg_type %in% c("r1", "r1r2") && penalty_due) {
      d_loss <- d_loss +
        lazy_interval * (gamma / 2) * r1_penalty(d_net, real_data_packed, device)
    }
    if (reg_type %in% c("r2", "r1r2") && penalty_due) {
      d_loss <- d_loss +
        lazy_interval * (gamma / 2) * r2_penalty(d_net, fake_data_packed$detach(), device)
    }

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
    # Note that this is a *fresh* draw even under distillation: the adversarial
    # term never reuses the teacher's pair noise, and the discriminator never
    # sees the teacher's targets as real data.
    z <-
      sample_noise(c(batch_size, noise_dim))$to(device = device)
    # Produce fake data --------------------------------------------------------
    # The fake rows reuse the conditions of the real rows of this step, which
    # is CTGAN's design: the discriminator then compares (real, cond) against
    # (fake, cond) rather than learning to separate the conditions themselves.
    fake_data <- g_net(attach_condition(z, cond))
    # Keep the raw generator output for the consistency loss, before the
    # instance noise / augmentation that only the discriminator path needs.
    fake_data_clean <- fake_data

    # The same instance noise and augmentation the discriminator was trained on
    # is applied to the generator's fake batch. Both are differentiable, so the
    # generator gradient flows through them.
    if (!is.null(instance_noise) && !is.null(sigma) && sigma > 0) {
      fake_data <- add_instance_noise(fake_data, sigma, noise_scale)
    }
    if (!is.null(augmentation) && na_state$p > 0) {
      fake_data <- augment_batch(fake_data, na_state$p, augmentation$ops,
                                 output_info)
    }

    # Append the condition per sample, then pack for PacGAN --------------------
    fake_data_packed <- attach_condition(fake_data, cond)
    if (pac > 1) {
      fake_data_packed <- pack_samples(fake_data_packed, pac)
    }

    if (!is.null(instance_noise) && isTRUE(instance_noise$condition_d)) {
      fake_data_packed <- append_sigma_column(fake_data_packed, sigma)
    }

    # Calculate discriminator score for fake data ------------------------------
    dis_fake <- d_net(fake_data_packed)
    # Real scores for the generator loss ---------------------------------------
    # The graph of dis_real was consumed by the discriminator backward pass, so
    # the real scores are always passed detached. For a paired (relativistic)
    # loss the generator loss genuinely depends on them, so they are recomputed
    # under the updated discriminator.
    if (paired_loss) {
      dis_real_g <- torch::with_no_grad(d_net(real_data_packed))
    } else {
      dis_real_g <- dis_real$detach()
    }
    # Get generator loss based on scores ---------------------------------------
    g_loss <- value_function(dis_real_g, dis_fake)[["g_loss"]]
    # Consistency loss ---------------------------------------------------------
    # Penalise the generator for changing the cells its condition observes.
    if (!is.null(conditioning) && conditioning$consistency_weight > 0) {
      g_loss <- g_loss + conditioning$consistency_weight *
        conditioning_consistency_loss(fake_data_clean, cond_mask, cond_values,
                                      conditioning_meta)
    }
    # <<< end of the adversarial path
    } else {
      # Pure-regression ablation: no discriminator, no real batch, no
      # adversarial term. The generator loss starts at zero and the pair
      # regression below is all there is. `cond` stays NULL, so the generator
      # is fed the all-unobserved condition when conditioning is on.
      g_loss <- torch::torch_zeros(c(1), device = device)$squeeze()
    }

    # Pair-regression distillation term ----------------------------------------
    # Diffusion2GAN (Kang et al., ECCV 2024): draw a minibatch of teacher pairs
    # (z_i, x_i) and pull G(z_i) towards x_i. With `noise_match` the generator
    # is fed the teacher's own z, so the student learns the teacher's
    # noise-to-data map and not merely its marginal.
    distill_value <- NULL
    if (!is.null(distillation)) {
      n_pairs <- distillation$z_t$size(1)
      take <- min(batch_size, n_pairs)
      idx <- torch::torch_tensor(
        sample.int(n_pairs, take, replace = take > n_pairs),
        dtype = torch::torch_long(), device = distillation$z_t$device)
      x_target <- distillation$x_t[idx, ]
      z_pair <- if (isTRUE(distillation$noise_match)) {
        distillation$z_t[idx, ]
      } else {
        # Ablation: the pairing is broken, only the targets are used.
        sample_noise(c(take, noise_dim))$to(device = device)
      }
      # The all-unobserved condition, the unconditional mode cond_dropout
      # trains, so that the regression target (an unconditional teacher
      # sample) is compared against an unconditional generator draw.
      cond_pair <- NULL
      if (!is.null(conditioning) && !is.null(conditioning_meta)) {
        cond_pair <- torch::torch_zeros(
          c(take, as.integer(conditioning_meta$cond_dim)),
          device = device)$to(dtype = z_pair$dtype)
      }
      g_fake_pair <- g_net(attach_condition(z_pair, cond_pair))
      distill_loss <- distill_regression_loss(
        g_fake_pair, x_target, weights = distillation$weight_t,
        loss = distillation$loss, delta = distillation$huber_delta)
      distill_value <- torch::as_array(distill_loss$detach()$cpu())
      g_loss <- g_loss + distillation$lambda * distill_loss
    }

    # What follows is one update step for the generator net --------------------
    # First set all previous gradients to zero
    g_optim$zero_grad()

    # Pass the loss backward through the net
    g_loss$backward()

    # Take one step of the optimizer
    g_optim$step()

    # Adapt the instance-noise ceiling and the augmentation probability -------
    # r_d = mean(sign(D(real))) on the (noised, augmented) real batch the
    # discriminator has just seen; this is the ADA heuristic of Karras et al.
    # (2020), also used by Diffusion-GAN. For the sigmoid output of the
    # "original" value function the sign is taken around 0.5.
    if (do_adv && !is.null(na_state)) {
      needs_update <-
        (!is.null(instance_noise) && isTRUE(instance_noise$adaptive) &&
           (step %% instance_noise$update_every) == 0) ||
        (!is.null(augmentation) && isTRUE(augmentation$adaptive) &&
           (step %% augmentation$update_every) == 0)
      if (needs_update) {
        rd_mode <- if (is.null(na_state$rd_mode)) "sign_real" else na_state$rd_mode
        r_d <- switch(
          rd_mode,
          # Relativistic and critic losses: only the paired difference is
          # meaningful, so r_d is the fraction of correctly ranked pairs
          # mapped to [-1, 1].
          "paired" = torch::as_array(
            torch::torch_sign(dis_real_d - dis_fake_d)$mean()$cpu()),
          # Sigmoid output of the "original" value function: threshold 0.5.
          "sigmoid" = torch::as_array(
            torch::torch_sign(dis_real_d - 0.5)$mean()$cpu()),
          # Zero-anchored logit output: the original ADA statistic.
          torch::as_array(torch::torch_sign(dis_real_d)$mean()$cpu())
        )
        if (!is.null(instance_noise) && isTRUE(instance_noise$adaptive) &&
            (step %% instance_noise$update_every) == 0) {
          delta <- instance_noise$sigma_max / 100
          na_state$T <- na_state$T + if (r_d > instance_noise$target_rd) delta else -delta
          na_state$T <- min(max(na_state$T, instance_noise$sigma_min),
                            instance_noise$sigma_max)
        }
        if (!is.null(augmentation) && isTRUE(augmentation$adaptive) &&
            (step %% augmentation$update_every) == 0) {
          delta <- augmentation$p_max / 100
          na_state$p <- na_state$p + if (r_d > augmentation$target_rd) delta else -delta
          na_state$p <- min(max(na_state$p, 0), augmentation$p_max)
        }
      }
    }

    if(track_loss){

      out <- list(
        g_loss = torch::as_array(g_loss$detach()$cpu()),
        # With adversarial = FALSE there is no discriminator step; report 0 so
        # that downstream collapse checks see a finite trajectory.
        d_loss = if (is.null(d_loss)) 0 else torch::as_array(d_loss$detach()$cpu())
      )
      if (!is.null(distill_value)) out$distill_loss <- distill_value
      return(out)


    }
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

#' @title Choose how the discriminator-overfitting statistic is measured
#'
#' @description The adaptive controllers of `instance_noise` and `augmentation`
#'   steer on `r_d`, a statistic in `[-1, 1]` that measures how confidently the
#'   discriminator separates real from fake. What the discriminator's output
#'   *means* differs between value functions, so `r_d` cannot be defined the
#'   same way for all of them:
#'
#'   \describe{
#'     \item{`"paired"`}{`r_d = mean(sign(D(real_i) - D(fake_i)))`, the fraction
#'       of correctly ranked pairs mapped to `[-1, 1]`. Used for relativistic
#'       (paired) losses, where only the difference `D(real) - D(fake)` enters
#'       the loss and the absolute level of `D` is unconstrained, and for the
#'       Wasserstein/critic family (`"wasserstein"`, `"wgan-gp"`, `"f-wgan"`),
#'       whose critic output is likewise not centred on zero. For those losses
#'       `mean(sign(D(real)))` carries no information about overfitting: the
#'       discriminator can shift its whole output up or down for free.}
#'     \item{`"sigmoid"`}{`r_d = mean(sign(D(real) - 0.5))`, for the
#'       `"original"` value function, whose default discriminator has a sigmoid
#'       output with decision threshold 0.5.}
#'     \item{`"sign_real"`}{`r_d = mean(sign(D(real)))`, the original ADA
#'       statistic of Karras et al. (2020), for logit losses whose output is
#'       anchored at zero: `"non-saturating"` and `"hinge"`.}
#'   }
#'
#'   A user-supplied value function is treated as `"paired"` when it carries
#'   the attribute `requires_real_scores = TRUE` and as `"sign_real"`
#'   otherwise.
#'
#' @param value_function The `value_function` argument of [gan_trainer()].
#' @param paired_loss Does the generator loss depend on the real scores?
#'
#' @return One of `"paired"`, `"sigmoid"` or `"sign_real"`.
#' @keywords internal
resolve_rd_mode <- function(value_function, paired_loss = FALSE) {
  if (isTRUE(paired_loss)) return("paired")
  if (is.character(value_function) && length(value_function) == 1) {
    if (value_function %in% c("wasserstein", "wgan-gp", "f-wgan")) {
      return("paired")
    }
    if (value_function == "original") return("sigmoid")
  }
  "sign_real"
}


#' @title Add Gaussian instance noise to a batch
#'
#' @description Adds `N(0, (scale_j * sigma)^2)` noise to column `j` of a batch.
#'   The operation is differentiable with respect to `x`, so it can be applied
#'   to generator output inside the generator step.
#'
#' @param x A torch tensor of shape (batch, data_dim).
#' @param sigma The noise level of this step.
#' @param noise_scale Optional torch tensor of shape (1, data_dim) with the
#'   per-column multipliers. NULL means a multiplier of 1 everywhere.
#'
#' @return A torch tensor of the same shape as `x`.
#' @keywords internal
add_instance_noise <- function(x, sigma, noise_scale = NULL) {
  eps <- torch::torch_randn_like(x)
  if (!is.null(noise_scale)) {
    eps <- eps * noise_scale
  }
  x + sigma * eps
}


#' @title Append the current noise level as a discriminator input column
#'
#' @description Diffusion-GAN's noise-level conditioning: the sampled `sigma`
#'   of the current step is broadcast to one extra column of the (already
#'   packed) discriminator input.
#'
#' @param x A torch tensor of shape (batch, d).
#' @param sigma The noise level of this step.
#'
#' @return A torch tensor of shape (batch, d + 1).
#' @keywords internal
append_sigma_column <- function(x, sigma) {
  if (is.null(sigma)) sigma <- 0
  sigma_col <- torch::torch_full(c(x$shape[1], 1), sigma,
                                 device = x$device)$to(dtype = x$dtype)
  torch::torch_cat(list(x, sigma_col), dim = 2)
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
    n_rows <- nrow(dataset)
    # Use replacement if batch_size > n_rows
    use_replace <- batch_size > n_rows
    dataset[sample(n_rows, size = batch_size, replace = use_replace)]$to(device = device)
  }
}


#' @title Pack Samples for PacGAN
#'
#' @description Reshapes a batch of samples for PacGAN by concatenating `pac`
#'   consecutive samples along the feature dimension. This allows the discriminator
#'   to see multiple samples at once, helping it detect mode collapse.
#'
#' @param data A torch tensor of shape (batch_size, data_dim)
#' @param pac Number of samples to pack together. batch_size must be divisible by pac.
#'
#' @return A torch tensor of shape (batch_size/pac, data_dim*pac)
#' @keywords internal
pack_samples <- function(data, pac) {
  if (pac == 1) {
    return(data)
  }

  batch_size <- data$shape[1]
  data_dim <- data$shape[2]

  # Reshape from (batch_size, data_dim) to (batch_size/pac, pac*data_dim)
  # First reshape to (batch_size/pac, pac, data_dim)
  # Then flatten the last two dimensions
  packed <- data$view(c(batch_size %/% pac, pac * data_dim))

  return(packed)
}


#' @title Adjust Learning Rate
#'
#' @description Internal helper function to adjust optimizer learning rates
#'   according to the specified schedule.
#'
#' @param optimizer A torch optimizer object
#' @param initial_lr The initial learning rate
#' @param current_epoch The current epoch number (1-indexed)
#' @param total_epochs Total number of training epochs
#' @param lr_schedule The learning rate schedule type
#' @param lr_decay_factor Decay factor for step/exponential schedules
#' @param lr_decay_steps Epochs between decays for step schedule
#'
#' @return The new learning rate (invisibly). Modifies optimizer in place.
#' @keywords internal
adjust_learning_rate <- function(optimizer,
                                  initial_lr,
                                  current_epoch,
                                  total_epochs,
                                  lr_schedule,
                                  lr_decay_factor,
                                  lr_decay_steps) {
  new_lr <- switch(
    lr_schedule,
    "constant" = initial_lr,
    "step" = {
      # Reduce LR by decay_factor every decay_steps epochs
      num_decays <- current_epoch %/% lr_decay_steps
      initial_lr * (lr_decay_factor ^ num_decays)
    },
    "exponential" = {
      # Exponential decay each epoch
      initial_lr * (lr_decay_factor ^ (current_epoch - 1))
    },
    "cosine" = {
      # Cosine annealing from initial_lr to 0
      initial_lr * (1 + cos(pi * current_epoch / total_epochs)) / 2
    },
    initial_lr  # fallback
  )

  # Update learning rate in optimizer
  # In R torch, we need to modify param_groups in place using index access
  for (i in seq_along(optimizer$param_groups)) {
    optimizer$param_groups[[i]]$lr <- new_lr
  }

  invisible(new_lr)
}


#' @title Resolve the discriminator regularization settings
#'
#' @description Internal helper that turns the user facing `regularization`
#'   argument (and the legacy `gp_lambda` argument) of [gan_trainer()] into a
#'   fully specified list with the fields `type`, `gamma` and `lazy_interval`.
#'
#' @param regularization The `regularization` argument, either NULL or a list.
#' @param gp_lambda The legacy `gp_lambda` argument.
#' @param gp_lambda_supplied Was `gp_lambda` supplied explicitly by the caller?
#' @param value_function The `value_function` argument of [gan_trainer()].
#'
#' @return A list with the fields type, gamma and lazy_interval.
#' @keywords internal
resolve_regularization <- function(regularization,
                                   gp_lambda = 10,
                                   gp_lambda_supplied = FALSE,
                                   value_function = "original") {
  valid_types <- c("none", "wgan-gp", "r1", "r2", "r1r2")

  is_wgan_gp <- is.character(value_function) &&
    length(value_function) == 1 && value_function == "wgan-gp"

  if (is.null(regularization)) {
    # Preserve the previous behaviour exactly.
    if (is_wgan_gp && is.numeric(gp_lambda) && gp_lambda > 0) {
      return(list(type = "wgan-gp", gamma = gp_lambda, lazy_interval = 1L))
    }
    return(list(type = "none", gamma = 0, lazy_interval = 1L))
  }

  if (!is.list(regularization)) {
    stop("regularization must be NULL or a list with fields type, gamma and lazy_interval")
  }

  type <- regularization$type
  if (is.null(type)) {
    stop("regularization must have a 'type' field")
  }
  type <- match.arg(as.character(type)[1], valid_types)

  gamma <- if (is.null(regularization$gamma)) 10 else regularization$gamma
  if (!is.numeric(gamma) || length(gamma) != 1 || gamma < 0) {
    stop("regularization$gamma must be a single non-negative number")
  }

  lazy_interval <- if (is.null(regularization$lazy_interval)) {
    1L
  } else {
    regularization$lazy_interval
  }
  if (!is.numeric(lazy_interval) || length(lazy_interval) != 1 ||
      lazy_interval < 1 || lazy_interval != as.integer(lazy_interval)) {
    stop("regularization$lazy_interval must be a single integer >= 1")
  }
  lazy_interval <- as.integer(lazy_interval)

  if (type == "none") {
    gamma <- 0
  }

  # regularization takes precedence over gp_lambda; warn on disagreement.
  if (gp_lambda_supplied) {
    disagrees <- if (type == "wgan-gp") {
      !isTRUE(all.equal(gamma, gp_lambda))
    } else {
      is.numeric(gp_lambda) && gp_lambda > 0
    }
    if (disagrees) {
      warning(
        sprintf(
          "Both 'regularization' and 'gp_lambda' were supplied and they disagree (regularization: type = '%s', gamma = %s; gp_lambda = %s). 'regularization' takes precedence and 'gp_lambda' is ignored.",
          type, format(gamma), format(gp_lambda)
        )
      )
    }
  }

  list(type = type, gamma = gamma, lazy_interval = lazy_interval)
}
