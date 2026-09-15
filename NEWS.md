# RGAN 0.3.0 (development)

## Any-order conditioning with condition dropout

Experiment 5 of the 2026 experiment plan (report items T3 and L7). One new
`conditioning` argument to `gan_trainer()` gives CTGAN's conditional vector,
TabMT-style random-mask any-order conditioning, conditional sampling and
imputation from a single model. Every new argument defaults to the previous
behaviour, and with `conditioning = NULL` the training path -- including the
random number draws -- is bit-identical to before.

* `gan_trainer(..., conditioning = list(type, mask_prob, cond_dropout,
  log_frequency, resample_rare, consistency_weight))`. The condition of a row
  is `[mask, masked values]`: a 0/1 mask over the column blocks of the
  transformed matrix (the blocks of `output_info_blocks()`, so a
  `"mode_specific"` column contributes a mode-indicator block and a value
  block, and the two are always observed together), followed by the row's
  transformed values with every unobserved block zeroed. The generator is fed
  `cat(z, condition)`, the discriminator `cat(row, condition)` per sample,
  appended **before** PacGAN packing. The fake rows of a step reuse the
  conditions of the real rows, CTGAN's design, so the critic compares
  `(real, cond)` against `(fake, cond)`.
  * `type = "single_column"` reproduces CTGAN (Xu et al., NeurIPS 2019): the
    mask observes exactly one categorical block, the column is drawn
    uniformly and the category with probability proportional to
    `log(count + 1)`, and the real batch is sampled **by** that condition
    (training-by-sampling), from a per-(column, category) row index built once
    at the start of training.
  * `type = "random_mask"` is the TabMT / TabularARGN any-order variant: the
    observed subset is drawn per row (`mask_prob = "uniform"` draws the
    per-row Bernoulli rate from `U(0, 1)`, so every masking rate is trained)
    and the observed values are the real row's own, so no sampler is needed.
    `resample_rare = TRUE` re-draws a fraction of the rows by a log-frequency
    category of one of their observed categorical blocks, so rare levels are
    conditioned on far more often than their empirical frequency; the real
    row is re-drawn, not only the condition values, so the
    `(real row, condition)` pair the critic sees stays consistent.
  * `cond_dropout` (default 0.1) replaces a row's condition with the
    all-unobserved mask, the classifier-free-guidance device of TabDiff. It is
    what makes unconditional sampling from a conditionally trained model well
    defined.
  * `consistency_weight` (default 1) weights `conditioning_consistency_loss()`
    on the generator: cross-entropy against the observed one-hot for
    categorical blocks (`"softmax"` and the mode-indicator sub-block of
    `"mode_specific"`), mean squared error for numeric and `"cossin"` blocks,
    averaged over the observed cells. This is the any-order generalisation of
    CTGAN's cross-entropy term on its single conditional column.
  * The resolved list is returned as `settings$conditioning`, its width as
    `settings$cond_dim`, and the per-block metadata that sampling needs
    (blocks, block-to-column mapping, categorical block ids, category values,
    counts, frequency and log-frequency tables, per-category row index) as
    `trained_gan$conditioning_meta`.
* `TabularGenerator(..., cond_dim = 0)` and `TokenGenerator(..., cond_dim)`
  take the concatenated `noise_dim + cond_dim` input; `cond_dim = 0` (the
  default) leaves both architectures untouched. `gan_trainer()` builds its
  default generator with `cond_dim = n_blocks + data_dim`, the default MLP
  discriminator with `pac * (data_dim + cond_dim)` inputs, and a custom
  `TokenDiscriminator` needs `extra_inputs = n_blocks + data_dim`.
* `sample_synthetic_data(trained_gan, transformer, n_samples, condition)`
  takes a `condition` data.frame in the **original** column space, where `NA`
  means "not observed"; one row is broadcast to all samples, `n` rows give one
  condition per sample. With `condition = NULL` a `"random_mask"` model is
  sampled with the all-unobserved mask, while a `"single_column"` model
  follows CTGAN and draws a `(column, category)` condition per row from the
  **empirical** frequency table (not the log-frequency table used in
  training), so the marginal of the conditioned column is preserved.
* `impute_synthetic(trained_gan, transformer, data_with_NA, n_draws)`
  conditions every row on its own observed cells and returns the data with the
  `NA` cells filled from the generator; observed cells are copied back
  unchanged.
* Post-GAN boosting samples conditionally trained generators with the
  all-unobserved condition; conditional post-GAN boosting (a condition-aware
  discriminator score) is not implemented yet.
* Tests in `tests/testthat/test-conditioning.R`, including the bit-identity of
  the `conditioning = NULL` path against a reference recorded before the
  change.

## Column-token architectures with attention

Experiment 4 of the 2026 experiment plan (report items T1, T4, T5). Two new
exported `torch::nn_module`s treat every column of the transformed data as a
token and run attention over those tokens, instead of pushing a flat vector
through an MLP. Both are drop-in replacements in `gan_trainer()` and neither
changes any default, so existing code is unaffected.

* `TokenDiscriminator(output_info, d_model, n_heads, n_layers, attention,
  row_attention, pac, extra_inputs, dropout, sigmoid)` -- an FT-Transformer
  critic.
  * Tokeniser: one `nn_linear(dim_j, d_model)` per `output_info` block plus a
    learned column-ID embedding, so a width-1 numeric block becomes
    `W_j x_j + b_j` (Gorishniy et al., NeurIPS 2021).
  * `attention = "l2"` (the default) is the ViTGAN attention of Lee et al.
    (ICLR 2022, arXiv:2107.04589): logits `-||q_i - k_j||^2 / sqrt(d_head)`
    with the query and key projections **tied** to one weight matrix.
    Dot-product attention is not Lipschitz and, as that paper shows, spectral
    norm or a gradient penalty alone do not stabilise a transformer critic.
    `attention = "dot"` gives the standard scaled dot product. The squared
    distance is computed by expansion rather than with `torch_cdist()`, whose
    square root has an undefined derivative on the zero diagonal that tied
    Q/K weights produce.
  * `row_attention = TRUE` adds a SAINT-style inter-sample attention block
    after every column-attention block (Somepalli et al. 2021,
    arXiv:2106.01342): the token tensor is transposed so the batch becomes the
    sequence dimension and each token attends across the rows of the
    minibatch. This is a learned, differentiable generalisation of PacGAN and
    minibatch discrimination. Cost is O(batch^2) per token, which is fine for
    batches of a few hundred rows.
  * `pac > 1` is supported and must match the `pac` given to `gan_trainer()`:
    the `pac` samples of a pack become `pac * n_cols` tokens, sharing the
    column-ID embedding and carrying an extra learned pack-slot embedding, so
    the critic can compare the samples inside a pack.
  * `extra_inputs` reserves one additional token, fed by its own linear layer
    from the last `extra_inputs` columns of each sample. This is the slot for
    an instance-noise level sigma or a condition vector.
  * Pooling is by a learned `[CLS]` token, then LayerNorm and
    `nn_linear(d_model, 1)`, with an optional sigmoid.
  * The module contains no in-place operations, so it is twice differentiable
    and works with `regularization = list(type = "r1r2", gamma = ...)`.
* `TokenGenerator(noise_dim, output_info, d_model, n_heads, n_layers,
  attention, tau, dropout, cond_dim)` -- the generator-side counterpart. The
  noise (optionally concatenated with a condition vector) is projected to
  `n_cols` tokens, column-ID embeddings are added, `n_layers` pre-LayerNorm
  transformer blocks run over the tokens, and each token is decoded by its own
  `nn_linear(d_model, dim_j)` head. The `forward(input, hard = NULL)` signature
  matches `TabularGenerator`.

Because there is no positional encoding beyond the learned column-ID
embedding, both modules are permutation equivariant in the columns (report
item T4); the test suite checks this explicitly.

The per-block output activations of `TabularGenerator` are now in an internal
helper, `apply_output_heads(data, output_info, tau, hard, training)`, shared
with `TokenGenerator`, so the two generators emit values in exactly the same
space. `TabularGenerator`'s behaviour is unchanged; its `output_info`
validation moved to the internal `validate_output_heads()`.

### Corrected `SelfAttention` documentation

The internal `SelfAttention` layer used by `TabularGenerator(attention = TRUE)`
was documented as capturing "relationships between features" and dependencies
between columns. It does no such thing: it splits *one hidden vector* into
`num_heads` slices and attends over those slices, with nothing aligned to the
data columns. The layer is unchanged for backward compatibility and the
documentation now describes it as hidden-vector slice mixing, pointing at
`TokenGenerator`/`TokenDiscriminator` for genuine per-column attention.

## Instance noise and adaptive augmentation

Experiment 3 of the 2026 experiment plan. `gan_trainer()` gains two arguments,
both defaulting to `NULL`, in which case the training path (including every
random number drawn) is bit-identical to the previous behaviour.

* `instance_noise`, a list turning on Diffusion-GAN style adaptive instance
  noise (Wang et al., ICLR 2023, arXiv:2206.02262), simplified here to plain
  Gaussian noise without the diffusion mixture. A noise ceiling `T` is kept in
  `[sigma_min, sigma_max]`; every discriminator step draws one
  `sigma ~ U(sigma_min, T)` for the batch and adds `N(0, sigma^2)` noise to the
  real *and* the fake batch before PacGAN packing and before the discriminator.
  The R1/R2/WGAN-GP penalties are computed on exactly those noised, packed
  tensors, and the generator step re-uses the same `sigma`, with the gradient
  flowing through the noise addition.
  * `adaptive = TRUE` (default) moves `T` by `sigma_max / 100` every
    `update_every` (default 4) discriminator steps, up if the
    discriminator-overfitting statistic `r_d` exceeds `target_rd`
    (default 0.6) and down otherwise, clamped to `[sigma_min, sigma_max]`.
  * How `r_d` is measured depends on what the discriminator output means under
    the chosen value function (new internal `resolve_rd_mode()`). The original
    ADA statistic `mean(sign(D(real)))` is only informative when the output is
    anchored at zero:
    * relativistic (paired) losses such as `"rpgan"`, and the
      Wasserstein/critic family (`"wasserstein"`, `"wgan-gp"`, `"f-wgan"`),
      leave the absolute level of `D` unconstrained -- only
      `D(real) - D(fake)` enters the loss -- so
      `r_d = mean(sign(D(real_i) - D(fake_i)))` over the paired batch, the
      fraction of correctly ranked pairs mapped to `[-1, 1]`. With the
      original statistic the controllers of an RpGAN run stayed pinned at
      their floor for the whole of training.
    * `"original"`, whose default discriminator has a sigmoid output, uses
      `r_d = mean(sign(D(real) - 0.5))`;
    * the zero-anchored logit losses `"non-saturating"` and `"hinge"` keep
      `r_d = mean(sign(D(real)))`;
    * a user-supplied value function uses the paired definition when it
      carries the attribute `requires_real_scores = TRUE`, and
      `mean(sign(D(real)))` otherwise.
  * `per_type = TRUE` (default) halves the noise on categorical blocks
    (`"softmax"`, `"cossin"`) and on the mode-indicator sub-block of a
    `"mode_specific"` block, a simple per-type schedule inspired by CDTD
    (Mueller et al., ICLR 2025). Numeric columns keep a multiplier of 1, and
    without `output_info` every column counts as numeric.
  * `condition_d = TRUE` appends the current `sigma` as an extra discriminator
    input column, so the default discriminator's input dimension becomes
    `data_dim * pac + 1`. A custom `discriminator` must be built with that
    extra input itself.
* `augmentation`, a list turning on ADA-style adaptive, differentiable
  augmentation of the discriminator inputs (Karras et al., NeurIPS 2020),
  applied to real and fake batches after the instance noise and before packing.
  The operations live in the new internal `augment_batch()` in
  `R/augmentation.R`: `"mask"` (a row's column block is zeroed with probability
  `p`), `"mixup"` (the numeric part of a row is mixed with a random other row
  of the batch with weight `lambda ~ U(0.5, 1)`) and `"swap"` (paired rows
  exchange one random categorical block, a cheap tabular CutMix in the spirit
  of TabCutMix, Fang et al., ICML 2025, which preserves the multiset of values
  in each block). All three are differentiable in the data, so the generator
  gradient flows through them, and all randomness comes from torch, so the
  result is deterministic given `torch::torch_manual_seed()`. `p` is adapted on
  the same `r_d` statistic, by `p_max / 100` every `update_every` steps,
  clamped to `[0, p_max]`. All three ops are fully vectorised over the batch:
  `"swap"` in particular builds one `(batch, n_cat_blocks)` selection matrix
  and one partner-row index vector, expands the selection to columns by a
  single matmul with a `(n_cat_blocks, data_dim)` block-to-column map, and
  blends `x` with one row-gathered copy of itself, so its cost does not grow
  with the number of categorical columns.
* Both features work with `pac > 1` (noise and augmentation are applied per row
  before packing), with every value function, and with every `regularization`
  type.
* With `track_loss = TRUE` the trajectories are returned as `losses$noise_T`
  and `losses$aug_p`; the final values are in `settings$noise_T_final` and
  `settings$aug_p_final`, and the resolved setting lists in
  `settings$instance_noise` and `settings$augmentation`.
  `print.trained_RGAN()` reports both.
* New tests in `tests/testthat/test-instance-noise.R` and
  `tests/testthat/test-augmentation.R`.

## Representation: quantile numerics and smooth categoricals

Experiment 2 of the 2026 experiment plan. `data_transformer$fit()` gains three
arguments, all defaulting to the previous behaviour:

* `continuous_transform`, one of `"zscore"`, `"mode_specific"` or
  `"quantile"`. When `NULL` (the default) it is derived from the old
  `mode_specific` flag, so existing code is unaffected; if both are supplied
  and disagree, `continuous_transform` wins and a warning is emitted.
  * `"quantile"` is the Gaussian quantile transform used by TabDDPM and TabSyn
    (Kotelnikov et al. 2023): a column is mapped through its empirical CDF and
    then `qnorm()`, so the transformed column is approximately standard normal
    whatever the original shape. The CDF is stored as a table of up to
    `n_quantiles` (default 1000) empirical quantiles and both directions are a
    linear interpolation over that table, which makes the round trip exact for
    values inside the fitted range. The CDF position is clipped to
    `[1e-4, 1 - 1e-4]`, so the two extreme order statistics come back
    approximately. Constant columns are handled. New `output_info` type
    `"quantile"` with one output column.
* `categorical_encoding`, one of `"onehot"` (unchanged, type `"softmax"`),
  `"cossin"` or `"embedding_ready"`.
  * `"cossin"` is the TabRep encoding (Hsi et al. 2025, arXiv:2504.04798):
    level `c` of `K` becomes `(cos(2*pi*(c-1)/K), sin(2*pi*(c-1)/K))`, so a
    categorical column costs two output dimensions whatever its cardinality
    and needs no Gumbel-Softmax. The inverse takes `atan2()` and snaps to the
    nearest level by circular distance. New `output_info` type `"cossin"`.
  * `"embedding_ready"` keeps the column as a single integer index in `1..K`
    (`output_info` type `"index"`). This is only the data path for a future
    learned-embedding generator; `TabularGenerator` has no head for it.
* `n_quantiles`, the size of the quantile reference table.

`data_transformer$last_snap_stats` reports, per cos/sin column, the mean and
90th percentile of the snap distance of the most recent `inverse_transform()`
call, measured in units of the inter-level angle `2*pi/K` (so it lies in
`[0, 0.5]`). This is the diagnostic the experiment plan uses to decide whether
the learned-embedding variant is needed.

`TabularGenerator` gains the two matching heads:

* `"quantile"`: a plain linear output with no activation. Quantile-normalised
  columns are unbounded, and a `tanh` head would pile generated mass on the
  clipping value exactly in the tails the representation is meant to fix.
* `"cossin"`: the two outputs are projected onto the unit circle
  (`x / (||x|| + 1e-8)`), so the generator can only emit points the encoding
  can produce.

`TabularGenerator` now validates the `output_info` types at construction and
gives a clear error for the head-less `"index"` type.

## Modern GAN losses and zero-centred gradient penalties

Experiment 1 of the 2026 experiment plan: the R3GAN recipe
(Huang et al., NeurIPS 2024, arXiv:2501.05441) is now available in
`gan_trainer()`, alongside two further standard losses. Every new argument
defaults to the previous behaviour, so existing code is unaffected.

* New value functions in `R/value-functions.R`, all exported:
  * `RpGAN_value_fct()` -- the relativistic pairing loss
    (Jolicoeur-Martineau 2019), which R3GAN builds on. Real and fake scores are
    paired by batch index, so the two batches must have the same size (they
    always do in `gan_update_step()`).
  * `NSGAN_value_fct()` -- the non-saturating GAN loss written on logits with
    `softplus`, for use with the linear-output discriminator.
  * `hinge_value_fct()` -- the hinge loss of Lim & Ye (2017).
  * `r1_penalty()` and `r2_penalty()` -- zero-centred gradient penalties
    (Mescheder et al. 2018): the mean squared L2 norm of the gradient of the
    discriminator output with respect to its real (R1) or fake (R2) input,
    computed with `create_graph = TRUE`. The trainer multiplies them by
    `gamma / 2`. The discriminator's output bias receives no gradient from
    these penalties, which is expected.

* `gan_trainer()` gains:
  * `value_function` options `"non-saturating"`, `"hinge"` and `"rpgan"`. As
    for the Wasserstein options, these use a linear-output discriminator; only
    `"original"` uses a sigmoid. An unknown string now raises an error instead
    of failing later.
  * `regularization`, a list with `type` (`"none"`, `"wgan-gp"`, `"r1"`,
    `"r2"`, `"r1r2"`), `gamma` (default 10) and `lazy_interval` (default 1:
    StyleGAN2-style lazy regularization, applying the penalty on every
    `lazy_interval`-th discriminator step and scaling it by `lazy_interval`).
    It takes precedence over `gp_lambda` and warns if an explicitly supplied
    `gp_lambda` disagrees. With `pac > 1` the penalties are computed on the
    packed tensors, i.e. exactly the inputs the discriminator is called with.
    The resolved list is stored in `settings$regularization`.
  * `adam_betas`, e.g. `c(0, 0.99)` as in R3GAN. `NULL` (the default) keeps
    torch's own `c(0.9, 0.999)`. Used for both default optimizers.
  * `discriminator_dropout` (default 0.5, the previous hard-coded value), so
    the R3GAN arm can train without dropout without supplying a custom module.
  * `settings$n_d_steps`, the total number of discriminator updates performed,
    so training budgets can be equalised across arms.

* `gan_update_step()` gains `regularization`, `step` and `paired_loss`
  arguments. `paired_loss = TRUE` recomputes the real scores (without gradient)
  under the updated discriminator before the generator step, which the
  relativistic loss needs; for every other loss the real scores are now passed
  to the generator loss detached, which changes no result.

* `print.trained_RGAN()` reports the regularization type and gamma (and the
  lazy interval, Adam betas and discriminator step count when set).

## One-step distillation of a flow-matching teacher

The second half of Experiment 6 (report item L5): the Diffusion2GAN recipe of
Kang et al. (ECCV 2024, arXiv:2405.05967) applied to tables. The teacher ODE
is a deterministic map from noise `z` to a sample `x_hat`; the GAN generator
is trained with `L_adv + lambda * ||G(z) - x_hat||^2` on exactly those pairs,
so the student keeps one-step sampling while inheriting the teacher's
coverage. As with every other addition, `distillation = NULL` leaves the
training path -- including the random number draws -- bit-identical to before.

* `gan_trainer(..., distillation = list(pairs, teacher, n_pairs, n_steps,
  solver, refresh_every, lambda, loss, huber_delta, per_type_weight,
  adversarial, noise_match))`. Every generator step draws a minibatch of
  `(z_i, x_i)` pairs, evaluates `G(z_i)` and adds `lambda` times the mean
  regression loss to the generator loss.
  * The adversarial part is **unchanged**: it still uses a fresh noise draw
    and the real data batch. The discriminator never sees the teacher's
    targets as "real", so this is distillation of the noise-to-data map, not
    training on teacher samples.
  * `pairs` is a `list(z, x)` from `flow_pairs()`. If it is missing, `teacher`
    (a `"trained_RFlow"`) generates `n_pairs` (default `4 * nrow(data)`) pairs
    with `n_steps` (default 50) `solver` (default Heun) steps at the start of
    training, and again every `refresh_every` epochs if that is set. Passing a
    bare `"trained_RFlow"` as `distillation` is shorthand for
    `list(teacher = ...)`.
  * `noise_match = TRUE` (default) feeds the teacher's own `z` to the
    generator, which is what makes the regression a *trajectory* regression;
    it requires `noise_dim == ncol(pairs$z)` and otherwise errors with the
    value to set. `noise_match = FALSE` draws fresh noise, an ablation that
    keeps only a distribution-level pull.
  * `loss` is `"mse"` (default) or `"huber"`, and `per_type_weight = TRUE`
    weights the per-column residual with `flow_loss_weights()`, the same
    CDTD-style block calibration the teacher trains under.
  * `adversarial = FALSE` skips the discriminator step **entirely** -- no real
    batch, no penalty, no `d_optim$step()` -- and trains the generator on the
    regression term alone. `settings$n_d_steps` then stays 0, and the new
    `settings$n_g_steps` counts the generator updates.
  * With `track_loss = TRUE` the per-step regression term (before `lambda`) is
    returned as `losses$distill_loss`, and the recipe is recorded in
    `settings$distillation`.

* `distill_to_gan(teacher, data, transformer, n_pairs, n_steps, lambda,
  solver, pair_seed, distillation, ...)` -- exported wrapper that generates
  the pairs, sets `noise_dim` to the teacher's `data_dim` and forwards
  everything else to `gan_trainer()`. A `data.frame` `data` is transformed
  with the supplied `transformer`, whose `output_info` is used by default.

* New internals: `distill_regression_loss()` (weighted MSE / Huber, exactly
  zero at the targets), `resolve_distillation()` and `validate_flow_pairs()`.
  The `print()` method for `"trained_RGAN"` reports the recipe and the
  generator-step count.

Functional check on `adult` (quantile + cos/sin, 5040 steps at batch 500,
96,516 pairs at 50 Heun steps, CPU): see `experiments/results/exp6_distill`.

## A flow-matching teacher (`R/flow-matching.R`)

Experiment 6 of the 2026 experiment plan, report item L5. A conditional
flow-matching model that RGAN can train on its own transformed data, intended
both as a diffusion-family fidelity reference and as the teacher of a later
one-step distillation. Nothing here touches `gan_trainer()`; distillation is a
separate, later change.

* `FlowVectorField(data_dim, hidden_units, time_embed_dim, activation,
  dropout)` -- an exported `torch::nn_module` for the velocity field
  `v(x_t, t)`. A DDPM-style sinusoidal embedding of `t` (frequencies from 1 to
  1e-4 on `t` scaled by 1000, Ho et al., arXiv:2006.11239) goes through a
  two-layer MLP and is concatenated to `x_t`; the result passes through
  pre-norm residual blocks `h + W2 act(W1 LayerNorm(h))` and a final
  LayerNorm + linear output of width `data_dim`. Blocks whose input and output
  widths differ get a linear shortcut. No operation is in place, so the module
  is safe under double backward.

* `flow_matching_trainer(data, output_info, epochs, batch_size, base_lr,
  lr_schedule, weight_decay, path, sigma_min, time_sampling, per_type_weight,
  ema_decay, vector_field, device, seed, track_loss, verbose)` -- simulation-free
  conditional flow matching (Lipman et al., ICLR 2023, arXiv:2210.02747; Tong
  et al., TMLR 2024, arXiv:2302.00482). For every data row `x1` it draws
  `x0 ~ N(0, I)` and a time `t`, forms the point on the conditional path in
  closed form and regresses the network on the conditional velocity with an
  MSE.
  * `path = "ot"` (default) is the optimal-transport / rectified path
    `x_t = (1 - (1 - sigma_min) t) x0 + t x1` with target
    `u = x1 - (1 - sigma_min) x0`. `path = "vp"` is a variance-preserving
    cosine schedule `x_t = sin(pi t/2) x1 + cos(pi t/2) x0` with its exact
    time derivative as the target.
  * `time_sampling = "logit_normal"` concentrates the training times in the
    middle of the trajectory (Esser et al., ICML 2024, arXiv:2403.03206).
  * `per_type_weight = TRUE` (with an `output_info`) weights the per-column
    squared error by `1 / block_dim`, rescaled to mean 1, so that a wide
    one-hot block does not outweigh a single numeric column. This is the CDTD
    loss calibration (Mueller et al., ICLR 2025, arXiv:2312.10431) carried over
    to a flow-matching MSE.
  * Gaussian noise is applied to **all** columns. That is the TabRep premise
    (arXiv:2504.04798): with `continuous_transform = "quantile"` and
    `categorical_encoding = "cossin"` every column is already a smooth real
    number, so no multinomial diffusion, no VAE and no Gumbel-softmax head is
    needed. Any finite numeric matrix is accepted, however.
  * Adam with an optional cosine learning-rate schedule and an EMA of the
    weights (warm-up `min(ema_decay, (1 + step) / (10 + step))`); the returned
    `vector_field` carries the EMA weights and `vector_field_raw` the final
    ones. The return value has class `"trained_RFlow"` and a `print()` method.

* `sample_flow(trained_flow, n_samples, n_steps, solver, z, return_trajectory,
  device)` -- integrates `dx/dt = v(x, t)` from `t = 0` (noise) to `t = 1`
  (data) with fixed steps, `solver = "euler"` (one network evaluation per step)
  or `"heun"` (two, second order). Passing `z` fixes the starting noise, which
  makes the map deterministic and reproducible.

* `flow_pairs(trained_flow, n_pairs, n_steps, solver, batch_size, seed)` --
  batched `(z, x_hat)` noise-to-sample pairs from the teacher ODE, the targets
  a Diffusion2GAN-style pair regression needs (Kang et al., ECCV 2024,
  arXiv:2405.05967).

First numbers on `adult` at the Experiment 1 screening budget (5040 steps at
batch 500, quantile + cos/sin, CPU): Shape 0.103, Trend 0.028, TSTR 0.910
(TRTR 0.929), rare-level recall 1.000, XGB-C2ST 0.999, distance-MIA AUC 0.502,
112 s to train, 4.3 s to sample 10k rows with 50 Heun steps. The best
Experiment 1 GAN (RpGAN + R1/R2) reached Shape 0.119, Trend 0.030, TSTR 0.890,
rare-level recall 0.854 on the same budget. On `toy_modes` the teacher reaches
mode coverage 1.000 and mode KL 0.178 against the GAN's 0.971 / 0.369.

### Bug fix: a `trained_RFlow` now survives a save/load round trip

* `flow_matching_trainer()` stored `settings$device` as a `torch_device`,
  which is an external pointer. After `saveRDS()` / `readRDS()` that pointer
  is dead, so every `sample_flow()` and `flow_pairs()` call on a reloaded
  teacher failed with "external pointer is not valid". `settings$device` is
  now the plain character label (`"cpu"`, `"mps"`, `"cuda:0"`, ...) and is
  resolved to a live device at the point of use by the new internal helpers
  `flow_device_label()` / `flow_resolve_device()`. A device recorded at
  training time but unavailable on the machine doing the sampling (an
  `"mps"` teacher reloaded on Linux, say) now falls back to the CPU with a
  warning instead of erroring.

* `flow_pairs()` gains the `device` argument that `sample_flow()` already
  had, so the teacher's integration device can be overridden on reload.

* New `save_flow()` / `load_flow()`. As of torch 0.17 an `nn_module` does
  **not** round-trip through `saveRDS()` -- the object reloads with the right
  class but every parameter is a dead external pointer -- so fixing the
  settings alone is not enough to make a teacher reusable across sessions.
  `save_flow(fit, path)` writes the network with `torch::torch_save()`
  (`<path>_vector_field.pt`, plus `<path>_vector_field_raw.pt` when the EMA
  and raw weights differ) and the settings and loss curve with `saveRDS()`
  (`<path>_settings.rds`); `load_flow(path, device = NULL)` returns a
  `"trained_RFlow"` that `sample_flow()` and `flow_pairs()` accept directly.
  The EMA weights are what `save_flow()` stores as the sampling network,
  because `trained_flow$vector_field` already carries them after training.
  This follows the `save_gan()` / `load_gan()` convention.

* Tests: `sample_flow()` with a fixed `z` gives bit-identical output before
  and after a `save_flow()` / `load_flow()` round trip for both solvers,
  `flow_pairs()` still reproduces the teacher ODE on the reloaded object, and
  the `settings$device` label survives a plain RDS round trip.

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
