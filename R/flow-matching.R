# ---------------------------------------------------------------------------
# R/flow-matching.R
#
# A conditional flow-matching (CFM) teacher for tabular data, Experiment 6 of
# the 2026 experiment plan.
#
# The model is deliberately representation-agnostic: it treats the transformed
# data matrix as one continuous vector space. That is the TabRep design
# (arXiv:2504.04798) -- with `continuous_transform = "quantile"` and
# `categorical_encoding = "cossin"` in `data_transformer`, every column is a
# smooth real number, so a plain Gaussian flow needs no multinomial diffusion,
# no VAE and no Gumbel-softmax head. Gaussian noise is therefore applied to all
# columns alike; the only type awareness is the optional per-type loss
# calibration (`per_type_weight`), which is the CDTD idea (arXiv:2312.10431)
# that per-type loss terms must be equalised so that wide one-hot blocks do not
# dominate a single numeric column.
#
# References
#   Lipman et al. (ICLR 2023, arXiv:2210.02747)  Flow matching
#   Tong et al. (TMLR 2024, arXiv:2302.00482)    Conditional / OT flow matching
#   Ho et al. (NeurIPS 2020, arXiv:2006.11239)   DDPM sinusoidal time embedding
#   Esser et al. (ICML 2024, arXiv:2403.03206)   logit-normal time sampling
# ---------------------------------------------------------------------------


#' @title Sinusoidal time embedding
#'
#' @description Internal helper. Vaswani-style sinusoidal features of a time
#'   vector `t` in \[0, 1\], following the DDPM convention of scaling `t` by
#'   1000 before applying frequencies that run geometrically from 1 down to
#'   1e-4 (i.e. periods 1 to 1e4).
#'
#' @param t A torch tensor of shape `(batch)` or `(batch, 1)` with values in
#'   \[0, 1\].
#' @param dim Integer, the embedding width. Odd widths are zero padded in the
#'   last position.
#' @param max_period Numeric, the longest period; 1e4 as in DDPM.
#' @param scale Numeric, the factor applied to `t` before embedding; 1000 as in
#'   DDPM, where the timestep index rather than the normalised time is
#'   embedded.
#'
#' @return A torch tensor of shape `(batch, dim)`.
#' @keywords internal
flow_time_embedding <- function(t, dim, max_period = 1e4, scale = 1000) {
  if (length(dim(t)) > 1) t <- t$reshape(c(-1))
  dim <- as.integer(dim)
  if (dim < 2L) stop("time_embed_dim must be at least 2")
  half <- dim %/% 2L

  device <- t$device
  idx <- torch::torch_arange(start = 0, end = half - 1, dtype = torch::torch_float(),
                             device = device)
  freqs <- torch::torch_exp(-log(max_period) * idx / half)
  args <- t$unsqueeze(2) * scale * freqs$unsqueeze(1)
  emb <- torch::torch_cat(list(torch::torch_sin(args), torch::torch_cos(args)),
                          dim = 2)
  if (2L * half < dim) {
    pad <- torch::torch_zeros(c(t$size(1), dim - 2L * half),
                              dtype = emb$dtype, device = device)
    emb <- torch::torch_cat(list(emb, pad), dim = 2)
  }
  emb
}


#' Resolve a character activation name to a torch functional.
#' @param activation Character, one of "silu", "relu", "gelu", "leaky_relu",
#'   "tanh".
#' @return A function of one tensor.
#' @keywords internal
flow_activation <- function(activation = "silu") {
  activation <- match.arg(activation,
                          c("silu", "relu", "gelu", "leaky_relu", "tanh"))
  switch(activation,
         silu = torch::nnf_silu,
         relu = torch::nnf_relu,
         gelu = torch::nnf_gelu,
         leaky_relu = function(x) torch::nnf_leaky_relu(x, 0.2),
         tanh = torch::torch_tanh)
}


#' @title Flow-matching vector field
#'
#' @description A `torch::nn_module` implementing the velocity field
#'   \eqn{v_\theta(x_t, t)} of a conditional flow-matching model for tabular
#'   data. The architecture is a pre-norm residual MLP: the input row `x_t` is
#'   concatenated with a sinusoidal embedding of the time `t` (passed through a
#'   two-layer MLP first), projected to the first hidden width, and then pushed
#'   through residual blocks of the form
#'   `h + W2 act(W1 LayerNorm(h))`. The output layer is applied after a final
#'   LayerNorm and returns a vector of the same width as the data, which is the
#'   velocity.
#'
#'   Nothing in the module is type aware: it is an ordinary Euclidean vector
#'   field. This is the TabRep (arXiv:2504.04798) premise that with a quantile
#'   transform for numerics and a cos/sin encoding for categoricals every
#'   column is already continuous. See [flow_matching_trainer()].
#'
#'   No operation in `forward()` is in place, so the module is safe under
#'   double backward and under `torch::autograd_grad(create_graph = TRUE)`.
#'
#' @param data_dim Integer, the number of columns of the (transformed) data.
#' @param hidden_units A list (or numeric vector) of hidden widths, one
#'   residual block per element.
#' @param time_embed_dim Integer, the width of the sinusoidal time embedding.
#' @param activation Character, one of `"silu"` (default), `"relu"`,
#'   `"gelu"`, `"leaky_relu"`, `"tanh"`.
#' @param dropout Numeric in \[0, 1), dropout rate inside each residual block.
#'
#' @return A `torch::nn_module`. `forward(x_t, t)` takes `x_t` of shape
#'   `(batch, data_dim)` and `t` of shape `(batch)` (or `(batch, 1)`) with
#'   values in \[0, 1\] and returns a tensor of shape `(batch, data_dim)`.
#'
#' @examples
#' \dontrun{
#' v <- FlowVectorField(data_dim = 8, hidden_units = list(64, 64))
#' x <- torch::torch_randn(5, 8)
#' t <- torch::torch_rand(5)
#' v(x, t)$shape
#' }
#' @export
FlowVectorField <- torch::nn_module(
  "FlowVectorField",
  initialize = function(data_dim,
                        hidden_units = list(512, 512, 512),
                        time_embed_dim = 64,
                        activation = "silu",
                        dropout = 0) {
    data_dim <- as.integer(data_dim)
    if (is.na(data_dim) || data_dim < 1L) stop("data_dim must be a positive integer")
    hidden_units <- as.integer(unlist(hidden_units))
    if (!length(hidden_units) || any(hidden_units < 1L)) {
      stop("hidden_units must contain at least one positive width")
    }
    time_embed_dim <- as.integer(time_embed_dim)
    if (time_embed_dim < 2L) stop("time_embed_dim must be at least 2")
    if (!is.numeric(dropout) || dropout < 0 || dropout >= 1) {
      stop("dropout must be in [0, 1)")
    }

    self$data_dim <- data_dim
    self$hidden_units <- hidden_units
    self$time_embed_dim <- time_embed_dim
    self$activation_name <- match.arg(
      activation, c("silu", "relu", "gelu", "leaky_relu", "tanh"))
    self$act <- flow_activation(self$activation_name)
    self$dropout_rate <- dropout
    self$n_blocks <- length(hidden_units)

    # Small MLP on top of the sinusoidal features (DDPM / Lipman convention).
    self$time_mlp <- torch::nn_sequential(
      torch::nn_linear(time_embed_dim, time_embed_dim),
      if (self$activation_name == "silu") torch::nn_silu() else torch::nn_relu(),
      torch::nn_linear(time_embed_dim, time_embed_dim)
    )

    self$input_proj <- torch::nn_linear(data_dim + time_embed_dim,
                                        hidden_units[1])

    # Residual blocks. A block whose input and output widths differ gets a
    # linear shortcut so the residual sum is always well defined.
    self$norms <- torch::nn_module_list()
    self$fc1 <- torch::nn_module_list()
    self$fc2 <- torch::nn_module_list()
    self$shortcuts <- torch::nn_module_list()
    self$drops <- torch::nn_module_list()

    dim_in <- hidden_units[1]
    for (i in seq_along(hidden_units)) {
      width <- hidden_units[i]
      self$norms$append(torch::nn_layer_norm(dim_in))
      self$fc1$append(torch::nn_linear(dim_in, width))
      self$fc2$append(torch::nn_linear(width, width))
      self$drops$append(torch::nn_dropout(dropout))
      self$shortcuts$append(
        if (dim_in == width) torch::nn_identity() else torch::nn_linear(dim_in, width)
      )
      dim_in <- width
    }

    self$out_norm <- torch::nn_layer_norm(dim_in)
    self$out_proj <- torch::nn_linear(dim_in, data_dim)
  },

  forward = function(x_t, t) {
    if (inherits(t, "numeric") || inherits(t, "integer")) {
      t <- torch::torch_tensor(as.numeric(t), dtype = x_t$dtype,
                               device = x_t$device)
    }
    if (length(dim(t)) > 1) t <- t$reshape(c(-1))
    if (t$size(1) == 1 && x_t$size(1) > 1) {
      t <- t$expand(c(x_t$size(1)))
    }

    temb <- flow_time_embedding(t, self$time_embed_dim)
    temb <- self$time_mlp(temb)

    h <- torch::torch_cat(list(x_t, temb), dim = 2)
    h <- self$input_proj(h)

    for (i in seq_len(self$n_blocks)) {
      z <- self$norms[[i]](h)
      z <- self$fc1[[i]](z)
      z <- self$act(z)
      z <- self$drops[[i]](z)
      z <- self$fc2[[i]](z)
      h <- self$shortcuts[[i]](h) + z
    }

    self$out_proj(self$out_norm(h))
  }
)


#' @title Per-column loss weights from `output_info`
#'
#' @description Internal helper implementing the CDTD-style loss calibration
#'   (Mueller et al., ICLR 2025, arXiv:2312.10431). Every `output_info` block
#'   should contribute the same amount to the flow-matching MSE on average,
#'   otherwise a 40-column one-hot block outweighs a single numeric column by a
#'   factor of 40. Each column of a block of width `d` gets weight `1/d`, and
#'   the whole vector is rescaled to mean 1 so that the weighted loss stays on
#'   the same scale as the unweighted one.
#'
#' @param output_info The `output_info` list of a [data_transformer], or NULL.
#' @param data_dim Integer, number of columns.
#'
#' @return A numeric vector of length `data_dim` with mean 1.
#' @keywords internal
flow_loss_weights <- function(output_info, data_dim) {
  w <- rep(1, data_dim)
  bl <- output_info_blocks(output_info, data_dim)
  if (!isTRUE(bl$ok)) return(w)
  for (b in bl$blocks) {
    d <- b$end - b$start + 1L
    w[b$start:b$end] <- 1 / d
  }
  m <- mean(w)
  if (!is.finite(m) || m <= 0) return(rep(1, data_dim))
  w / m
}


#' Character label of a device.
#'
#' @description Internal helper. A `torch_device` is an external pointer, so
#'   storing one in a settings list makes the list unusable after a
#'   `saveRDS()` / `readRDS()` round trip ("external pointer is not valid").
#'   Settings therefore store the *label* (`"cpu"`, `"mps"`, `"cuda:0"`, ...),
#'   which is plain character and survives serialisation; it is turned back
#'   into a device by [flow_resolve_device()] at the point of use.
#'
#' @param device A character string, a `torch_device`, or `NULL`.
#' @return A length-one character string, or `NULL` for `NULL` input.
#' @keywords internal
flow_device_label <- function(device) {
  if (is.null(device)) return(NULL)
  if (is.character(device)) return(device[[1]])
  if (inherits(device, "torch_device")) {
    ty <- tryCatch(device$type, error = function(e) NULL)
    if (is.null(ty)) return("cpu")            # dead pointer from an old RDS
    idx <- tryCatch(device$index, error = function(e) NULL)
    if (is.null(idx) || is.na(idx)) return(as.character(ty))
    return(sprintf("%s:%d", ty, as.integer(idx)))
  }
  as.character(device)[[1]]
}

#' Resolve a stored device label to a live torch device.
#'
#' @description Internal helper, the counterpart of [flow_device_label()].
#'   Accepts a character label, a live `torch_device` or a stale one recovered
#'   from an RDS file, and always returns a usable `torch_device`. A device
#'   that is not available on the current machine (for example `"cuda"` on a
#'   laptop, or an `"mps"` model reloaded on Linux) falls back to the CPU with
#'   a warning rather than erroring.
#'
#' @param device A character string, a `torch_device`, or `NULL`.
#' @param default Fallback used when `device` is `NULL` or unusable.
#' @return A `torch_device`.
#' @keywords internal
flow_resolve_device <- function(device, default = "cpu") {
  lab <- flow_device_label(device)
  if (is.null(lab) || !nzchar(lab)) lab <- flow_device_label(default) %||% "cpu"
  dev <- tryCatch(torch::torch_device(lab), error = function(e) NULL)
  if (is.null(dev)) {
    warning(sprintf("unknown device '%s'; falling back to \"cpu\"", lab),
            call. = FALSE)
    return(torch::torch_device("cpu"))
  }
  # Check the device can actually hold a tensor on this machine.
  ok <- tryCatch({
    torch::torch_zeros(1, device = dev)
    TRUE
  }, error = function(e) FALSE)
  if (!ok) {
    warning(sprintf("device '%s' is not available here; falling back to \"cpu\"",
                    lab), call. = FALSE)
    return(torch::torch_device("cpu"))
  }
  dev
}


#' Sample training times for flow matching.
#' @param n Integer, number of samples.
#' @param time_sampling "uniform" or "logit_normal".
#' @param device A torch device.
#' @return A torch tensor of shape `(n)` with values in (0, 1).
#' @keywords internal
flow_sample_time <- function(n, time_sampling = "uniform", device = "cpu") {
  if (identical(time_sampling, "logit_normal")) {
    # Stable Diffusion 3 / Esser et al. (2024): emphasise the middle of the
    # trajectory, where the vector field is hardest to learn.
    torch::torch_sigmoid(torch::torch_randn(n, device = device))
  } else {
    torch::torch_rand(n, device = device)
  }
}


#' Build (x_t, target) for one flow-matching minibatch.
#' @param x0 Noise tensor `(batch, dim)`.
#' @param x1 Data tensor `(batch, dim)`.
#' @param t Time tensor `(batch)`.
#' @param path "ot" or "vp".
#' @param sigma_min Numeric, the residual noise of the OT path at `t = 1`.
#' @return A list with `x_t` and `u` (the conditional target velocity).
#' @keywords internal
flow_path <- function(x0, x1, t, path = "ot", sigma_min = 1e-4) {
  tt <- t$unsqueeze(2)
  if (identical(path, "vp")) {
    # Variance-preserving path with a cosine schedule:
    #   x_t = sin(pi t / 2) x1 + cos(pi t / 2) x0
    # so x_0 is pure noise and x_1 is the data point. The conditional target
    # is the exact time derivative of that interpolation.
    a <- torch::torch_sin(tt * (pi / 2))
    s <- torch::torch_cos(tt * (pi / 2))
    x_t <- a * x1 + s * x0
    u <- (pi / 2) * (s * x1 - a * x0)
  } else {
    # Optimal-transport / rectified path (Lipman et al. 2023; Liu et al. 2023)
    x_t <- (1 - (1 - sigma_min) * tt) * x0 + tt * x1
    u <- x1 - (1 - sigma_min) * x0
  }
  list(x_t = x_t, u = u)
}


#' @title Train a conditional flow-matching model on tabular data
#'
#' @description Fits a continuous normalising flow by conditional flow matching
#'   (Lipman et al., ICLR 2023, arXiv:2210.02747; Tong et al., TMLR 2024,
#'   arXiv:2302.00482) on an already transformed numeric data matrix. Training
#'   is simulation free: for every row \eqn{x_1} of the data a noise point
#'   \eqn{x_0 \sim N(0, I)} and a time \eqn{t} are drawn, the point on the
#'   conditional path \eqn{x_t} is formed in closed form, and the network is
#'   regressed on the conditional velocity \eqn{u_t(x_t \mid x_1)} with a mean
#'   squared error.
#'
#'   With the OT (rectified) path,
#'   \deqn{x_t = (1 - (1 - \sigma_{min}) t) x_0 + t x_1, \qquad
#'         u = x_1 - (1 - \sigma_{min}) x_0,}
#'   with the VP path a cosine schedule
#'   \eqn{x_t = \sin(\pi t/2) x_1 + \cos(\pi t/2) x_0} and its exact
#'   derivative.
#'
#'   This is the teacher of Experiment 6 of the RGAN experiment plan, the
#'   "TabRep with flow matching" design: all columns, numeric and categorical
#'   alike, are treated as continuous and corrupted with Gaussian noise. Use it
#'   with `data_transformer$fit(continuous_transform = "quantile",
#'   categorical_encoding = "cossin")`, which encodes level `c` of a
#'   `K`-level column as `(cos 2 pi c/K, sin 2 pi c/K)` and snaps back to the
#'   nearest level on `inverse_transform()`. Any numeric matrix works, however
#'   (a z-scored matrix, a one-hot matrix, or a VAE latent).
#'
#' @param data A numeric matrix, data.frame of numerics, or a torch tensor of
#'   shape `(n, data_dim)`. Typically the output of
#'   `data_transformer$transform()`.
#' @param output_info Optional `output_info` list from a [data_transformer].
#'   Only used for the per-block loss calibration; the model itself is type
#'   agnostic.
#' @param epochs Integer, number of passes over the data.
#' @param batch_size Integer, minibatch size.
#' @param base_lr Numeric, Adam learning rate.
#' @param lr_schedule `"cosine"` (default; anneals `base_lr` to 0 over
#'   `epochs`) or `"constant"`.
#' @param weight_decay Numeric, Adam weight decay.
#' @param path `"ot"` (default, the optimal-transport / rectified path) or
#'   `"vp"` (variance preserving, cosine schedule).
#' @param sigma_min Numeric, residual noise level of the OT path at `t = 1`.
#' @param time_sampling `"uniform"` (default) or `"logit_normal"` (Esser et
#'   al. 2024), which concentrates the training times in the middle of the
#'   trajectory.
#' @param per_type_weight Logical. If `TRUE` (default) and `output_info` is
#'   supplied, weight the per-column squared error so that every `output_info`
#'   block contributes equally on average (weight `1/block_dim`, rescaled to
#'   mean 1). This is the CDTD loss-calibration idea (arXiv:2312.10431) carried
#'   over to a flow-matching MSE.
#' @param ema_decay Numeric in \[0, 1\), the decay of the exponential moving
#'   average of the weights used for sampling. `0` disables the EMA. A warm-up
#'   `min(ema_decay, (1 + step) / (10 + step))` is applied so that early steps
#'   are not dominated by the initialisation.
#' @param vector_field Optional pre-built `torch::nn_module` with signature
#'   `forward(x_t, t)`. Defaults to a [FlowVectorField] with
#'   `hidden_units = list(512, 512, 512)`.
#' @param device `"cpu"`, `"cuda"`, `"mps"` or a `torch_device`.
#' @param seed Optional integer seed for R and torch.
#' @param track_loss Logical, record the mean loss per epoch.
#' @param verbose Logical, print progress.
#'
#' @return An object of class `"trained_RFlow"`, a list with
#'   \describe{
#'     \item{vector_field}{the trained `nn_module`, holding the EMA weights
#'       when `ema_decay > 0`}
#'     \item{vector_field_raw}{the module with the final (non-EMA) weights}
#'     \item{settings}{the resolved training settings, including `data_dim`}
#'     \item{losses}{numeric vector of mean losses per epoch (or `NULL`)}
#'   }
#'
#' @examples
#' \dontrun{
#' x <- scale(RGAN::sample_toydata(n = 2000))
#' fm <- flow_matching_trainer(x, epochs = 300, batch_size = 100,
#'                             vector_field = FlowVectorField(2, list(128, 128)))
#' syn <- sample_flow(fm, n_samples = 1000, n_steps = 50, solver = "heun")
#' plot(syn)
#' }
#' @export
flow_matching_trainer <- function(data,
                                  output_info = NULL,
                                  epochs = 200,
                                  batch_size = 500,
                                  base_lr = 1e-3,
                                  lr_schedule = c("cosine", "constant"),
                                  weight_decay = 0,
                                  path = c("ot", "vp"),
                                  sigma_min = 1e-4,
                                  time_sampling = c("uniform", "logit_normal"),
                                  per_type_weight = TRUE,
                                  ema_decay = 0.999,
                                  vector_field = NULL,
                                  device = "cpu",
                                  seed = NULL,
                                  track_loss = TRUE,
                                  verbose = TRUE) {

  lr_schedule <- match.arg(lr_schedule)
  path <- match.arg(path)
  time_sampling <- match.arg(time_sampling)

  epochs <- as.integer(epochs)
  batch_size <- as.integer(batch_size)
  if (is.na(epochs) || epochs < 1L) stop("epochs must be a positive integer")
  if (is.na(batch_size) || batch_size < 1L) stop("batch_size must be a positive integer")
  if (!is.numeric(base_lr) || base_lr <= 0) stop("base_lr must be positive")
  if (!is.numeric(ema_decay) || ema_decay < 0 || ema_decay >= 1) {
    stop("ema_decay must be in [0, 1)")
  }

  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }

  # Keep the label around: `settings$device` must be plain character so that
  # the returned object survives saveRDS()/readRDS() (a torch_device is an
  # external pointer and is dead after a round trip).
  device_label <- flow_device_label(device)
  device <- if (inherits(device, "torch_device")) device else torch::torch_device(device)

  if (inherits(data, "torch_tensor")) {
    x_data <- data$to(device = device, dtype = torch::torch_float())
  } else {
    m <- as.matrix(data)
    storage.mode(m) <- "double"
    if (anyNA(m) || any(!is.finite(m))) {
      stop("data contains NA/NaN/Inf; the flow-matching teacher needs a finite numeric matrix")
    }
    x_data <- torch::torch_tensor(m, dtype = torch::torch_float(), device = device)
  }
  if (length(dim(x_data)) != 2) stop("data must be two dimensional")

  n <- x_data$size(1)
  data_dim <- x_data$size(2)
  if (batch_size > n) batch_size <- n

  if (is.null(vector_field)) {
    vector_field <- FlowVectorField(data_dim = data_dim)
  }
  vector_field$to(device = device)

  # Per-column loss calibration.
  weights <- if (isTRUE(per_type_weight)) {
    flow_loss_weights(output_info, data_dim)
  } else {
    rep(1, data_dim)
  }
  w_t <- torch::torch_tensor(matrix(weights, nrow = 1),
                             dtype = torch::torch_float(), device = device)

  optim <- torch::optim_adam(vector_field$parameters, lr = base_lr,
                             weight_decay = weight_decay)

  params <- vector_field$parameters
  use_ema <- ema_decay > 0
  ema_shadow <- if (use_ema) {
    lapply(params, function(p) p$detach()$clone())
  } else NULL

  n_batches <- max(1L, as.integer(floor(n / batch_size)))
  losses <- if (isTRUE(track_loss)) numeric(epochs) else NULL
  step <- 0L
  report_every <- max(1L, epochs %/% 10L)

  t_start <- Sys.time()
  for (epoch in seq_len(epochs)) {
    vector_field$train()

    if (lr_schedule == "cosine") {
      lr_now <- base_lr * 0.5 * (1 + cos(pi * (epoch - 1) / epochs))
      for (g in seq_along(optim$param_groups)) optim$param_groups[[g]]$lr <- lr_now
    }

    perm <- torch::torch_randperm(n, dtype = torch::torch_long(),
                                  device = device) + 1L
    epoch_loss <- 0
    for (b in seq_len(n_batches)) {
      idx <- perm[((b - 1L) * batch_size + 1L):(b * batch_size)]
      x1 <- x_data[idx, ]
      x0 <- torch::torch_randn(c(x1$size(1), data_dim), device = device)
      tt <- flow_sample_time(x1$size(1), time_sampling, device = device)

      pth <- flow_path(x0, x1, tt, path = path, sigma_min = sigma_min)
      pred <- vector_field(pth$x_t, tt)
      loss <- torch::torch_mean(w_t * (pred - pth$u)^2)

      optim$zero_grad()
      loss$backward()
      optim$step()
      step <- step + 1L

      if (use_ema) {
        d <- min(ema_decay, (1 + step) / (10 + step))
        torch::with_no_grad({
          for (i in seq_along(params)) {
            ema_shadow[[i]]$mul_(d)$add_(params[[i]]$detach(), alpha = 1 - d)
          }
        })
      }

      epoch_loss <- epoch_loss + as.numeric(loss$item())
    }

    epoch_loss <- epoch_loss / n_batches
    if (isTRUE(track_loss)) losses[epoch] <- epoch_loss
    if (isTRUE(verbose) && (epoch %% report_every == 0 || epoch == epochs)) {
      message(sprintf("[flow] epoch %d/%d  loss %.5f  (%.1fs)",
                      epoch, epochs, epoch_loss,
                      as.numeric(difftime(Sys.time(), t_start, units = "secs"))))
    }
  }

  train_seconds <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

  # The sampling network carries the EMA weights; keep the raw one around for
  # diagnostics and for continued training.
  vector_field$eval()
  vf_ema <- vector_field
  if (use_ema) {
    vf_ema <- vector_field$clone(deep = TRUE)
    ema_params <- vf_ema$parameters
    torch::with_no_grad({
      for (i in seq_along(ema_params)) ema_params[[i]]$copy_(ema_shadow[[i]])
    })
    vf_ema$eval()
  }

  out <- list(
    vector_field = vf_ema,
    vector_field_raw = vector_field,
    settings = list(
      data_dim = data_dim,
      n_rows = n,
      epochs = epochs,
      batch_size = batch_size,
      n_batches = n_batches,
      steps = step,
      base_lr = base_lr,
      lr_schedule = lr_schedule,
      weight_decay = weight_decay,
      path = path,
      sigma_min = sigma_min,
      time_sampling = time_sampling,
      per_type_weight = isTRUE(per_type_weight),
      loss_weights = weights,
      ema_decay = ema_decay,
      output_info = output_info,
      # character, not a torch_device: see flow_device_label()
      device = device_label,
      seed = seed,
      train_seconds = train_seconds
    ),
    losses = losses
  )
  class(out) <- "trained_RFlow"
  out
}


#' @title Sample from a trained flow-matching model
#'
#' @description Integrates the learned ODE \eqn{dx/dt = v_\theta(x, t)} from
#'   `t = 0` (standard Gaussian noise) to `t = 1` (data) with a fixed step
#'   size, using either the explicit Euler method (one network evaluation per
#'   step) or Heun's second-order method (two evaluations per step, and
#'   markedly more accurate at the same wall time budget in practice).
#'
#' @param trained_flow An object of class `"trained_RFlow"` from
#'   [flow_matching_trainer()], or a bare `nn_module` vector field.
#' @param n_samples Integer, number of rows to generate. Ignored when `z` is
#'   supplied.
#' @param n_steps Integer, number of ODE steps.
#' @param solver `"euler"` or `"heun"`.
#' @param z Optional starting noise: a torch tensor or numeric matrix of shape
#'   `(n_samples, data_dim)`. Supplying `z` makes the map deterministic, which
#'   is what pairs \eqn{(z, \hat{x})} for Diffusion2GAN-style distillation
#'   require (see [flow_pairs()]).
#' @param return_trajectory Logical. If `TRUE`, also return the state after
#'   every step.
#' @param device Optional device override; defaults to the device the model was
#'   trained on.
#'
#' @return A numeric matrix of shape `(n_samples, data_dim)`, or, when
#'   `return_trajectory = TRUE`, a list with `x` (that matrix), `trajectory`
#'   (a list of `n_steps + 1` matrices) and `t` (the grid).
#'
#' @examples
#' \dontrun{
#' syn <- sample_flow(fm, n_samples = 1000, n_steps = 50, solver = "heun")
#' }
#' @export
sample_flow <- function(trained_flow, n_samples, n_steps = 50,
                        solver = c("euler", "heun"), z = NULL,
                        return_trajectory = FALSE, device = NULL) {
  solver <- match.arg(solver)
  n_steps <- as.integer(n_steps)
  if (is.na(n_steps) || n_steps < 1L) stop("n_steps must be a positive integer")

  if (inherits(trained_flow, "trained_RFlow")) {
    vf <- trained_flow$vector_field
    dev <- device %||% trained_flow$settings$device
    data_dim <- trained_flow$settings$data_dim
  } else if (inherits(trained_flow, "nn_module")) {
    vf <- trained_flow
    dev <- device %||% "cpu"
    data_dim <- vf$data_dim
  } else {
    stop("trained_flow must be a 'trained_RFlow' object or a torch nn_module")
  }
  dev <- flow_resolve_device(dev)
  vf$to(device = dev)
  vf$eval()

  if (is.null(z)) {
    if (missing(n_samples)) stop("give either n_samples or z")
    n_samples <- as.integer(n_samples)
    if (is.na(n_samples) || n_samples < 1L) stop("n_samples must be a positive integer")
    if (is.null(data_dim)) stop("cannot infer data_dim; supply z")
    x <- torch::torch_randn(c(n_samples, data_dim), device = dev)
  } else {
    if (!inherits(z, "torch_tensor")) {
      z <- torch::torch_tensor(as.matrix(z), dtype = torch::torch_float())
    }
    x <- z$to(device = dev, dtype = torch::torch_float())
    if (length(dim(x)) != 2) stop("z must be a (n, data_dim) matrix or tensor")
    n_samples <- x$size(1)
  }

  dt <- 1 / n_steps
  traj <- if (isTRUE(return_trajectory)) vector("list", n_steps + 1L) else NULL
  if (!is.null(traj)) traj[[1]] <- as.matrix(x$to(device = "cpu"))

  torch::with_no_grad({
    for (k in seq_len(n_steps)) {
      t_cur <- (k - 1) * dt
      tv <- torch::torch_full(c(n_samples), t_cur, device = dev,
                              dtype = torch::torch_float())
      v1 <- vf(x, tv)
      if (solver == "euler") {
        x <- x + dt * v1
      } else {
        x_e <- x + dt * v1
        tv2 <- torch::torch_full(c(n_samples), t_cur + dt, device = dev,
                                 dtype = torch::torch_float())
        v2 <- vf(x_e, tv2)
        x <- x + (dt / 2) * (v1 + v2)
      }
      if (!is.null(traj)) traj[[k + 1L]] <- as.matrix(x$to(device = "cpu"))
    }
  })

  out <- as.matrix(x$to(device = "cpu"))
  if (isTRUE(return_trajectory)) {
    list(x = out, trajectory = traj, t = seq(0, 1, length.out = n_steps + 1L))
  } else {
    out
  }
}


#' @title Noise-to-sample pairs from a flow-matching teacher
#'
#' @description Generates \eqn{(z, \hat{x})} pairs by integrating the teacher
#'   ODE from a fixed noise draw `z`. These are exactly the targets a
#'   Diffusion2GAN-style pair regression needs (Kang et al., ECCV 2024,
#'   arXiv:2405.05967): a one-step student `G` is then trained with
#'   `L_adv + lambda * ||G(z) - x_hat||^2` on the same `z`. Because the teacher
#'   ODE is deterministic, the pairs can be regenerated exactly from the same
#'   seed.
#'
#'   Pairs are produced in batches so that a large `n_pairs` does not need
#'   `n_pairs * data_dim * n_steps` of memory at once.
#'
#' @param trained_flow An object of class `"trained_RFlow"`.
#' @param n_pairs Integer, how many pairs to generate.
#' @param n_steps Integer, ODE steps per batch.
#' @param solver `"heun"` (default) or `"euler"`.
#' @param batch_size Integer, rows per integration batch.
#' @param seed Optional integer seed for the noise draw.
#' @param device Optional device override; defaults to the device the model
#'   was trained on (`trained_flow$settings$device`, stored as a character
#'   label so that it survives serialisation).
#'
#' @return A list with `z` and `x`, both numeric matrices of shape
#'   `(n_pairs, data_dim)`.
#'
#' @examples
#' \dontrun{
#' pairs <- flow_pairs(fm, n_pairs = 50000, n_steps = 50, seed = 1)
#' str(pairs)
#' }
#' @export
flow_pairs <- function(trained_flow, n_pairs, n_steps = 50, solver = "heun",
                       batch_size = 2000, seed = NULL, device = NULL) {
  if (!inherits(trained_flow, "trained_RFlow")) {
    stop("trained_flow must be an object of class 'trained_RFlow'")
  }
  n_pairs <- as.integer(n_pairs)
  batch_size <- as.integer(batch_size)
  if (is.na(n_pairs) || n_pairs < 1L) stop("n_pairs must be a positive integer")
  if (is.na(batch_size) || batch_size < 1L) stop("batch_size must be a positive integer")
  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }

  data_dim <- trained_flow$settings$data_dim
  dev <- flow_resolve_device(device %||% trained_flow$settings$device)

  z_out <- matrix(0, nrow = n_pairs, ncol = data_dim)
  x_out <- matrix(0, nrow = n_pairs, ncol = data_dim)

  done <- 0L
  while (done < n_pairs) {
    b <- min(batch_size, n_pairs - done)
    z <- torch::torch_randn(c(b, data_dim), device = dev)
    x <- sample_flow(trained_flow, n_samples = b, n_steps = n_steps,
                     solver = solver, z = z, device = dev)
    z_out[(done + 1L):(done + b), ] <- as.matrix(z$to(device = "cpu"))
    x_out[(done + 1L):(done + b), ] <- x
    done <- done + b
  }

  list(z = z_out, x = x_out)
}


#' @title Print a trained flow-matching model
#'
#' @description Print method for objects of class `"trained_RFlow"` returned by
#'   [flow_matching_trainer()].
#'
#' @param x An object of class `"trained_RFlow"`.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#' @export
print.trained_RFlow <- function(x, ...) {
  s <- x$settings
  cat("Trained flow-matching model (RGAN)\n")
  cat(sprintf("  data:          %d rows x %d columns\n", s$n_rows, s$data_dim))
  cat(sprintf("  path:          %s (sigma_min %g), time sampling %s\n",
              s$path, s$sigma_min, s$time_sampling))
  cat(sprintf("  training:      %d epochs x %d batches = %d steps, batch %d\n",
              s$epochs, s$n_batches, s$steps, s$batch_size))
  cat(sprintf("  optimiser:     Adam lr %g (%s), weight decay %g, EMA %g\n",
              s$base_lr, s$lr_schedule, s$weight_decay, s$ema_decay))
  cat(sprintf("  loss weights:  %s\n",
              if (isTRUE(s$per_type_weight) && !is.null(s$output_info))
                "per output_info block (CDTD-style calibration)" else "uniform"))
  if (!is.null(s$train_seconds)) {
    cat(sprintf("  train time:    %.1f seconds\n", s$train_seconds))
  }
  if (!is.null(x$losses) && length(x$losses)) {
    cat(sprintf("  loss:          first %.5f, last %.5f, min %.5f\n",
                x$losses[1], x$losses[length(x$losses)], min(x$losses)))
  }
  cat("  sample with:   sample_flow(fit, n_samples, n_steps = 50, solver = \"heun\")\n")
  invisible(x)
}


#' @title Save a trained flow-matching model
#'
#' @description Writes a `"trained_RFlow"` object to disk in a form that
#'   reloads correctly.
#'
#'   **Why a helper is needed.** A `torch` `nn_module` holds its parameters in
#'   external pointers, and as of torch 0.17 those pointers are *not*
#'   re-created by `readRDS()`: a model written with `saveRDS()` reloads with
#'   the right class but every tensor access fails with "external pointer is
#'   not valid". `save_flow()` therefore writes the network with
#'   [torch::torch_save()] and only the plain-R parts (settings, loss curve)
#'   with `saveRDS()`. The EMA weights are preserved, because the EMA network
#'   *is* `trained_flow$vector_field` after training.
#'
#'   Three files are written, following the convention of [save_gan()]:
#'   \itemize{
#'     \item `<path>_vector_field.pt` -- the sampling network (EMA weights when
#'       `ema_decay > 0`),
#'     \item `<path>_vector_field_raw.pt` -- the final non-EMA network, when it
#'       differs from the above,
#'     \item `<path>_settings.rds` -- `settings` and `losses`.
#'   }
#'
#' @param trained_flow An object of class `"trained_RFlow"` from
#'   [flow_matching_trainer()].
#' @param path Path prefix for the files to write (no extension).
#'
#' @return `path`, invisibly.
#' @seealso [load_flow()], [save_gan()]
#'
#' @examples
#' \dontrun{
#' fm <- flow_matching_trainer(x, epochs = 50)
#' save_flow(fm, file.path(tempdir(), "teacher"))
#' fm2 <- load_flow(file.path(tempdir(), "teacher"))
#' }
#' @export
save_flow <- function(trained_flow, path) {
  if (!inherits(trained_flow, "trained_RFlow")) {
    stop("trained_flow must be an object of class 'trained_RFlow'")
  }
  if (!is.character(path) || length(path) != 1L || !nzchar(path)) {
    stop("path must be a single non-empty character string")
  }
  dir_ <- dirname(path)
  if (!dir.exists(dir_)) stop("directory does not exist: ", dir_)

  vf <- trained_flow$vector_field
  if (!inherits(vf, "nn_module")) stop("trained_flow$vector_field is not an nn_module")

  # Save on the CPU so that an MPS/CUDA-trained model reloads anywhere.
  vf_dev <- flow_device_label(tryCatch(vf$parameters[[1]]$device,
                                       error = function(e) "cpu"))
  vf$to(device = "cpu")
  on.exit(try(vf$to(device = flow_resolve_device(vf_dev)), silent = TRUE),
          add = TRUE)
  torch::torch_save(vf, paste0(path, "_vector_field.pt"))

  raw <- trained_flow$vector_field_raw
  has_raw <- inherits(raw, "nn_module") && !identical(raw, vf)
  if (has_raw) {
    raw$to(device = "cpu")
    torch::torch_save(raw, paste0(path, "_vector_field_raw.pt"))
  }

  settings <- trained_flow$settings
  # Defensive: older objects may still carry a torch_device external pointer.
  settings$device <- flow_device_label(settings$device) %||% "cpu"

  saveRDS(list(settings = settings,
               losses = trained_flow$losses,
               has_raw = has_raw,
               rgan_version = as.character(utils::packageVersion("RGAN"))),
          paste0(path, "_settings.rds"))

  invisible(path)
}


#' @title Load a trained flow-matching model
#'
#' @description Reads back what [save_flow()] wrote and returns a
#'   `"trained_RFlow"` object that behaves like the one
#'   [flow_matching_trainer()] returned, so [sample_flow()] and [flow_pairs()]
#'   work directly on it.
#'
#' @param path The path prefix given to [save_flow()].
#' @param device Device to place the network on. Defaults to `NULL`, which
#'   uses the device recorded at training time, falling back to the CPU when
#'   that device is not available on this machine.
#'
#' @return An object of class `"trained_RFlow"`.
#' @seealso [save_flow()]
#'
#' @examples
#' \dontrun{
#' fm2 <- load_flow(file.path(tempdir(), "teacher"))
#' syn <- sample_flow(fm2, n_samples = 100, n_steps = 20)
#' }
#' @export
load_flow <- function(path, device = NULL) {
  if (!is.character(path) || length(path) != 1L || !nzchar(path)) {
    stop("path must be a single non-empty character string")
  }
  vf_path <- paste0(path, "_vector_field.pt")
  set_path <- paste0(path, "_settings.rds")
  if (!file.exists(vf_path)) stop("vector field file not found: ", vf_path)
  if (!file.exists(set_path)) stop("settings file not found: ", set_path)

  meta <- readRDS(set_path)
  settings <- meta$settings
  dev <- flow_resolve_device(device %||% settings$device)
  settings$device <- flow_device_label(dev)

  vf <- torch::torch_load(vf_path)
  vf$to(device = dev)
  vf$eval()

  raw_path <- paste0(path, "_vector_field_raw.pt")
  raw <- if (isTRUE(meta$has_raw) && file.exists(raw_path)) {
    r <- torch::torch_load(raw_path)
    r$to(device = dev)
    r$eval()
    r
  } else vf

  out <- list(vector_field = vf,
              vector_field_raw = raw,
              settings = settings,
              losses = meta$losses)
  class(out) <- "trained_RFlow"
  out
}


# ---------------------------------------------------------------------------
# Diffusion2GAN-style pair-regression distillation
#
# Experiment 6 of the 2026 experiment plan, report item L5. The teacher's ODE
# is a deterministic map z -> x_hat; a one-step student G is trained with
#
#     L = L_adv + lambda * || G(z) - x_hat ||^2
#
# on exactly those pairs (Kang et al., ECCV 2024, arXiv:2405.05967). Only the
# regression term touches the pairs: the adversarial term keeps using fresh
# noise and the *real* data batch, so the discriminator never learns to call
# teacher output "real". That is the distinction between this and simply
# training on teacher samples.
# ---------------------------------------------------------------------------


#' @title Weighted regression loss between a generated batch and its targets
#'
#' @description Internal helper implementing the pair-regression term of
#'   [gan_trainer()]'s `distillation` argument. The per-column residual is
#'   weighted by `weights` (the CDTD-style per-block calibration of
#'   [flow_loss_weights()], so that a wide one-hot block does not outweigh a
#'   single numeric column) and averaged over all cells.
#'
#' @param fake A torch tensor of shape `(batch, data_dim)`, the generator
#'   output.
#' @param target A torch tensor of the same shape, the teacher's `x_hat`.
#' @param weights Optional torch tensor of shape `(1, data_dim)` with the
#'   per-column weights, or NULL for uniform weights.
#' @param loss `"mse"` (default) or `"huber"` (smooth L1, less sensitive to
#'   the occasional badly integrated pair).
#' @param delta Numeric, the Huber transition point. Defaults to 1.
#'
#' @return A scalar torch tensor. It is exactly zero when `fake == target`.
#' @keywords internal
distill_regression_loss <- function(fake, target, weights = NULL,
                                    loss = c("mse", "huber"), delta = 1) {
  loss <- match.arg(loss)
  if (!identical(fake$shape, target$shape)) {
    stop(sprintf(
      "distillation: generator output (%s) and pair targets (%s) must have the same shape",
      paste(fake$shape, collapse = " x "), paste(target$shape, collapse = " x ")))
  }
  r <- fake - target
  per_cell <- if (identical(loss, "huber")) {
    a <- torch::torch_abs(r)
    torch::torch_where(a < delta, 0.5 * r^2, delta * (a - 0.5 * delta))
  } else {
    r^2
  }
  if (!is.null(weights)) per_cell <- weights * per_cell
  torch::torch_mean(per_cell)
}


#' @title Resolve the `distillation` argument of `gan_trainer()`
#'
#' @description Internal helper. Fills in the defaults of the `distillation`
#'   list, validates it, and -- when only a `teacher` is given -- generates the
#'   `(z, x_hat)` pairs with [flow_pairs()].
#'
#' @param distillation The user's `distillation` argument (NULL, FALSE, a
#'   `"trained_RFlow"` object, or a list).
#' @param noise_dim The generator's noise dimension.
#' @param data_dim The number of columns of the transformed data.
#' @param n_rows Number of training rows; the default `n_pairs` is `4 * n_rows`.
#' @param output_info Optional `output_info`, used for the per-type weights.
#' @param device The training device.
#'
#' @return NULL, or a resolved list with the fields documented in
#'   [gan_trainer()] plus `z_t` / `x_t` (torch tensors on `device`),
#'   `weights` and `n_pairs`.
#' @keywords internal
resolve_distillation <- function(distillation, noise_dim, data_dim,
                                 n_rows, output_info = NULL, device = "cpu") {
  if (is.null(distillation) || isFALSE(distillation)) return(NULL)
  if (inherits(distillation, "trained_RFlow")) {
    distillation <- list(teacher = distillation)
  }
  if (!is.list(distillation)) {
    stop("distillation must be NULL, a 'trained_RFlow' teacher, or a list")
  }
  get1 <- function(name, default) {
    v <- distillation[[name]]
    if (is.null(v)) default else v
  }

  pairs <- distillation$pairs
  teacher <- distillation$teacher
  if (is.null(pairs) && is.null(teacher)) {
    stop("distillation needs either `pairs` (from flow_pairs()) or `teacher` (a trained_RFlow)")
  }
  if (!is.null(teacher) && !inherits(teacher, "trained_RFlow")) {
    stop("distillation$teacher must be an object of class 'trained_RFlow'")
  }

  lambda <- as.numeric(get1("lambda", 10))
  if (length(lambda) != 1 || !is.finite(lambda) || lambda < 0) {
    stop("distillation$lambda must be a single non-negative number")
  }
  loss <- match.arg(as.character(get1("loss", "mse"))[1], c("mse", "huber"))
  huber_delta <- as.numeric(get1("huber_delta", 1))
  adversarial <- isTRUE(get1("adversarial", TRUE))
  noise_match <- isTRUE(get1("noise_match", TRUE))
  per_type_weight <- isTRUE(get1("per_type_weight", TRUE))
  n_pairs <- get1("n_pairs", NULL)
  n_steps <- as.integer(get1("n_steps", 50))
  solver <- match.arg(as.character(get1("solver", "heun"))[1], c("heun", "euler"))
  refresh_every <- get1("refresh_every", NULL)
  pair_seed <- get1("seed", NULL)
  if (!is.null(refresh_every)) {
    refresh_every <- as.integer(refresh_every)
    if (is.na(refresh_every) || refresh_every < 1L) {
      stop("distillation$refresh_every must be a positive integer or NULL")
    }
    if (is.null(teacher)) {
      stop("distillation$refresh_every needs a `teacher` to regenerate pairs from")
    }
  }

  if (is.null(pairs)) {
    n_pairs <- as.integer(n_pairs %||% (4L * as.integer(n_rows)))
    pairs <- flow_pairs(teacher, n_pairs = n_pairs, n_steps = n_steps,
                        solver = solver, seed = pair_seed)
  }
  pairs <- validate_flow_pairs(pairs, data_dim)
  n_pairs <- nrow(pairs$z)

  z_dim <- ncol(pairs$z)
  if (noise_match && z_dim != noise_dim) {
    stop(sprintf(
      paste0("distillation with noise_match = TRUE feeds the teacher's own noise z to the ",
             "generator, so noise_dim must equal the pair noise dimension: ",
             "noise_dim = %d but ncol(pairs$z) = %d. Set noise_dim = ncol(pairs$z) ",
             "(= %d, the teacher's data_dim), or set noise_match = FALSE."),
      as.integer(noise_dim), z_dim, z_dim))
  }

  weights <- if (per_type_weight) flow_loss_weights(output_info, data_dim) else rep(1, data_dim)

  list(
    pairs = pairs,
    teacher = teacher,
    lambda = lambda,
    loss = loss,
    huber_delta = huber_delta,
    adversarial = adversarial,
    noise_match = noise_match,
    per_type_weight = per_type_weight,
    n_pairs = n_pairs,
    n_steps = n_steps,
    solver = solver,
    refresh_every = refresh_every,
    seed = pair_seed,
    weights = weights,
    z_t = torch::torch_tensor(pairs$z, dtype = torch::torch_float())$to(device = device),
    x_t = torch::torch_tensor(pairs$x, dtype = torch::torch_float())$to(device = device),
    weight_t = torch::torch_tensor(matrix(weights, nrow = 1),
                                   dtype = torch::torch_float())$to(device = device)
  )
}


#' Validate a `(z, x)` pair list.
#' @param pairs A list with `z` and `x`.
#' @param data_dim Expected number of columns of `x`.
#' @return The pairs, with `z` and `x` coerced to numeric matrices.
#' @keywords internal
validate_flow_pairs <- function(pairs, data_dim) {
  if (!is.list(pairs) || is.null(pairs$z) || is.null(pairs$x)) {
    stop("distillation$pairs must be a list with elements `z` and `x` (see flow_pairs())")
  }
  z <- as.matrix(pairs$z); storage.mode(z) <- "double"
  x <- as.matrix(pairs$x); storage.mode(x) <- "double"
  if (nrow(z) != nrow(x)) {
    stop(sprintf("distillation$pairs: z has %d rows but x has %d", nrow(z), nrow(x)))
  }
  if (nrow(z) < 1L) stop("distillation$pairs must have at least one pair")
  if (ncol(x) != data_dim) {
    stop(sprintf(
      "distillation$pairs: x has %d columns but the training data has %d",
      ncol(x), as.integer(data_dim)))
  }
  if (anyNA(z) || anyNA(x) || any(!is.finite(z)) || any(!is.finite(x))) {
    stop("distillation$pairs contains NA/NaN/Inf")
  }
  list(z = z, x = x)
}


#' @title Distil a flow-matching teacher into a one-step GAN generator
#'
#' @description Convenience wrapper around [flow_pairs()] and [gan_trainer()]
#'   implementing the Diffusion2GAN recipe (Kang et al., ECCV 2024,
#'   arXiv:2405.05967) for tabular data: the teacher ODE is run from a fixed
#'   noise draw `z` to produce targets `x_hat`, and the GAN generator is
#'   trained with `L_adv + lambda * ||G(z) - x_hat||^2` on those pairs. The
#'   student samples in a single forward pass, at GAN cost, while the teacher
#'   needs `n_steps` (times two for Heun) network evaluations per row.
#'
#'   The generator's noise dimension is set to the teacher's data dimension so
#'   that the teacher's own `z` can be fed to the student (`noise_match`), and
#'   the pairs are passed to `gan_trainer()` through its `distillation`
#'   argument. Everything else -- value function, regularization, `pac`,
#'   `conditioning`, budgets -- is forwarded through `...`.
#'
#' @param teacher A trained teacher of class `"trained_RFlow"` from
#'   [flow_matching_trainer()].
#' @param data The student's training data: a numeric matrix / tensor already
#'   in the teacher's representation, or a `data.frame` together with a
#'   fitted `transformer`.
#' @param transformer Optional fitted [data_transformer]. When given and
#'   `data` is a `data.frame`, `data` is transformed with it, and its
#'   `output_info` is used unless `output_info` is passed explicitly.
#' @param n_pairs Number of `(z, x_hat)` pairs to generate. Defaults to
#'   `4 * nrow(data)`.
#' @param n_steps Integer, ODE steps used to build the pairs. Defaults to 50.
#' @param lambda Weight of the regression term. Defaults to 10.
#' @param solver `"heun"` (default) or `"euler"`.
#' @param pair_seed Optional seed for the pair generation.
#' @param distillation Optional extra fields merged into the `distillation`
#'   list passed to [gan_trainer()] (e.g. `list(loss = "huber")`,
#'   `list(adversarial = FALSE)`).
#' @param ... Further arguments passed to [gan_trainer()]. `noise_dim` is set
#'   from the pairs and must not be supplied.
#'
#' @return An object of class `"trained_RGAN"`, as returned by
#'   [gan_trainer()], whose `settings$distillation` records the recipe.
#'
#' @examples
#' \dontrun{
#' fm <- flow_matching_trainer(X, output_info = tr$output_info, epochs = 100)
#' student <- distill_to_gan(fm, X, output_info = tr$output_info,
#'                           lambda = 10, epochs = 50, batch_size = 500,
#'                           value_function = "rpgan",
#'                           regularization = list(type = "r1r2", gamma = 10))
#' syn <- sample_synthetic_data(student, tr)
#' }
#' @export
distill_to_gan <- function(teacher, data, transformer = NULL, n_pairs = NULL,
                           n_steps = 50, lambda = 10, solver = "heun",
                           pair_seed = NULL, distillation = list(), ...) {
  if (!inherits(teacher, "trained_RFlow")) {
    stop("teacher must be an object of class 'trained_RFlow'")
  }
  dots <- list(...)
  if (!is.null(dots$noise_dim)) {
    stop("distill_to_gan() sets noise_dim from the teacher's data_dim; do not pass noise_dim")
  }
  if (!is.list(distillation)) stop("distillation must be a list")

  if (is.data.frame(data)) {
    if (is.null(transformer)) {
      stop("a data.frame `data` needs a fitted `transformer`")
    }
    data <- transformer$transform(data)
  }
  if (!is.null(transformer) && is.null(dots$output_info)) {
    dots$output_info <- transformer$output_info
  }

  n_rows <- if (inherits(data, "torch_tensor")) data$size(1) else nrow(data)
  n_pairs <- as.integer(n_pairs %||% (4L * as.integer(n_rows)))
  pairs <- flow_pairs(teacher, n_pairs = n_pairs, n_steps = n_steps,
                      solver = solver, seed = pair_seed)

  distillation$pairs <- pairs
  distillation$lambda <- lambda
  distillation$n_steps <- n_steps
  distillation$solver <- solver
  if (is.null(distillation$teacher)) distillation$teacher <- teacher

  do.call(gan_trainer,
          c(list(data = data, noise_dim = ncol(pairs$z),
                 distillation = distillation), dots))
}
