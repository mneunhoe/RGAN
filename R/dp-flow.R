#' @title Per-Record Clipped Gradient Accumulation for DP-SGD
#'
#' @description Accumulate the clipped per-record gradients of a network into a
#'   running sum, and return the sum together with the realised clipping
#'   coefficient.
#'
#' @details For every record the caller has already run a backward pass, so the
#'   parameter gradients in \code{params} hold \eqn{\nabla_\theta L_i}. The
#'   global L2 norm over all parameter tensors is formed, the record's gradient
#'   is scaled by \eqn{\min(1, C / (\|g_i\|_2 + \epsilon))}, and the result is
#'   added to the accumulator.
#'
#'   The clipping coefficient is returned so that a caller can report the
#'   clipping rate; that rate is a function of the private data and must not be
#'   released.
#'
#' @param acc The accumulator from a previous call, or \code{NULL} to start.
#' @param params List of parameter tensors whose \code{grad} fields are set.
#' @param max_grad_norm The clipping threshold \eqn{C}.
#'
#' @return A list with \code{acc} (the updated sum) and \code{coef} (the
#'   clipping coefficient applied to this record).
#'
#' @keywords internal
dp_accumulate_clipped <- function(acc, params, max_grad_norm) {
  grads <- vector("list", length(params))
  sq <- NULL
  for (j in seq_along(params)) {
    g <- params[[j]]$grad
    if (is.null(g)) g <- torch::torch_zeros_like(params[[j]])
    grads[[j]] <- g
    s <- torch::torch_sum(g * g)
    sq <- if (is.null(sq)) s else sq + s
  }
  # One host synchronisation per record rather than one per parameter.
  norm <- sqrt(as.numeric(sq$item()))
  coef <- min(1, max_grad_norm / (norm + 1e-8))

  if (is.null(acc)) {
    acc <- lapply(grads, function(g) g * coef)
  } else {
    for (j in seq_along(acc)) acc[[j]] <- acc[[j]] + grads[[j]] * coef
  }
  list(acc = acc, coef = coef)
}

#' @title Release a DP-SGD Update from a Clipped Gradient Sum
#'
#' @description Divide the clipped gradient sum by the constant divisor, add
#'   Gaussian noise calibrated to the clipping threshold, install the result as
#'   the parameter gradients, and take an optimiser step.
#'
#' @details The divisor is \code{sampling_rate * n}, the \emph{expected} batch
#'   size. It must not depend on the realised Poisson draw: the accountant
#'   covers the noisy sum, and dividing by a constant is post-processing, while
#'   dividing by the realised draw would make both the location and the scale of
#'   the released quantity a function of a data-dependent count.
#'
#'   \code{acc == NULL} is the empty-draw case: the clipped sum is zero and the
#'   mechanism releases pure noise. Returning early instead would make "no
#'   optimiser step" an observable event with data-dependent probability.
#'
#' @param acc The clipped gradient sum, or \code{NULL} for an empty draw.
#' @param params List of parameter tensors.
#' @param optim The optimiser.
#' @param max_grad_norm Clipping threshold \eqn{C}.
#' @param noise_multiplier Noise multiplier \eqn{\sigma}.
#' @param expected_batch_size The constant divisor \eqn{q n}.
#' @param secure_rng Use OpenDP's cryptographically secure Gaussian sampler.
#'
#' @return The L2 norm of the released (noisy, divided) gradient. This is a
#'   function of the release alone, hence post-processing, hence free to keep.
#'
#' @keywords internal
dp_release_update <- function(acc, params, optim, max_grad_norm,
                              noise_multiplier, expected_batch_size,
                              secure_rng = FALSE) {
  noise_scale <- noise_multiplier * max_grad_norm / expected_batch_size
  optim$zero_grad()

  released_sq <- NULL
  for (j in seq_along(params)) {
    mean_grad <- if (is.null(acc)) {
      torch::torch_zeros_like(params[[j]])
    } else {
      acc[[j]] / expected_batch_size
    }
    noise <- if (secure_rng) {
      sample_secure_gaussian_like(mean_grad, noise_scale)
    } else {
      torch::torch_randn_like(mean_grad) * noise_scale
    }
    released <- mean_grad + noise
    params[[j]]$grad <- released
    s <- torch::torch_sum(released * released)
    released_sq <- if (is.null(released_sq)) s else released_sq + s
  }
  optim$step()
  sqrt(as.numeric(released_sq$item()))
}

#' @title Profile the Per-Record Gradient Norm of a Vector Field
#'
#' @description Draw per-record flow-matching losses, back-propagate each one on
#'   its own, and return the distribution of the resulting gradient norms.
#'
#' @details This exists to set the clipping threshold \eqn{C} without touching
#'   the private data. Run it on a \emph{synthetic} pilot table — for instance
#'   rows drawn from an already-published model — and use a low quantile of the
#'   returned norms. Measuring the same quantity on real rows would itself be a
#'   release and would have to be paid for.
#'
#' @param data A numeric matrix or torch tensor of transformed rows.
#' @param vector_field A \code{torch::nn_module} with signature
#'   \code{forward(x_t, t)}. Defaults to an untrained \code{FlowVectorField}.
#' @param output_info Optional block description for the per-column weights.
#' @param path,sigma_min,time_sampling Path and time controls matching the
#'   trainer that will be run.
#' @param n_records How many records to draw.
#' @param device Torch device.
#' @param seed Optional seed.
#'
#' @return A numeric vector of per-record gradient norms.
#'
#' @keywords internal
dp_flow_grad_norm_profile <- function(data, vector_field = NULL, output_info = NULL,
                                     path = "ot", sigma_min = 1e-4,
                                     time_sampling = "uniform",
                                     noise_multiplicity = 1L,
                                     n_records = 200, device = "cpu",
                                     seed = NULL) {
  noise_multiplicity <- as.integer(noise_multiplicity)
  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }
  device <- if (inherits(device, "torch_device")) device else torch::torch_device(device)
  if (inherits(data, "torch_tensor")) {
    x_data <- data$to(device = device, dtype = torch::torch_float())
  } else {
    m <- as.matrix(data)
    storage.mode(m) <- "double"
    x_data <- torch::torch_tensor(m, dtype = torch::torch_float(), device = device)
  }
  data_dim <- x_data$size(2)
  n <- x_data$size(1)
  if (is.null(vector_field)) vector_field <- FlowVectorField(data_dim = data_dim)
  vector_field$to(device = device)
  vector_field$eval()

  weights <- flow_loss_weights(output_info, data_dim)
  w_t <- torch::torch_tensor(matrix(weights, nrow = 1),
                             dtype = torch::torch_float(), device = device)

  params <- vector_field$parameters
  norms <- numeric(n_records)
  for (i in seq_len(n_records)) {
    sel <- sample.int(n, 1L)
    vector_field$zero_grad()
    x1 <- x_data[sel:sel, ]
    loss_i <- NULL
    for (k in seq_len(noise_multiplicity)) {
      x0 <- torch::torch_randn(c(1L, data_dim), device = device)
      tt <- flow_sample_time(1L, time_sampling, device = device)
      pth <- flow_path(x0, x1, tt, path = path, sigma_min = sigma_min)
      pred <- vector_field(pth$x_t, tt)
      term <- torch::torch_mean(w_t * (pred - pth$u)^2) / noise_multiplicity
      loss_i <- if (is.null(loss_i)) term else loss_i + term
    }
    loss_i$backward()
    sq <- NULL
    for (j in seq_along(params)) {
      g <- params[[j]]$grad
      if (is.null(g)) g <- torch::torch_zeros_like(params[[j]])
      s <- torch::torch_sum(g * g)
      sq <- if (is.null(sq)) s else sq + s
    }
    norms[i] <- sqrt(as.numeric(sq$item()))
  }
  norms
}

#' @title Train a Flow-Matching Model under Differential Privacy
#'
#' @description Differentially private counterpart of
#'   [flow_matching_trainer()]. The objective is unchanged: each record draws
#'   its own noise point \eqn{x_0}, its own time \eqn{t}, and contributes its
#'   own squared error against the conditional velocity. Because the batch loss
#'   is the mean of those per-record terms, DP-SGD applies to it verbatim.
#'
#' @details
#' The mechanism is the sampled Gaussian mechanism, exactly as implemented for
#' the discriminator in \code{dp_discriminator_step()} and certified by
#' \code{rdp_sgm()}:
#' \enumerate{
#'   \item each step draws a \strong{Poisson} batch of rate \eqn{q = B / n},
#'     by independent per-record inclusion — not a permutation;
#'   \item each record's gradient \eqn{\nabla_\theta L_i} is clipped in L2 norm
#'     to \eqn{C};
#'   \item the clipped gradients are summed, Gaussian noise
#'     \eqn{N(0, \sigma^2 C^2)} is added per parameter, and the result is divided
#'     by the constant \eqn{q n};
#'   \item an empty Poisson draw still releases noise, so that "no step" is not
#'     an observable event with data-dependent probability.
#' }
#' The release is then composed over \eqn{T} steps by the RDP accountant and
#' converted to \eqn{(\epsilon, \delta)} by the CKS bound, all of which lives in
#' \code{dp_accountant_poisson}.
#'
#' Four properties of the non-private trainer carry over, and one does not.
#' \itemize{
#'   \item \strong{The EMA is free.} The shadow weights are a deterministic
#'     function of the sequence of released parameter values, so averaging them
#'     is post-processing. The model that is sampled from is the EMA model.
#'   \item \strong{Adam is free.} The optimiser step is a deterministic function
#'     of the sequence of released noisy gradients. Momentum and adaptive
#'     scaling change the effective noise per update, which is a tuning
#'     question, not a privacy one.
#'   \item \strong{The cosine schedule is free}, because it is fixed by the
#'     (public) step count.
#'   \item \strong{Sampling is free}, because it reads only the released vector
#'     field.
#'   \item \strong{The loss trace is not free.} A per-step mean loss is a
#'     noiseless function of the private batch — and this is true whatever the
#'     divisor: the constant divisor would make it a scaled sum, not a private
#'     quantity. There is therefore no \code{losses} field. What is recorded
#'     instead, when \code{track_loss} is set, is the norm of the
#'     \emph{released} noisy gradient, which is a function of the release alone
#'     and costs no budget. It is a perfectly good divergence diagnostic.
#' }
#'
#' \strong{Noise multiplicity} (\code{noise_multiplicity = K}) draws \eqn{K}
#' independent \eqn{(x_0, t)} pairs for each record and averages their loss
#' terms \emph{before} clipping. This is the technique of Dockhorn et al.
#' (TMLR 2023) for DP diffusion models. It is free under the accountant: the
#' averaged quantity still depends on record \eqn{i} alone and on nothing else
#' in the batch, so the sensitivity argument is unchanged and \eqn{\epsilon} is
#' set by \eqn{(q, \sigma, T)} exactly as before. It reduces the variance of the
#' per-record gradient estimate by a factor of \eqn{K} at a factor of \eqn{K}
#' in compute, so it trades time for the sampling component of the gradient
#' noise. It does \emph{not} reduce the DP noise itself, which is usually the
#' dominant term; expect a gain only where the two are comparable. When using
#' \eqn{K > 1}, profile \eqn{C} with the same \eqn{K} — the averaged gradient
#' norms are smaller and less dispersed than the single-draw ones.
#'
#' @param data Numeric matrix or torch tensor of transformed training rows.
#' @param output_info Optional block description from the data transformer.
#' @param steps Total number of DP-SGD steps. When \code{NULL} it is derived
#'   from \code{epochs} as \code{epochs * ceiling(nrow(data) / batch_size)}.
#' @param epochs Number of passes; only used when \code{steps} is \code{NULL}.
#' @param batch_size The expected Poisson batch size \eqn{q n}.
#' @param base_lr Initial learning rate.
#' @param lr_schedule \code{"cosine"} or \code{"constant"}, over \code{steps}.
#' @param weight_decay Adam weight decay.
#' @param path \code{"ot"} or \code{"vp"}, as in [flow_matching_trainer()].
#' @param sigma_min Residual noise of the OT path at \eqn{t = 1}.
#' @param time_sampling \code{"uniform"} or \code{"logit_normal"}.
#' @param per_type_weight Use the per-column loss calibration.
#' @param ema_decay EMA decay for the sampled weights; \code{0} disables.
#' @param vector_field Optional pre-built \code{torch::nn_module}.
#' @param device Torch device.
#' @param seed Optional integer seed for R and torch.
#' @param verbose Print progress.
#' @param noise_multiplier The Gaussian noise multiplier \eqn{\sigma}. When
#'   \code{NULL} it is calibrated so that the run lands at \code{target_epsilon}.
#' @param target_epsilon The target \eqn{\epsilon}. Ignored when
#'   \code{noise_multiplier} is supplied.
#' @param target_delta The target \eqn{\delta}.
#' @param max_grad_norm The clipping threshold \eqn{C}. Set it from
#'   \code{dp_flow_grad_norm_profile()} on a synthetic pilot.
#' @param noise_multiplicity How many independent \eqn{(x_0, t)} draws to
#'   average inside each record's loss. See the details section.
#' @param secure_rng Use OpenDP's cryptographically secure RNG. Far slower;
#'   reserve it for a final confirmation run.
#' @param track_loss Record the norm of the released gradient per step.
#'
#' @return An object of class \code{"trained_RFlow"}, with the fields of
#'   [flow_matching_trainer()] plus \code{privacy} (the realised
#'   \eqn{(\epsilon, \delta)} and its ingredients) and, in place of a loss
#'   trace, \code{settings$released_grad_norm}.
#'
#' @seealso [flow_matching_trainer()], [sample_flow()], [dp_gan_trainer()]
#'
#' @examples
#' \dontrun{
#' x <- scale(RGAN::sample_toydata(n = 2000))
#' syn_pilot <- x[sample(nrow(x), 200), ]
#' C <- stats::quantile(
#'   RGAN:::dp_flow_grad_norm_profile(syn_pilot, n_records = 100), 0.5)
#' fm <- dp_flow_matching_trainer(x, steps = 500, batch_size = 100,
#'                                target_epsilon = 4, max_grad_norm = C)
#' sample_flow(fm, n_samples = 200, n_steps = 50)
#' }
#' @export
dp_flow_matching_trainer <- function(data,
                                     output_info = NULL,
                                     steps = NULL,
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
                                     verbose = TRUE,
                                     noise_multiplier = NULL,
                                     target_epsilon = NULL,
                                     target_delta = 1e-5,
                                     max_grad_norm = 1,
                                     noise_multiplicity = 1L,
                                     secure_rng = FALSE,
                                     track_loss = TRUE) {
  lr_schedule <- match.arg(lr_schedule)
  path <- match.arg(path)
  time_sampling <- match.arg(time_sampling)

  batch_size <- as.integer(batch_size)
  noise_multiplicity <- as.integer(noise_multiplicity)
  if (is.na(noise_multiplicity) || noise_multiplicity < 1L) {
    stop("noise_multiplicity must be a positive integer")
  }
  if (is.na(batch_size) || batch_size < 1L) stop("batch_size must be a positive integer")
  if (!is.numeric(max_grad_norm) || max_grad_norm <= 0) {
    stop("max_grad_norm must be positive")
  }
  if (!is.numeric(target_delta) || target_delta <= 0 || target_delta >= 1) {
    stop("target_delta must be in (0, 1)")
  }

  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }

  device_label <- flow_device_label(device)
  device <- if (inherits(device, "torch_device")) device else torch::torch_device(device)

  if (inherits(data, "torch_tensor")) {
    x_data <- data$to(device = device, dtype = torch::torch_float())
  } else {
    m <- as.matrix(data)
    storage.mode(m) <- "double"
    if (anyNA(m) || any(!is.finite(m))) {
      stop("data contains NA/NaN/Inf; the DP flow trainer needs a finite numeric matrix")
    }
    x_data <- torch::torch_tensor(m, dtype = torch::torch_float(), device = device)
  }
  if (length(dim(x_data)) != 2) stop("data must be two dimensional")

  n <- x_data$size(1)
  data_dim <- x_data$size(2)
  if (batch_size > n) {
    stop("batch_size (", batch_size, ") exceeds the number of rows (", n,
         "); the sampling rate would exceed 1 and the accountant's ",
         "amplification bound would not apply.", call. = FALSE)
  }
  sampling_rate <- batch_size / n

  if (is.null(steps)) {
    epochs <- as.integer(epochs)
    if (is.na(epochs) || epochs < 1L) stop("epochs must be a positive integer")
    steps <- as.integer(epochs) * as.integer(ceiling(n / batch_size))
  }
  steps <- as.integer(steps)
  if (is.na(steps) || steps < 1L) stop("steps must be a positive integer")

  # Calibrate the noise multiplier against the *planned* step count, so that the
  # accountant knows T in advance and no step is taken outside the budget.
  calibrated <- is.null(noise_multiplier)
  if (calibrated) {
    if (is.null(target_epsilon)) {
      stop("supply either noise_multiplier or target_epsilon")
    }
    noise_multiplier <- calibrate_noise_multiplier(
      target_epsilon = target_epsilon, target_delta = target_delta,
      sampling_rate = sampling_rate, total_steps = steps)
  }
  accountant <- dp_accountant_poisson$new(
    sampling_rate = sampling_rate, noise_multiplier = noise_multiplier,
    target_delta = target_delta)
  accountant$step(steps)
  epsilon <- accountant$get_epsilon()

  if (is.null(vector_field)) vector_field <- FlowVectorField(data_dim = data_dim)
  vector_field$to(device = device)

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
  ema_shadow <- if (use_ema) lapply(params, function(p) p$detach()$clone()) else NULL

  expected_batch_size <- sampling_rate * n
  released_norm <- if (isTRUE(track_loss)) numeric(steps) else NULL

  if (isTRUE(verbose)) {
    message(sprintf(
      "[dp-flow] n=%d dim=%d q=%.4g sigma=%.4g C=%.4g K=%d steps=%d -> eps=%.3f (delta=%.1e)",
      n, data_dim, sampling_rate, noise_multiplier, max_grad_norm,
      noise_multiplicity, steps, epsilon, target_delta))
  }

  t_start <- Sys.time()
  report_every <- max(1L, steps %/% 10L)
  vector_field$train()

  for (step in seq_len(steps)) {
    if (lr_schedule == "cosine") {
      lr_now <- base_lr * 0.5 * (1 + cos(pi * (step - 1) / steps))
      for (g in seq_along(optim$param_groups)) optim$param_groups[[g]]$lr <- lr_now
    }

    idx <- if (secure_rng) {
      secure_poisson_subsample(n, sampling_rate)
    } else {
      fast_poisson_subsample(n, sampling_rate)
    }

    acc <- NULL
    if (length(idx) > 0L) {
      for (sel in idx) {
        optim$zero_grad()
        x1 <- x_data[sel:sel, ]
        # K independent (x0, t) draws, averaged before clipping. The average is
        # still a per-record quantity, so the sensitivity argument is unchanged.
        loss_i <- NULL
        for (k in seq_len(noise_multiplicity)) {
          x0 <- torch::torch_randn(c(1L, data_dim), device = device)
          tt <- flow_sample_time(1L, time_sampling, device = device)
          pth <- flow_path(x0, x1, tt, path = path, sigma_min = sigma_min)
          pred <- vector_field(pth$x_t, tt)
          term <- torch::torch_mean(w_t * (pred - pth$u)^2) / noise_multiplicity
          loss_i <- if (is.null(loss_i)) term else loss_i + term
        }
        loss_i$backward()
        acc <- dp_accumulate_clipped(acc, params, max_grad_norm)$acc
      }
    }

    norm <- dp_release_update(acc, params, optim, max_grad_norm,
                              noise_multiplier, expected_batch_size, secure_rng)
    if (!is.null(released_norm)) released_norm[step] <- norm

    if (use_ema) {
      d <- min(ema_decay, (1 + step) / (10 + step))
      torch::with_no_grad({
        for (i in seq_along(params)) {
          ema_shadow[[i]]$mul_(d)$add_(params[[i]]$detach(), alpha = 1 - d)
        }
      })
    }

    if (isTRUE(verbose) && (step %% report_every == 0 || step == steps)) {
      message(sprintf("[dp-flow] step %d/%d  released-grad-norm %.5f  (%.1fs)",
                      step, steps, norm,
                      as.numeric(difftime(Sys.time(), t_start, units = "secs"))))
    }
  }

  train_seconds <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
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
    privacy = list(
      epsilon = epsilon,
      delta = target_delta,
      noise_multiplier = noise_multiplier,
      max_grad_norm = max_grad_norm,
      sampling_rate = sampling_rate,
      steps = steps,
      calibrated = calibrated,
      target_epsilon = target_epsilon
    ),
    settings = list(
      data_dim = data_dim,
      n_rows = n,
      steps = steps,
      batch_size = batch_size,
      sampling_rate = sampling_rate,
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
      device = device_label,
      seed = seed,
      train_seconds = train_seconds,
      noise_multiplicity = noise_multiplicity,
      secure_rng = secure_rng,
      released_grad_norm = released_norm
    ),
    # Deliberately empty: a per-step loss on real rows is not a private release.
    losses = NULL
  )
  class(out) <- "trained_RFlow"
  out
}
