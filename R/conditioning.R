#' @title Any-order conditioning helpers
#'
#' @description Internal helpers implementing Experiment 5 of the 2026
#'   experiment plan: CTGAN's single-column conditional vector with
#'   training-by-sampling (Xu et al., NeurIPS 2019, arXiv:1907.00503) and
#'   TabMT / TabularARGN style random-mask any-order conditioning (report item
#'   T3) with classifier-free-guidance style condition dropout (report item
#'   L7, as in TabDiff).
#'
#'   The condition of one row is the concatenation of
#'   \itemize{
#'     \item a 0/1 **mask** of length `n_blocks`, one entry per column block of
#'       the transformed matrix (the blocks of [output_info_blocks()], so a
#'       `"mode_specific"` column contributes a mode-indicator block and a
#'       value block), and
#'     \item the **masked values**, the row's transformed values with the
#'       columns of every unobserved block set to zero (length `data_dim`).
#'   }
#'   so the condition vector is `n_blocks + data_dim` wide. The generator takes
#'   `cat(z, condition)` and the discriminator takes `cat(row, condition)`
#'   per sample (before PacGAN packing).
#'
#' @name conditioning
#' @keywords internal
NULL


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

#' @title Resolve the conditioning settings
#'
#' @description Internal helper that fills in the defaults of the
#'   `conditioning` argument of [gan_trainer()] and validates it.
#'
#' @param conditioning NULL or a list, see [gan_trainer()].
#'
#' @return NULL, or a fully specified list with the fields `type`,
#'   `mask_prob`, `cond_dropout`, `log_frequency`, `resample_rare`,
#'   `resample_rare_frac` and `consistency_weight`.
#' @keywords internal
resolve_conditioning <- function(conditioning) {
  if (is.null(conditioning)) return(NULL)
  if (isFALSE(conditioning)) return(NULL)
  if (is.character(conditioning) && length(conditioning) == 1) {
    conditioning <- list(type = conditioning)
  }
  if (!is.list(conditioning)) {
    stop("conditioning must be NULL or a list")
  }
  get1 <- function(name, default) {
    v <- conditioning[[name]]
    if (is.null(v)) default else v
  }

  type <- as.character(get1("type", "random_mask"))[1]
  if (identical(type, "none")) return(NULL)
  valid_types <- c("single_column", "random_mask")
  if (!type %in% valid_types) {
    stop(sprintf("conditioning$type must be one of: %s",
                 paste(c(valid_types, "none"), collapse = ", ")))
  }

  mask_prob <- get1("mask_prob", "uniform")
  if (is.character(mask_prob)) {
    if (!identical(mask_prob[1], "uniform")) {
      stop("conditioning$mask_prob must be \"uniform\" or a number in [0, 1]")
    }
    mask_prob <- "uniform"
  } else {
    if (!is.numeric(mask_prob) || length(mask_prob) != 1 ||
        is.na(mask_prob) || mask_prob < 0 || mask_prob > 1) {
      stop("conditioning$mask_prob must be \"uniform\" or a number in [0, 1]")
    }
    mask_prob <- as.numeric(mask_prob)
  }

  cond_dropout <- get1("cond_dropout", 0.1)
  if (!is.numeric(cond_dropout) || length(cond_dropout) != 1 ||
      is.na(cond_dropout) || cond_dropout < 0 || cond_dropout > 1) {
    stop("conditioning$cond_dropout must be a single number in [0, 1]")
  }

  consistency_weight <- get1("consistency_weight", 1)
  if (!is.numeric(consistency_weight) || length(consistency_weight) != 1 ||
      is.na(consistency_weight) || consistency_weight < 0) {
    stop("conditioning$consistency_weight must be a single non-negative number")
  }

  # resample_rare is either a logical switch (TRUE means the default fraction
  # of 0.5) or directly the fraction of rows that are re-drawn by a
  # log-frequency category.
  resample_rare <- get1("resample_rare", FALSE)
  if (is.logical(resample_rare)) {
    resample_rare_frac <- if (isTRUE(resample_rare)) {
      get1("resample_rare_frac", 0.5)
    } else {
      0
    }
    resample_rare <- isTRUE(resample_rare)
  } else if (is.numeric(resample_rare) && length(resample_rare) == 1 &&
             !is.na(resample_rare) && resample_rare >= 0 && resample_rare <= 1) {
    resample_rare_frac <- as.numeric(resample_rare)
    resample_rare <- resample_rare_frac > 0
  } else {
    stop("conditioning$resample_rare must be TRUE/FALSE or a fraction in [0, 1]")
  }
  if (!is.numeric(resample_rare_frac) || length(resample_rare_frac) != 1 ||
      is.na(resample_rare_frac) || resample_rare_frac < 0 ||
      resample_rare_frac > 1) {
    stop("conditioning$resample_rare_frac must be a single number in [0, 1]")
  }

  list(
    type = type,
    mask_prob = mask_prob,
    cond_dropout = as.numeric(cond_dropout),
    log_frequency = isTRUE(get1("log_frequency", TRUE)),
    resample_rare = resample_rare,
    resample_rare_frac = resample_rare_frac,
    consistency_weight = as.numeric(consistency_weight)
  )
}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

#' @title Build the conditioning metadata of a training matrix
#'
#' @description Turns `output_info` and the transformed training matrix into
#'   everything the conditional training and the conditional sampling need:
#'   the column blocks, the mapping from block to original column, and for
#'   every categorical block the category of each training row, the category
#'   frequencies, the CTGAN log-frequency sampling probabilities and the row
#'   index per category used by training-by-sampling.
#'
#'   Categories are read off the transformed matrix, not the transformer, so
#'   the metadata is self-contained: `"softmax"` blocks are decoded by
#'   `which.max()` and `"cossin"` blocks by their angle.
#'
#' @param output_info The `output_info` list of a fitted `data_transformer`.
#' @param data A numeric matrix (rows x `data_dim`) of transformed data.
#'
#' @return A list with `n_blocks`, `data_dim`, `blocks`, `block_of_col`,
#'   `group_of_block`, `blocks_of_group`, `n_groups`, `cat_blocks` and
#'   `cat_block_ids`.
#' @keywords internal
build_conditioning_meta <- function(output_info, data) {
  if (is.null(output_info)) {
    stop("conditioning requires output_info (the transformer's output_info)")
  }
  data <- as.matrix(data)
  data_dim <- ncol(data)
  bl <- output_info_blocks(output_info, data_dim)
  if (!bl$ok) {
    stop("conditioning requires an output_info consistent with the data (sum of block dimensions must equal ncol(data))")
  }
  blocks <- bl$blocks

  # Which original column (element of output_info) does each block belong to?
  # A "mode_specific" element produces two blocks (mode indicators and value),
  # and they always share the same group, so that a condition observes a
  # column as a whole.
  group_of_block <- integer(length(blocks))
  g <- 0L
  pos <- 1L
  b <- 1L
  for (info in output_info) {
    g <- g + 1L
    dim <- as.integer(info[[1]])
    n_sub <- if (identical(as.character(info[[2]]), "mode_specific") && dim > 1) 2L else 1L
    for (k in seq_len(n_sub)) {
      group_of_block[b] <- g
      b <- b + 1L
    }
    pos <- pos + dim
  }
  n_groups <- g
  blocks_of_group <- split(seq_along(blocks), group_of_block)
  names(blocks_of_group) <- NULL

  block_of_col <- integer(data_dim)
  for (i in seq_along(blocks)) {
    block_of_col[blocks[[i]]$start:blocks[[i]]$end] <- i
  }

  # Categorical blocks: category per row, counts, samplers, row index.
  cat_blocks <- list()
  cat_block_ids <- integer(0)
  for (i in seq_along(blocks)) {
    bk <- blocks[[i]]
    # Only genuinely discrete columns are conditioning columns. The
    # mode-indicator sub-block of a "mode_specific" column is categorical for
    # the consistency loss, but it is not a discrete column of the data, so
    # CTGAN's column sampler must not draw it.
    if (!bk$type %in% c("softmax", "cossin")) next
    sub <- data[, bk$start:bk$end, drop = FALSE]
    if (identical(bk$type, "cossin")) {
      angle <- atan2(sub[, 2], sub[, 1]) %% (2 * pi)
      keys <- round(angle, 6)
      lev <- sort(unique(keys))
      ids <- match(keys, lev)
      values <- cbind(cos(lev), sin(lev))
    } else {
      ids <- max.col(sub, ties.method = "first")
      n_cat <- ncol(sub)
      lev <- seq_len(n_cat)
      values <- diag(1, nrow = n_cat, ncol = n_cat)
    }
    n_cat <- nrow(values)
    counts <- tabulate(ids, nbins = n_cat)
    # CTGAN's training-by-sampling category sampler: probability proportional
    # to log(count + 1) within the column (Xu et al. 2019, section 4.3).
    logw <- log(counts + 1)
    logw[counts == 0] <- 0
    log_prob <- if (sum(logw) > 0) logw / sum(logw) else rep(1 / n_cat, n_cat)
    freq_prob <- if (sum(counts) > 0) counts / sum(counts) else rep(1 / n_cat, n_cat)
    rows <- split(seq_len(nrow(data)), factor(ids, levels = seq_len(n_cat)))
    names(rows) <- NULL
    cat_blocks[[length(cat_blocks) + 1]] <- list(
      block = i,
      group = group_of_block[i],
      start = bk$start,
      end = bk$end,
      type = bk$type,
      n_cat = n_cat,
      values = values,
      counts = counts,
      log_prob = log_prob,
      freq_prob = freq_prob,
      rows = rows
    )
    cat_block_ids <- c(cat_block_ids, i)
  }

  list(
    n_blocks = length(blocks),
    data_dim = data_dim,
    cond_dim = length(blocks) + data_dim,
    blocks = blocks,
    block_of_col = block_of_col,
    group_of_block = group_of_block,
    blocks_of_group = blocks_of_group,
    n_groups = n_groups,
    cat_blocks = cat_blocks,
    cat_block_ids = cat_block_ids,
    n_rows = nrow(data)
  )
}


#' @title Expand a block mask to a column mask
#'
#' @param mask A (batch x n_blocks) numeric matrix.
#' @param meta The conditioning metadata from [build_conditioning_meta()].
#'
#' @return A (batch x data_dim) numeric matrix.
#' @keywords internal
expand_block_mask <- function(mask, meta) {
  mask[, meta$block_of_col, drop = FALSE]
}


#' @title Draw categories with probability proportional to log-frequency
#'
#' @param cat_block One element of `meta$cat_blocks`.
#' @param n Number of draws.
#' @param log_frequency Use the log-frequency probabilities (CTGAN, default)
#'   or the plain empirical frequencies?
#'
#' @return An integer vector of category indices of length `n`.
#' @keywords internal
draw_categories <- function(cat_block, n, log_frequency = TRUE) {
  p <- if (isTRUE(log_frequency)) cat_block$log_prob else cat_block$freq_prob
  sample.int(cat_block$n_cat, size = n, replace = TRUE, prob = p)
}


# ---------------------------------------------------------------------------
# Per-step condition construction
# ---------------------------------------------------------------------------

#' @title Draw `n` rows from a category's row pool
#'
#' @param pool An integer vector of row indices carrying a category.
#' @param n Number of draws (with replacement).
#' @param n_rows Total number of training rows, used when the pool is empty.
#'
#' @return An integer vector of length `n`.
#' @keywords internal
draw_rows <- function(pool, n, n_rows) {
  if (n == 0) return(integer(0))
  if (length(pool) == 0) return(sample.int(n_rows, n, replace = TRUE))
  if (length(pool) == 1) return(rep(pool, n))
  pool[sample.int(length(pool), n, replace = TRUE)]
}


#' @title Build the real batch and its condition for one training step
#'
#' @description Draws the real batch and the per-row condition of one training
#'   step. For `type = "single_column"` this is CTGAN's training-by-sampling:
#'   a discrete column is drawn uniformly, a category within it with
#'   probability proportional to `log(count + 1)`, and the real row is drawn
#'   uniformly from the rows that carry that category, so the real batch is
#'   sampled *by* the condition. For `type = "random_mask"` the observed subset
#'   is drawn per row (TabMT-style any-order training) and the observed values
#'   are simply the real row's own values, so no sampler is needed.
#'
#' @param data The training data (torch tensor or matrix).
#' @param data_matrix The training data as a plain numeric matrix, used for the
#'   condition values.
#' @param batch_size Batch size.
#' @param conditioning The resolved conditioning list.
#' @param meta The conditioning metadata from [build_conditioning_meta()].
#' @param device Torch device.
#'
#' @return A list with `real_data` (torch tensor), `cond` (torch tensor,
#'   `batch x (n_blocks + data_dim)`), `mask` and `values` (torch tensors), and
#'   `idx` (the row indices drawn).
#' @keywords internal
build_step_condition <- function(data, data_matrix, batch_size, conditioning,
                                 meta, device = "cpu") {
  n_rows <- meta$n_rows
  n_blocks <- meta$n_blocks

  if (identical(conditioning$type, "single_column")) {
    if (length(meta$cat_blocks) == 0) {
      stop("conditioning$type = \"single_column\" requires at least one categorical column in output_info")
    }
    # Uniformly over the discrete columns, then the category by log-frequency.
    which_cat <- sample.int(length(meta$cat_blocks), size = batch_size,
                            replace = TRUE)
    idx <- integer(batch_size)
    mask <- matrix(0, nrow = batch_size, ncol = n_blocks)
    for (k in unique(which_cat)) {
      sel <- which(which_cat == k)
      cb <- meta$cat_blocks[[k]]
      cats <- draw_categories(cb, length(sel), conditioning$log_frequency)
      # Draw all rows of a (column, category) cell in one call; a per-row
      # loop here would dominate the runtime of a step.
      for (cat_j in unique(cats)) {
        target <- sel[cats == cat_j]
        idx[target] <- draw_rows(cb$rows[[cat_j]], length(target), n_rows)
      }
      mask[sel, cb$block] <- 1
    }
  } else {
    idx <- sample.int(n_rows, size = batch_size, replace = batch_size > n_rows)
    # Per-row Bernoulli mask over the original columns, so that the two blocks
    # of a mode-specific column are observed together.
    rate <- if (identical(conditioning$mask_prob, "uniform")) {
      stats::runif(batch_size)
    } else {
      rep(conditioning$mask_prob, batch_size)
    }
    group_mask <- matrix(
      stats::rbinom(batch_size * meta$n_groups, 1, rep(rate, meta$n_groups)),
      nrow = batch_size, ncol = meta$n_groups
    )
    mask <- group_mask[, meta$group_of_block, drop = FALSE]

    # resample_rare: for a fraction of the rows one observed categorical block
    # is re-drawn by log-frequency and the real row is replaced by a row that
    # carries the drawn category. Replacing the row (rather than only the
    # condition values) keeps the (real row, condition) pair consistent, which
    # the discriminator relies on, while still conditioning on rare levels far
    # more often than their empirical frequency.
    if (isTRUE(conditioning$resample_rare) && length(meta$cat_blocks) > 0 &&
        conditioning$resample_rare_frac > 0) {
      hit <- which(stats::runif(batch_size) < conditioning$resample_rare_frac)
      if (length(hit) > 0) {
        # One observed categorical block per hit row, uniformly: give every
        # observed block an independent U(0, 1) score and take the argmax.
        obs <- mask[hit, meta$cat_block_ids, drop = FALSE]
        score <- matrix(stats::runif(length(obs)), nrow = nrow(obs)) * obs
        any_obs <- rowSums(obs) > 0
        chosen <- max.col(score, ties.method = "first")
        for (k in unique(chosen[any_obs])) {
          rows_k <- hit[any_obs & chosen == k]
          cb <- meta$cat_blocks[[k]]
          cats <- draw_categories(cb, length(rows_k), conditioning$log_frequency)
          for (cat_j in unique(cats)) {
            target <- rows_k[cats == cat_j]
            idx[target] <- draw_rows(cb$rows[[cat_j]], length(target), n_rows)
          }
        }
      }
    }
  }

  batch_matrix <- data_matrix[idx, , drop = FALSE]

  # Condition dropout (classifier-free guidance style, report item L7): with
  # probability cond_dropout the row's condition is the all-unobserved mask.
  if (conditioning$cond_dropout > 0) {
    drop <- stats::runif(batch_size) < conditioning$cond_dropout
    if (any(drop)) mask[drop, ] <- 0
  }

  values <- batch_matrix * expand_block_mask(mask, meta)

  real_data <- if (inherits(data, "torch_tensor")) {
    data[idx]$to(device = device)
  } else {
    torch::torch_tensor(batch_matrix)$to(device = device)
  }
  mask_t <- torch::torch_tensor(mask)$to(device = device, dtype = real_data$dtype)
  values_t <- torch::torch_tensor(values)$to(device = device, dtype = real_data$dtype)

  list(
    real_data = real_data,
    mask = mask_t,
    values = values_t,
    cond = torch::torch_cat(list(mask_t, values_t), dim = 2),
    idx = idx
  )
}


#' @title Append a condition to every sample of a batch
#'
#' @description The condition is appended per sample, *before* PacGAN packing,
#'   so that the packed discriminator input is
#'   `(batch / pac, pac * (data_dim + cond_dim))` with the
#'   `[row, condition]` layout that [TokenDiscriminator()] expects from its
#'   `extra_inputs` argument.
#'
#' @param x A (batch x data_dim) torch tensor.
#' @param cond NULL or a (batch x cond_dim) torch tensor.
#'
#' @return A torch tensor.
#' @keywords internal
attach_condition <- function(x, cond) {
  if (is.null(cond)) return(x)
  torch::torch_cat(list(x, cond), dim = 2)
}


# ---------------------------------------------------------------------------
# Consistency loss
# ---------------------------------------------------------------------------

#' @title Consistency loss between generated rows and the observed condition
#'
#' @description Penalises the generator for changing the cells its condition
#'   observes, the any-order generalisation of CTGAN's cross-entropy term on
#'   its single conditional column. Per observed (row, block) cell the term is
#'   \itemize{
#'     \item cross-entropy of the generated probabilities against the observed
#'       one-hot vector for categorical blocks (`"softmax"` and the
#'       mode-indicator sub-block of a `"mode_specific"` column),
#'     \item the mean squared error between the generated and the observed
#'       values for numeric blocks and for `"cossin"` blocks (whose two
#'       outputs live on the unit circle, so a squared error is the natural
#'       chordal distance).
#'   }
#'   The terms are averaged over the observed cells, so the value does not
#'   depend on the masking rate.
#'
#' @param fake The generator output, a (batch x data_dim) torch tensor.
#' @param mask The (batch x n_blocks) condition mask.
#' @param values The (batch x data_dim) masked condition values.
#' @param meta The conditioning metadata from [build_conditioning_meta()].
#'
#' @return A scalar torch tensor.
#' @keywords internal
conditioning_consistency_loss <- function(fake, mask, values, meta) {
  total <- NULL
  count <- NULL
  for (i in seq_along(meta$blocks)) {
    bk <- meta$blocks[[i]]
    m <- mask[, i]
    f <- fake[, bk$start:bk$end, drop = FALSE]
    v <- values[, bk$start:bk$end, drop = FALSE]
    if (isTRUE(bk$categorical) && !identical(bk$type, "cossin")) {
      # "softmax" and the mode-indicator sub-block of "mode_specific".
      term <- -(v * torch::torch_log(f + 1e-8))$sum(dim = 2)
    } else {
      term <- ((f - v)^2)$mean(dim = 2)
    }
    contrib <- (term * m)$sum()
    total <- if (is.null(total)) contrib else total + contrib
    cnt <- m$sum()
    count <- if (is.null(count)) cnt else count + cnt
  }
  if (is.null(total)) {
    return(torch::torch_zeros(1, device = fake$device)$squeeze())
  }
  total / (count + 1e-8)
}


# ---------------------------------------------------------------------------
# Conditions for sampling
# ---------------------------------------------------------------------------

#' @title Build a condition matrix from a user data.frame
#'
#' @description Turns a data.frame in the *original* column space, where `NA`
#'   means "not observed", into the `(n x (n_blocks + data_dim))` condition
#'   matrix the conditional generator expects. The NA cells are first filled
#'   with a placeholder (the first level of a discrete column, zero for a
#'   continuous one) so that `transformer$transform()` can run, and the blocks
#'   of the NA columns are then masked out, so the placeholder never reaches
#'   the generator.
#'
#' @param condition A data.frame with (a subset of) the original column names.
#'   One row is broadcast to all `n_samples`; otherwise it must have
#'   `n_samples` rows.
#' @param transformer The fitted `data_transformer`.
#' @param meta The conditioning metadata of the trained GAN.
#' @param n_samples The number of samples to generate.
#'
#' @return A list with `mask` and `values` matrices and the combined `cond`.
#' @keywords internal
condition_matrix_from_df <- function(condition, transformer, meta, n_samples) {
  if (is.null(transformer)) {
    stop("a `condition` needs the `transformer` that was used to fit the data")
  }
  if (!is.data.frame(condition)) condition <- as.data.frame(condition)
  if (nrow(condition) == 1 && n_samples > 1) {
    condition <- condition[rep(1, n_samples), , drop = FALSE]
  }
  if (nrow(condition) != n_samples) {
    stop(sprintf("`condition` must have 1 or %d rows but has %d",
                 n_samples, nrow(condition)))
  }
  metas <- transformer$meta
  if (length(metas) != meta$n_groups) {
    stop("the transformer does not match the conditioning metadata of this model")
  }
  names_all <- vapply(metas, function(m) as.character(m$name), character(1))
  unknown <- setdiff(colnames(condition), names_all)
  if (length(unknown) > 0) {
    stop(sprintf("unknown condition column(s): %s",
                 paste(unknown, collapse = ", ")))
  }

  filled <- vector("list", length(metas))
  observed <- matrix(FALSE, nrow = n_samples, ncol = length(metas))
  for (i in seq_along(metas)) {
    m <- metas[[i]]
    nm <- as.character(m$name)
    discrete <- "levs" %in% names(m)
    if (nm %in% colnames(condition)) {
      col <- condition[[nm]]
      obs <- !is.na(col)
    } else {
      col <- rep(NA, n_samples)
      obs <- rep(FALSE, n_samples)
    }
    if (discrete) {
      col <- as.character(col)
      bad <- obs & !(col %in% m$levs)
      if (any(bad)) {
        stop(sprintf("condition column '%s' contains level(s) not seen in training: %s",
                     nm, paste(unique(col[bad]), collapse = ", ")))
      }
      col[!obs] <- m$levs[1]
    } else {
      col <- as.numeric(col)
      col[!obs] <- 0
    }
    filled[[i]] <- col
    observed[, i] <- obs
  }
  names(filled) <- names_all
  filled_df <- as.data.frame(filled, stringsAsFactors = FALSE)
  colnames(filled_df) <- names_all

  values <- transformer$transform(filled_df)
  values <- matrix(as.numeric(values), nrow = n_samples)
  if (ncol(values) != meta$data_dim) {
    stop("the transformer output width does not match the trained model")
  }
  values[is.na(values)] <- 0

  mask <- matrix(0, nrow = n_samples, ncol = meta$n_blocks)
  for (i in seq_along(metas)) {
    bs <- meta$blocks_of_group[[i]]
    mask[, bs] <- ifelse(observed[, i], 1, 0)
  }
  values <- values * expand_block_mask(mask, meta)

  list(mask = mask, values = values, cond = cbind(mask, values),
       observed = observed)
}


#' @title Sample CTGAN-style unconditional conditions
#'
#' @description For a model trained with `conditioning$type = "single_column"`,
#'   unconditional sampling follows CTGAN: for every row a discrete column is
#'   drawn uniformly and a category within it is drawn with its *empirical*
#'   frequency (not the log-frequency used during training), so that the
#'   marginal of the conditioned column is preserved.
#'
#' @param meta The conditioning metadata of the trained GAN.
#' @param n_samples Number of rows.
#'
#' @return A list with `mask`, `values` and `cond` matrices.
#' @keywords internal
sample_single_column_condition <- function(meta, n_samples) {
  if (length(meta$cat_blocks) == 0) {
    stop("single_column sampling requires at least one categorical column")
  }
  mask <- matrix(0, nrow = n_samples, ncol = meta$n_blocks)
  values <- matrix(0, nrow = n_samples, ncol = meta$data_dim)
  which_cat <- sample.int(length(meta$cat_blocks), size = n_samples,
                          replace = TRUE)
  for (k in unique(which_cat)) {
    sel <- which(which_cat == k)
    cb <- meta$cat_blocks[[k]]
    cats <- draw_categories(cb, length(sel), log_frequency = FALSE)
    mask[sel, cb$block] <- 1
    values[sel, cb$start:cb$end] <- cb$values[cats, , drop = FALSE]
  }
  list(mask = mask, values = values, cond = cbind(mask, values))
}


#' @title An all-unobserved condition
#'
#' @param meta The conditioning metadata of the trained GAN.
#' @param n_samples Number of rows.
#'
#' @return A list with `mask`, `values` and `cond` matrices of zeros.
#' @keywords internal
zero_condition <- function(meta, n_samples) {
  mask <- matrix(0, nrow = n_samples, ncol = meta$n_blocks)
  values <- matrix(0, nrow = n_samples, ncol = meta$data_dim)
  list(mask = mask, values = values, cond = cbind(mask, values))
}
