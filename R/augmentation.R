#' @title Instance noise and adaptive augmentation helpers
#'
#' @description Internal helpers implementing Experiment 3 of the 2026
#'   experiment plan: Diffusion-GAN style adaptive instance noise
#'   (Wang et al., ICLR 2023, arXiv:2206.02262, simplified here to plain
#'   Gaussian noise without the diffusion mixture) and ADA-style adaptive
#'   differentiable augmentation (Karras et al., NeurIPS 2020) with
#'   table-specific operations (TabCutMix-like categorical swapping,
#'   Fang et al., ICML 2025).
#'
#' @name augmentation
#' @keywords internal
NULL


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

#' @title Column blocks of a transformed data matrix
#'
#' @description Turns an `output_info` list (as produced by
#'   `data_transformer$output_info`) into a flat description of the column
#'   blocks of the transformed matrix. Each block is one element of
#'   `output_info`, except that a `"mode_specific"` block is split into its
#'   categorical mode-indicator sub-block (the first `dim - 1` columns) and its
#'   numeric value sub-block (the last column), because the two behave
#'   differently under noise and augmentation.
#'
#' @param output_info The `output_info` list, or NULL.
#' @param data_dim The number of columns of the data matrix.
#'
#' @return A list with the elements `blocks` (a list of lists with `start`,
#'   `end`, `type` and `categorical`) and `ok` (was `output_info` usable?).
#'   If `output_info` is NULL or inconsistent with `data_dim`, every column
#'   becomes its own numeric block.
#' @keywords internal
output_info_blocks <- function(output_info, data_dim) {
  numeric_fallback <- function() {
    list(
      blocks = lapply(seq_len(data_dim), function(j) {
        list(start = j, end = j, type = "linear", categorical = FALSE)
      }),
      ok = FALSE
    )
  }

  if (is.null(output_info) || !is.list(output_info) || length(output_info) == 0) {
    return(numeric_fallback())
  }

  total <- 0
  for (info in output_info) {
    if (!is.list(info) || length(info) < 2) return(numeric_fallback())
    total <- total + as.integer(info[[1]])
  }
  if (total != data_dim) {
    return(numeric_fallback())
  }

  blocks <- list()
  pos <- 1L
  for (info in output_info) {
    dim <- as.integer(info[[1]])
    type <- as.character(info[[2]])
    if (type == "mode_specific" && dim > 1) {
      # [1 .. dim-1] one-hot mode indicators, [dim] the normalised value.
      blocks[[length(blocks) + 1]] <-
        list(start = pos, end = pos + dim - 2L, type = "mode_indicator",
             categorical = TRUE)
      blocks[[length(blocks) + 1]] <-
        list(start = pos + dim - 1L, end = pos + dim - 1L,
             type = "mode_value", categorical = FALSE)
    } else {
      blocks[[length(blocks) + 1]] <-
        list(start = pos, end = pos + dim - 1L, type = type,
             categorical = type %in% c("softmax", "cossin"))
    }
    pos <- pos + dim
  }

  list(blocks = blocks, ok = TRUE)
}


# ---------------------------------------------------------------------------
# Instance noise
# ---------------------------------------------------------------------------

#' @title Per-column instance-noise scale
#'
#' @description Builds the per-column multiplier applied to the instance noise.
#'   With `per_type = TRUE` the noise on categorical blocks (`"softmax"`,
#'   `"cossin"`) and on the mode-indicator sub-block of a `"mode_specific"`
#'   block is halved, while numeric columns keep a multiplier of 1. This is a
#'   deliberately simple per-type noise schedule, inspired by the finding of
#'   CDTD (Mueller et al., ICLR 2025) that continuous and categorical columns
#'   need different noise levels; it is not CDTD's learned per-type schedule.
#'   If `output_info` is NULL every column is treated as numeric.
#'
#' @param output_info The `output_info` list, or NULL.
#' @param data_dim The number of columns of the data matrix.
#' @param per_type Use the per-type schedule? If FALSE every column gets 1.
#'
#' @return A numeric vector of length `data_dim`.
#' @keywords internal
instance_noise_scale <- function(output_info, data_dim, per_type = TRUE) {
  scale <- rep(1, data_dim)
  if (!isTRUE(per_type)) return(scale)
  bl <- output_info_blocks(output_info, data_dim)
  if (!bl$ok) return(scale)
  for (b in bl$blocks) {
    if (isTRUE(b$categorical)) {
      scale[b$start:b$end] <- 0.5
    }
  }
  scale
}


#' @title Resolve the instance-noise settings
#'
#' @description Internal helper that fills in the defaults of the
#'   `instance_noise` argument of [gan_trainer()].
#'
#' @param instance_noise NULL or a list, see [gan_trainer()].
#'
#' @return NULL, or a fully specified list.
#' @keywords internal
resolve_instance_noise <- function(instance_noise) {
  if (is.null(instance_noise)) return(NULL)
  if (isFALSE(instance_noise)) return(NULL)
  if (!is.list(instance_noise)) {
    stop("instance_noise must be NULL or a list")
  }
  get1 <- function(name, default) {
    v <- instance_noise[[name]]
    if (is.null(v)) default else v
  }
  out <- list(
    sigma_max = get1("sigma_max", 0.5),
    sigma_min = get1("sigma_min", 0),
    adaptive = isTRUE(get1("adaptive", TRUE)),
    target_rd = get1("target_rd", 0.6),
    per_type = isTRUE(get1("per_type", TRUE)),
    condition_d = isTRUE(get1("condition_d", FALSE)),
    update_every = as.integer(get1("update_every", 4))
  )
  if (!is.numeric(out$sigma_max) || length(out$sigma_max) != 1 || out$sigma_max < 0) {
    stop("instance_noise$sigma_max must be a single non-negative number")
  }
  if (!is.numeric(out$sigma_min) || length(out$sigma_min) != 1 || out$sigma_min < 0) {
    stop("instance_noise$sigma_min must be a single non-negative number")
  }
  if (out$sigma_min > out$sigma_max) {
    stop("instance_noise$sigma_min must not exceed instance_noise$sigma_max")
  }
  if (!is.numeric(out$target_rd) || length(out$target_rd) != 1) {
    stop("instance_noise$target_rd must be a single number")
  }
  if (is.na(out$update_every) || out$update_every < 1) {
    stop("instance_noise$update_every must be a positive integer")
  }
  out
}


#' @title Resolve the augmentation settings
#'
#' @description Internal helper that fills in the defaults of the
#'   `augmentation` argument of [gan_trainer()].
#'
#' @param augmentation NULL or a list, see [gan_trainer()].
#'
#' @return NULL, or a fully specified list.
#' @keywords internal
resolve_augmentation <- function(augmentation) {
  if (is.null(augmentation)) return(NULL)
  if (isFALSE(augmentation)) return(NULL)
  if (!is.list(augmentation)) {
    stop("augmentation must be NULL or a list")
  }
  get1 <- function(name, default) {
    v <- augmentation[[name]]
    if (is.null(v)) default else v
  }
  ops <- get1("ops", c("mask", "mixup", "swap"))
  valid_ops <- c("mask", "mixup", "swap")
  if (!is.character(ops) || length(ops) == 0 || !all(ops %in% valid_ops)) {
    stop(sprintf("augmentation$ops must be a non-empty subset of: %s",
                 paste(valid_ops, collapse = ", ")))
  }
  out <- list(
    p_max = get1("p_max", 0.5),
    adaptive = isTRUE(get1("adaptive", TRUE)),
    target_rd = get1("target_rd", 0.6),
    ops = ops,
    update_every = as.integer(get1("update_every", 4))
  )
  if (!is.numeric(out$p_max) || length(out$p_max) != 1 ||
      out$p_max < 0 || out$p_max > 1) {
    stop("augmentation$p_max must be a single number in [0, 1]")
  }
  if (!is.numeric(out$target_rd) || length(out$target_rd) != 1) {
    stop("augmentation$target_rd must be a single number")
  }
  if (is.na(out$update_every) || out$update_every < 1) {
    stop("augmentation$update_every must be a positive integer")
  }
  out
}


#' @title Draw uniform numbers with an optional torch generator
#' @param size Integer vector with the shape.
#' @param generator Optional `torch::torch_generator()`.
#' @param device Device string.
#' @return A torch tensor.
#' @keywords internal
aug_rand <- function(size, generator = NULL, device = "cpu") {
  if (is.null(generator)) {
    torch::torch_rand(size, device = device)
  } else {
    # torch_rand() has no generator argument in R torch 0.17; the in-place
    # uniform_() does.
    torch::torch_empty(size, device = device)$uniform_(0, 1,
                                                       generator = generator)
  }
}


# ---------------------------------------------------------------------------
# augment_batch
# ---------------------------------------------------------------------------

#' @title Differentiable adaptive augmentation of a tabular batch
#'
#' @description Applies the tabular augmentation operations of Experiment 3 to
#'   a batch, independently per row and independently per operation, each with
#'   probability `p`. The result is differentiable with respect to `x`, so the
#'   same function can be applied to generator output inside the generator
#'   step (ADA's requirement, Karras et al., NeurIPS 2020).
#'
#'   \describe{
#'     \item{`"mask"`}{Each column block of a row is set to 0 with probability
#'       `p`. Blocks come from `output_info`; without `output_info` every
#'       column is its own block. Implemented as a multiplication with a 0/1
#'       tensor.}
#'     \item{`"mixup"`}{With probability `p` the numeric columns of a row are
#'       replaced by a convex combination with a random other row of the batch,
#'       `lambda * x_i + (1 - lambda) * x_j` with `lambda ~ U(0.5, 1)`.
#'       Categorical blocks are untouched, so the row stays a valid one-hot
#'       row.}
#'     \item{`"swap"`}{Rows are paired at random; with probability `p` a pair
#'       exchanges one randomly chosen categorical block (a cheap tabular
#'       CutMix, TabCutMix, Fang et al., ICML 2025). Because the two rows of a
#'       pair exchange their values, the multiset of values in each
#'       categorical block is preserved exactly. Implemented as an index
#'       gather.}
#'   }
#'
#'   All randomness is drawn from torch, so the function is deterministic
#'   given `torch::torch_manual_seed()` (or the optional `generator`).
#'
#' @param x A torch tensor of shape (batch, data_dim).
#' @param p The augmentation probability, a number in `[0, 1]`. `p = 0` returns
#'   `x` unchanged (up to the identity operations, which are exact).
#' @param ops Character vector, a subset of `c("mask", "mixup", "swap")`.
#'   Applied in that order.
#' @param output_info Optional `output_info` list describing the column blocks.
#' @param generator Optional `torch::torch_generator()` used for all random
#'   draws. Defaults to NULL, i.e. the global torch RNG.
#'
#' @return A torch tensor of the same shape as `x`.
#' @keywords internal
augment_batch <- function(x, p, ops = c("mask", "mixup", "swap"),
                          output_info = NULL, generator = NULL) {
  if (length(ops) == 0 || is.null(p)) return(x)
  if (!is.numeric(p) || length(p) != 1 || is.na(p) || p < 0 || p > 1) {
    stop("p must be a single number in [0, 1]")
  }

  n <- x$shape[1]
  d <- x$shape[2]
  device <- x$device
  bl <- output_info_blocks(output_info, d)
  blocks <- bl$blocks

  numeric_cols <- unlist(lapply(blocks, function(b) {
    if (isTRUE(b$categorical)) integer(0) else b$start:b$end
  }))
  cat_blocks <- Filter(function(b) isTRUE(b$categorical), blocks)

  out <- x

  # --- mask -----------------------------------------------------------------
  if ("mask" %in% ops) {
    n_blocks <- length(blocks)
    keep_block <- (aug_rand(c(n, n_blocks), generator, device) >= p)$to(
      dtype = out$dtype)
    # Expand the per-block keep flags to per-column flags.
    col_of_block <- integer(d)
    for (bi in seq_along(blocks)) {
      col_of_block[blocks[[bi]]$start:blocks[[bi]]$end] <- bi
    }
    keep_cols <- torch::torch_index_select(
      keep_block, 2,
      torch::torch_tensor(as.integer(col_of_block), dtype = torch::torch_long(),
                          device = device)
    )
    out <- out * keep_cols
  }

  # --- mixup ----------------------------------------------------------------
  if ("mixup" %in% ops && length(numeric_cols) > 0 && n > 1) {
    u <- as.numeric(torch::as_array(aug_rand(c(n), generator, device)$cpu()))
    perm <- order(u)
    perm_t <- torch::torch_tensor(as.integer(perm), dtype = torch::torch_long(),
                                  device = device)
    lam <- 0.5 + 0.5 * aug_rand(c(n, 1), generator, device)
    sel <- (aug_rand(c(n, 1), generator, device) < p)$to(dtype = out$dtype)
    # lambda = 1 for rows that are not selected, so those rows are unchanged.
    eff_lam <- sel * lam + (1 - sel)
    partner <- torch::torch_index_select(out, 1, perm_t)
    mixed <- eff_lam * out + (1 - eff_lam) * partner
    # Only the numeric columns are mixed.
    num_mask <- rep(0, d)
    num_mask[numeric_cols] <- 1
    num_mask_t <- torch::torch_tensor(matrix(num_mask, nrow = 1),
                                      device = device)$to(dtype = out$dtype)
    out <- num_mask_t * mixed + (1 - num_mask_t) * out
  }

  # --- swap -----------------------------------------------------------------
  # Fully vectorised: no loop over blocks. Rows are shuffled into pairs; each
  # pair picks one categorical block and, with probability p, the two rows
  # exchange exactly that block. Because a row belongs to at most one pair and
  # exchanges at most one block, a single partner-row index vector and a
  # (batch, n_cat_blocks) selection matrix describe the whole operation. The
  # selection is expanded from blocks to columns by one matmul with a
  # (n_cat_blocks, data_dim) 0/1 block-to-column map, and the result is a
  # single mask blend of `out` with one row-gathered copy of itself, so the
  # whole op is two tensor ops and stays differentiable in the values.
  if ("swap" %in% ops && length(cat_blocks) > 0 && n > 1) {
    u <- as.numeric(torch::as_array(aug_rand(c(n), generator, device)$cpu()))
    shuffled <- order(u)
    n_pairs <- n %/% 2
    if (n_pairs > 0) {
      n_cat <- length(cat_blocks)
      which_block <- as.numeric(torch::as_array(
        aug_rand(c(n_pairs), generator, device)$cpu()))
      which_block <- pmin(n_cat, 1L + floor(which_block * n_cat))
      do_swap <- as.numeric(torch::as_array(
        aug_rand(c(n_pairs), generator, device)$cpu())) < p

      i1 <- shuffled[2 * seq_len(n_pairs) - 1]
      i2 <- shuffled[2 * seq_len(n_pairs)]

      # Partner row of every row; rows that do not swap stay their own partner,
      # which makes the gathered copy identical to `out` for them.
      partner <- seq_len(n)
      sw <- which(do_swap)
      if (length(sw) > 0) {
        partner[i1[sw]] <- i2[sw]
        partner[i2[sw]] <- i1[sw]
      }

      # (batch, n_cat_blocks) selection: 1 where this row exchanges that block.
      sel <- matrix(0, nrow = n, ncol = n_cat)
      if (length(sw) > 0) {
        sel[cbind(i1[sw], which_block[sw])] <- 1
        sel[cbind(i2[sw], which_block[sw])] <- 1
      }

      # (n_cat_blocks, data_dim) block-to-column map.
      block_cols <- matrix(0, nrow = n_cat, ncol = d)
      for (bi in seq_len(n_cat)) {
        block_cols[bi, cat_blocks[[bi]]$start:cat_blocks[[bi]]$end] <- 1
      }

      sel_t <- torch::torch_tensor(sel, device = device)$to(dtype = out$dtype)
      block_cols_t <- torch::torch_tensor(block_cols,
                                          device = device)$to(dtype = out$dtype)
      # (batch, data_dim) 0/1 column mask.
      col_mask <- torch::torch_matmul(sel_t, block_cols_t)

      partner_t <- torch::torch_tensor(as.integer(partner),
                                       dtype = torch::torch_long(),
                                       device = device)
      gathered <- torch::torch_index_select(out, 1, partner_t)
      out <- (1 - col_mask) * out + col_mask * gathered
    }
  }

  out
}
