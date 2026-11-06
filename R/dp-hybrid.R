# ---------------------------------------------------------------------------
# dp-hybrid.R -- the marginal/copula split of end-to-end-DP-RGAN.md section 9.3.
#
# The idea: do not ask one differentially private neural generator to learn the
# whole joint distribution. Split the problem.
#
#   1. The MARGINALS are privatised separately, with a Gaussian mechanism on a
#      histogram over PUBLIC bin edges. This is cheap, low-dimensional and
#      low-sensitivity, and it is the part of the problem where marginal-based
#      methods are known to do well.
#   2. The flow is then trained under DP-SGD on the COPULA -- the joint
#      distribution of the probability-integral-transformed columns -- and never
#      has to spend capacity or privacy budget on the marginals at all.
#
# So the privacy budget is shared between a stage that is easy to make private
# and a stage that is hard, instead of being spent entirely on the hard one.
#
# Two properties make this more than a repackaging of what already exists:
#
#   * The flow's job shrinks from "learn the whole distribution" to "learn the
#     dependence structure". The marginals are removed exactly, by construction,
#     rather than approximately, by fitting.
#   * In probit space the copula is closer to Gaussian than the raw data is, so
#     the optimal transport path the flow integrates is closer to straight --
#     fewer effective steps, and a better-conditioned regression target. This is
#     the same gaussianisation the quantile transform buys, obtained without any
#     per-column fitted state that would itself have to be released.
#
# PRECONDITIONS -- these are not conveniences, and the guarantee is void without
# them (see the main report, sections 2.2 and 4.2):
#
#   * The bin edges are PUBLIC. They are an input to the mechanism, not a
#     fitted quantity. A data-derived bin grid is an extra release that this
#     code does not account for.
#   * The value range of every continuous column is public, for the same reason.
#   * The level set of every categorical column is public (a codebook). If
#     categories are defined as "a value observed in the data", the level set is
#     itself a release and must be privatised before this code is used.
#
# Cost: epsilon_meta = c * epsilon_per_column by basic composition across the c
# columns, plus epsilon_train for the flow. The two stages are sequential and
# adaptive, so the budgets add.
# ---------------------------------------------------------------------------


#' @title Privatised Marginals over Public Bin Edges
#'
#' @description Release one noisy histogram per column under the Gaussian
#'   mechanism, and turn each into a quantile function that the copula transform
#'   can use.
#'
#' @details Each column contributes one histogram. Under add/remove-one
#'   adjacency a histogram has L2 sensitivity 1: adding or removing one record
#'   changes exactly one cell by one. The Gaussian mechanism therefore adds
#'   noise with standard deviation
#'   \deqn{\sigma_{cell} = \frac{\sqrt{2 \ln(1.25/\delta_{col})}}{\epsilon_{col}}}
#'   to every cell, where \eqn{\delta_{col}} is the column's share of the total
#'   \eqn{\delta}.
#'
#'   The bins are supplied by the caller or derived from a \emph{public} range.
#'   \code{bin_edges = NULL} with \code{range} is the intended path: a public
#'   range plus a public bin count is a constant of the mechanism. Deriving the
#'   range from the data would be a release that this function does not account
#'   for, and it warns when it has to fall back to it.
#'
#'   Counts are clamped at zero after noise is added -- the mechanism is still
#'   the Gaussian one over the reals, but the released table should not carry
#'   negative counts, and clamping is post-processing of the release.
#'
#' @param data A data.frame or matrix of the private rows.
#' @param continuous_columns Character vector of numeric column names.
#' @param categorical_columns Character vector of categorical column names. Each
#'   column's level set is taken to be public; it is read off the data and the
#'   number of levels is reported, so a caller with a codebook should pass the
#'   levels explicitly through \code{levels} to avoid any dependence on the data.
#' @param levels Optional named list of level vectors, one per categorical
#'   column, fixing the public level set and its order.
#' @param range Optional named list \code{list(col = c(lo, hi))} of PUBLIC
#'   ranges for the continuous columns.
#' @param n_bins Number of histogram bins per continuous column.
#' @param epsilon_per_column The per-column budget \eqn{\epsilon_{col}}.
#' @param delta The total \eqn{\delta}; each column gets \code{delta / c}.
#' @param seed Optional integer seed.
#'
#' @return An object of class \code{"dp_marginals"}: a per-column list of the
#'   public bins, the noisy counts, the normalised CDF and the interpolated
#'   quantile function, plus a \code{privacy} field carrying
#'   \code{epsilon_meta = c * epsilon_per_column}.
#'
#' @seealso [copula_transform()], [dp_copula_flow_trainer()]
#'
#' @examples
#' \dontrun{
#' m <- dp_marginals(toy_modes[, c("n1", "n2", "c1")],
#'                   continuous_columns = c("n1", "n2"),
#'                   categorical_columns = "c1",
#'                   range = list(n1 = c(-15, 15), n2 = c(-15, 15)),
#'                   epsilon_per_column = 0.1)
#' print(m)
#' }
#' @export
dp_marginals <- function(data,
                         continuous_columns = character(0),
                         categorical_columns = character(0),
                         levels = NULL,
                         range = NULL,
                         n_bins = 50L,
                         epsilon_per_column = 0.1,
                         delta = 1e-5,
                         seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  df <- as.data.frame(data)
  n_bins <- as.integer(n_bins)
  if (is.na(n_bins) || n_bins < 2L) stop("n_bins must be at least 2")
  if (!is.numeric(epsilon_per_column) || epsilon_per_column <= 0) {
    stop("epsilon_per_column must be positive")
  }

  cols <- c(continuous_columns, categorical_columns)
  if (!length(cols)) stop("no columns given")
  missing <- setdiff(cols, names(df))
  if (length(missing)) stop("columns not in data: ", paste(missing, collapse = ", "))
  c_total <- length(cols)
  delta_col <- delta / c_total
  # sigma for the Gaussian mechanism on a sensitivity-1 histogram.
  sigma_cell <- sqrt(2 * log(1.25 / delta_col)) / epsilon_per_column

  spec <- list()
  for (nm in cols) {
    x <- df[[nm]]
    if (nm %in% categorical_columns) {
      lev <- if (!is.null(levels) && !is.null(levels[[nm]])) {
        as.character(levels[[nm]])
      } else {
        sort(unique(as.character(x[!is.na(x)])))
      }
      counts <- tabulate(match(as.character(x), lev), nbins = length(lev))
      noisy <- pmax(counts + stats::rnorm(length(lev), 0, sigma_cell), 0)
      p <- noisy / sum(noisy)
      spec[[nm]] <- list(kind = "categorical", levels = lev,
                         count_true = counts, count_noisy = noisy, prob = p,
                         cdf = cumsum(p),
                         # public bin edges are not needed for a discrete column
                         edges = NULL)
    } else {
      rng <- if (!is.null(range) && !is.null(range[[nm]])) {
        as.numeric(range[[nm]])
      } else {
        warning("no public range supplied for '", nm,
                "'; using the data range, which is a release this function ",
                "does not account for.", call. = FALSE)
        r <- range(as.numeric(x), na.rm = TRUE)
        r
      }
      if (!all(is.finite(rng)) || rng[1] >= rng[2]) {
        stop("public range for '", nm, "' must be finite with lo < hi")
      }
      edges <- seq(rng[1], rng[2], length.out = n_bins + 1L)
      idx <- findInterval(as.numeric(x), edges, all.inside = TRUE)
      counts <- tabulate(idx, nbins = n_bins)
      noisy <- pmax(counts + stats::rnorm(n_bins, 0, sigma_cell), 0)
      p <- noisy / sum(noisy)
      spec[[nm]] <- list(kind = "continuous", levels = NULL,
                         count_true = counts, count_noisy = noisy, prob = p,
                         cdf = cumsum(p), edges = edges)
    }
  }

  out <- list(
    columns = spec,
    continuous_columns = continuous_columns,
    categorical_columns = categorical_columns,
    privacy = list(
      epsilon_meta = c_total * epsilon_per_column,
      epsilon_per_column = epsilon_per_column,
      delta = delta,
      delta_per_column = delta_col,
      sigma_cell = sigma_cell,
      n_columns = c_total,
      composition = "basic",
      n_bins = n_bins
    )
  )
  class(out) <- "dp_marginals"
  out
}

#' @title Quantile Function of a Privatised Marginal
#'
#' @description Map probabilities in \eqn{(0,1)} to values through one column's
#'   privatised CDF.
#'
#' @param column One element of \code{dp_marginals()$columns}.
#' @param u Numeric vector of probabilities.
#' @return Values in the column's domain: numeric for a continuous column, a
#'   factor level (character) for a categorical one.
#' @keywords internal
dp_marginal_quantile <- function(column, u) {
  u <- pmin(pmax(u, 1e-8), 1 - 1e-8)
  if (identical(column$kind, "categorical")) {
    idx <- findInterval(u, column$cdf) + 1L
    idx <- pmin(pmax(idx, 1L), length(column$levels))
    column$levels[idx]
  } else {
    # Invert the privatised CDF by interpolating position (cdf) against value
    # (edges). Both vectors must be the same length, so the cdf is prefixed with
    # its value at the left edge. ties = "ordered" because clamping the noisy
    # counts at zero can make the cdf flat over a run of bins.
    y <- stats::approx(x = c(0, column$cdf), y = column$edges, xout = u,
                       rule = 2, ties = "ordered")$y
    y
  }
}

#' @title Probability of a Value under a Privatised Marginal
#'
#' @description The forward map: value to probability integral transform.
#'
#' @param column One element of \code{dp_marginals()$columns}.
#' @param x Values in the column's domain.
#' @return Numeric vector of probabilities in \eqn{(0,1)}. For a categorical
#'   column the mid-rank of the level is returned, so that inverting with
#'   [dp_marginal_quantile()] recovers the level exactly.
#' @keywords internal
dp_marginal_cdf <- function(column, x) {
  if (identical(column$kind, "categorical")) {
    idx <- match(as.character(x), column$levels)
    lo <- c(0, column$cdf)[pmax(idx, 1L)]
    hi <- column$cdf[pmin(pmax(idx, 1L), length(column$cdf))]
    mid <- (lo + hi) / 2
    mid[is.na(idx)] <- NA_real_
    mid
  } else {
    e <- column$edges
    idx <- findInterval(as.numeric(x), e, all.inside = TRUE)
    lo <- c(0, column$cdf)[idx]
    hi <- column$cdf[idx]
    frac <- (as.numeric(x) - e[idx]) / (e[idx + 1L] - e[idx])
    lo + frac * (hi - lo)
  }
}

#' @title Copula Transform and Its Inverse
#'
#' @description Map rows of the private table into the unit hypercube using the
#'   privatised marginals, and back again.
#'
#' @details The copula representation is the joint distribution of the
#'   probability-integral-transformed columns. Every column becomes a uniform
#'   variate that carries only its rank position; all the marginal information
#'   lives in the released [dp_marginals()] object and none of it has to be
#'   relearned by the flow.
#'
#'   \code{probit = TRUE} additionally maps \eqn{u \mapsto \Phi^{-1}(u)}, which
#'   is what the flow is trained on: the transport path is closer to straight in
#'   probit space, so the vector field is easier to regress and the ODE needs
#'   fewer steps at sampling time.
#'
#'   For a categorical column the transform is the discrete copula: \eqn{u} is
#'   the mid-rank of the level, so a level with privatised probability \eqn{p}
#'   occupies an interval of length \eqn{p} and inverting any point inside it
#'   recovers the level. The copula is therefore a checkerboard copula, and its
#'   density is not continuous -- worth stating, because it is the reason a flow
#'   on mixed-type tables is harder than a flow on all-continuous ones.
#'
#' @param marginals A \code{"dp_marginals"} object.
#' @param data A data.frame with the same columns.
#' @param probit Map through \eqn{\Phi^{-1}}.
#' @return A numeric matrix, one column per column of \code{data}.
#'
#' @seealso [dp_marginals()], [copula_inverse()]
#' @export
copula_transform <- function(marginals, data, probit = TRUE) {
  df <- as.data.frame(data)
  cols <- names(marginals$columns)
  u <- vapply(cols, function(nm) dp_marginal_cdf(marginals$columns[[nm]], df[[nm]]),
              numeric(nrow(df)))
  u <- matrix(u, nrow = nrow(df), dimnames = list(NULL, cols))
  u <- pmin(pmax(u, 1e-6), 1 - 1e-6)
  if (isTRUE(probit)) u <- stats::qnorm(u)
  u
}

#' @title Invert the Copula Transform
#'
#' @description Turn unit-hypercube (or probit-space) rows back into data using
#'   the privatised marginals.
#'
#' @param marginals A \code{"dp_marginals"} object.
#' @param z A numeric matrix from [copula_transform()] or from the flow.
#' @param probit Whether \code{z} is in probit space.
#' @return A data.frame with the columns of the marginal object.
#'
#' @seealso [copula_transform()]
#' @export
copula_inverse <- function(marginals, z, probit = TRUE) {
  m <- as.matrix(z)
  cols <- names(marginals$columns)
  # sample_flow() returns a bare matrix with no dimnames, so fall back to the
  # marginal object's column order rather than indexing by name.
  if (is.null(colnames(m))) {
    if (ncol(m) != length(cols)) {
      stop("z has ", ncol(m), " columns but the marginal object has ",
           length(cols), "; supply dimnames or match them up")
    }
    colnames(m) <- cols
  }
  if (!all(cols %in% colnames(m))) {
    stop("z is missing columns: ",
         paste(setdiff(cols, colnames(m)), collapse = ", "))
  }
  m <- m[, cols, drop = FALSE]
  if (isTRUE(probit)) m <- stats::pnorm(m)
  m <- pmin(pmax(m, 1e-8), 1 - 1e-8)
  out <- lapply(cols, function(nm) {
    dp_marginal_quantile(marginals$columns[[nm]], m[, nm])
  })
  names(out) <- cols
  as.data.frame(out, stringsAsFactors = FALSE)
}

#' @title Train a DP Flow on the Copula, with Privatised Marginals
#'
#' @description The pipeline of section 9.3: privatise the marginals, transform
#'   the private table into the copula, train the DP flow of
#'   [dp_flow_matching_trainer()] on the copula, and report the composed budget.
#'
#' @details The composition is sequential and adaptive, so the budgets add:
#'   \deqn{\epsilon_{total} = \epsilon_{meta} + \epsilon_{train}.}
#'   \eqn{\epsilon_{meta} = c \cdot \epsilon_{col}} is a basic composition over
#'   the \eqn{c} columns; the accountant of [dp_flow_matching_trainer()] supplies
#'   \eqn{\epsilon_{train}}.
#'
#'   The copula transform is a deterministic function of the private data and the
#'   \emph{released} marginals, so it is post-processing of the marginal release
#'   and adds nothing on its own. The flow training then reads the transformed
#'   data and is charged for it. Note what the flow never sees: the raw scale of
#'   any column. Its regression target is a rank position, and a shift or a
#'   rescaling of a column changes the copula not at all.
#'
#' @param data A data.frame of the private rows.
#' @param continuous_columns,categorical_columns,levels,range,n_bins As in
#'   [dp_marginals()].
#' @param marginal_epsilon_per_column \eqn{\epsilon_{col}} for the marginals.
#' @param delta The total \eqn{\delta}, shared by both stages.
#' @param steps,batch_size,hidden_units,base_lr,path,ema_decay,noise_multiplier,target_epsilon,max_grad_norm,noise_multiplicity,vector_field,device,seed,verbose
#'   Passed to [dp_flow_matching_trainer()]. \code{target_epsilon} is the
#'   \emph{training} budget \eqn{\epsilon_{train}}, not the total.
#' @param probit Train the flow on \eqn{\Phi^{-1}(u)}.
#' @param seed_marginals A separate seed for the marginal mechanism, so that its
#'   noise is reproducible independently of the flow's.
#'
#' @return A list of class \code{"trained_RCopulaFlow"} with \code{marginals},
#'   \code{flow} (a \code{"trained_RFlow"}), \code{privacy} (the composed
#'   \eqn{\epsilon_{meta} + \epsilon_{train}}) and \code{probit}.
#'
#' @seealso [dp_marginals()], [copula_transform()], [sample_dp_copula()],
#'   [dp_flow_matching_trainer()]
#'
#' @examples
#' \dontrun{
#' fit <- dp_copula_flow_trainer(
#'   toy_modes[, c("n1", "n2", "c1")],
#'   continuous_columns = c("n1", "n2"), categorical_columns = "c1",
#'   range = list(n1 = c(-15, 15), n2 = c(-15, 15)),
#'   marginal_epsilon_per_column = 0.1, target_epsilon = 4,
#'   steps = 2000, batch_size = 500, max_grad_norm = 1)
#' syn <- sample_dp_copula(fit, n_samples = 2000)
#' }
#' @export
dp_copula_flow_trainer <- function(data,
                                   continuous_columns = character(0),
                                   categorical_columns = character(0),
                                   levels = NULL,
                                   range = NULL,
                                   n_bins = 50L,
                                   marginal_epsilon_per_column = 0.1,
                                   delta = 1e-5,
                                   steps = 2000,
                                   batch_size = 500,
                                   hidden_units = list(256, 256),
                                   base_lr = 1e-3,
                                   path = "ot",
                                   ema_decay = 0.999,
                                   noise_multiplier = NULL,
                                   target_epsilon = NULL,
                                   max_grad_norm = 1,
                                   noise_multiplicity = 1L,
                                   vector_field = NULL,
                                   device = "cpu",
                                   seed = NULL,
                                   verbose = TRUE,
                                   probit = TRUE,
                                   seed_marginals = NULL) {
  df <- as.data.frame(data)

  marginals <- dp_marginals(
    df,
    continuous_columns = continuous_columns,
    categorical_columns = categorical_columns,
    levels = levels, range = range, n_bins = n_bins,
    epsilon_per_column = marginal_epsilon_per_column,
    delta = delta,
    seed = if (is.null(seed_marginals)) seed else seed_marginals)

  # The copula transform reads the private rows, but only through the released
  # marginals' quantile functions, so it is post-processing of that release.
  Z <- copula_transform(marginals, df, probit = probit)
  if (anyNA(Z) || any(!is.finite(Z))) {
    stop("the copula transform produced non-finite values; the table probably ",
         "has missing or out-of-range values that the public bins do not cover")
  }

  flow <- dp_flow_matching_trainer(
    data = Z, output_info = NULL, steps = steps, batch_size = batch_size,
    base_lr = base_lr, path = path, ema_decay = ema_decay,
    vector_field = vector_field, device = device, seed = seed,
    verbose = verbose, noise_multiplier = noise_multiplier,
    target_epsilon = target_epsilon, target_delta = delta,
    max_grad_norm = max_grad_norm, track_loss = TRUE,
    noise_multiplicity = noise_multiplicity)

  eps_meta <- marginals$privacy$epsilon_meta
  eps_train <- flow$privacy$epsilon
  out <- list(
    marginals = marginals,
    flow = flow,
    probit = isTRUE(probit),
    dim = ncol(Z),
    privacy = list(
      epsilon_meta = eps_meta,
      epsilon_train = eps_train,
      epsilon_total = eps_meta + eps_train,
      delta = delta,
      composition = "basic over the marginal columns, then the RDP accountant",
      marginal_epsilon_per_column = marginal_epsilon_per_column,
      target_epsilon = target_epsilon
    ),
    settings = list(
      continuous_columns = continuous_columns,
      categorical_columns = categorical_columns,
      n_bins = n_bins,
      steps = steps,
      batch_size = batch_size,
      hidden_units = hidden_units,
      probit = isTRUE(probit)
    )
  )
  class(out) <- "trained_RCopulaFlow"
  if (isTRUE(verbose)) {
    message(sprintf(
      "[dp-copula] eps_meta=%.3f (c=%d x %.3g) + eps_train=%.3f -> total %.3f (delta=%.1e)",
      eps_meta, marginals$privacy$n_columns, marginal_epsilon_per_column,
      eps_train, eps_meta + eps_train, delta))
  }
  out
}

#' @title Sample from a DP Copula Flow
#'
#' @description Draw copula samples from the trained flow and push them back
#'   through the privatised marginals.
#'
#' @param fit A \code{"trained_RCopulaFlow"} from [dp_copula_flow_trainer()].
#' @param n_samples Number of rows to draw.
#' @param n_steps Number of ODE steps for [sample_flow()].
#' @param solver \code{"euler"} or \code{"heun"}.
#' @return A data.frame of synthetic rows, in the columns' original domains.
#'
#' @seealso [dp_copula_flow_trainer()]
#' @export
sample_dp_copula <- function(fit, n_samples, n_steps = 50, solver = "heun") {
  stopifnot(inherits(fit, "trained_RCopulaFlow"))
  Z <- sample_flow(fit$flow, n_samples = n_samples, n_steps = n_steps,
                   solver = solver)
  copula_inverse(fit$marginals, Z, probit = fit$probit)
}

#' @title Print a DP Copula Flow Fit
#' @param x A \code{"trained_RCopulaFlow"}.
#' @param ... Ignored.
#' @return \code{x}, invisibly.
#' @export
print.trained_RCopulaFlow <- function(x, ...) {
  p <- x$privacy
  cat("<trained_RCopulaFlow>\n")
  cat(sprintf("  columns      : %d\n", x$dim))
  cat(sprintf("  epsilon_meta : %.3f  (%d columns x %.3g per column)\n",
              p$epsilon_meta,
              x$marginals$privacy$n_columns, p$marginal_epsilon_per_column))
  cat(sprintf("  epsilon_train: %.3f  (sigma = %.4g, T = %d)\n",
              p$epsilon_train, x$flow$privacy$noise_multiplier,
              x$flow$privacy$steps))
  cat(sprintf("  epsilon_total: %.3f  at delta = %.1e\n",
              p$epsilon_total, p$delta))
  invisible(x)
}
