#' @title Data Transformer
#'
#' @description An R6 class for preprocessing tabular data before GAN training.
#'   The transformer learns normalization parameters from data and provides
#'   reversible transformations to convert between original and GAN-ready formats.
#'
#' @details
#' ## Overview
#'
#' GANs work best when input data is normalized to a consistent scale. The
#' `data_transformer` class handles this preprocessing automatically:
#'
#' 1. **Fit**: Learn transformation parameters from your data
#' 2. **Transform**: Convert data to normalized format for GAN training
#' 3. **Inverse Transform**: Convert GAN output back to original scale
#'
#' ## Normalization Methods
#'
#' ### Standard Normalization (default)
#'
#' Applies z-score standardization to continuous columns:
#' \deqn{z = \frac{x - \mu}{\sigma}}
#'
#' where \eqn{\mu} is the column mean and \eqn{\sigma} is the standard deviation.
#' This maps data to approximately zero mean and unit variance.
#'
#' **Best for**: Data with roughly Gaussian distributions or when simplicity is preferred.
#'
#' ### Mode-Specific Normalization (CTGAN-style)
#'
#' For columns with multi-modal, skewed, or complex distributions, mode-specific
#' normalization fits a Gaussian Mixture Model (GMM) and normalizes each value
#' within its assigned mode. This approach is used by CTGAN (Xu et al., 2019).
#'
#' **How it works**:
#' 1. Fit a GMM with `n_modes` components using the EM algorithm
#' 2. For each value, assign it to the most likely mode
#' 3. Normalize the value within that mode's distribution
#' 4. Output includes: one-hot encoded mode indicator + normalized value
#'
#' **Output dimensions**: For a column with `k` modes, the transformed output has
#' `k + 1` columns: `k` columns for the mode indicator (one-hot) and 1 column for
#' the normalized value (clipped to \[-1, 1\]).
#'
#' **Best for**: Columns with multiple peaks, heavy tails, or skewed distributions.
#' Significantly improves GAN performance on real-world tabular data.
#'
#' ### Gaussian Quantile Transform
#'
#' With `continuous_transform = "quantile"` each continuous column is mapped
#' through its own empirical CDF and then through `qnorm()`, so the transformed
#' column is approximately standard normal whatever the shape of the original
#' distribution. This is the numeric representation used by the diffusion models
#' TabDDPM and TabSyn (Kotelnikov et al., 2023).
#'
#' The CDF is stored as a table of at most `n_quantiles` empirical quantiles.
#' Both directions are a linear interpolation over that table (`approx()` with
#' `rule = 2`), so the round trip is exact for values inside the fitted range;
#' the CDF position is clipped to \[1e-4, 1 - 1e-4\] to keep `qnorm()` finite,
#' which means the two extreme values are reconstructed approximately.
#' Constant columns are handled (they map to 0 and back to their value).
#'
#' **Output dimensions**: one column, `output_info` type `"quantile"`.
#'
#' ## Categorical Columns
#'
#' ### One-hot (default)
#'
#' Categorical (discrete) columns are one-hot encoded. Each category becomes a
#' separate binary column. The inverse transform selects the category with the
#' highest value (argmax). `output_info` type `"softmax"`.
#'
#' ### cos/sin (TabRep)
#'
#' With `categorical_encoding = "cossin"` the `K` levels of a column are placed
#' equidistantly on the unit circle: level \eqn{c \in 1..K} becomes
#' \eqn{(\cos(2\pi(c-1)/K), \sin(2\pi(c-1)/K))} (Hsi et al., 2025). A column
#' costs two output dimensions regardless of its cardinality, which matters for
#' high-cardinality columns, and the generator needs no Gumbel-Softmax.
#' The inverse takes `atan2()` of the two outputs and snaps to the nearest level
#' by circular distance. `output_info` type `"cossin"`.
#'
#' The distance each row had to be snapped, in units of the inter-level angle
#' \eqn{2\pi/K}, is summarised per column in `last_snap_stats` after each
#' `inverse_transform()` call (mean, 90th percentile, number of rows and `K`).
#' Large snap distances indicate that the generator is not landing near the
#' level positions.
#'
#' ### Integer index (embedding-ready)
#'
#' With `categorical_encoding = "embedding_ready"` the column stays a single
#' integer index in 1..K, `output_info` type `"index"`. This is a data path for
#' a future learned-embedding generator; `TabularGenerator` has no head for it
#' and rejects the type.
#'
#' ## Fields
#'
#' After fitting, the transformer stores:
#' \describe{
#'   \item{meta}{List of metadata for each column (means, stds, levels, quantile tables, etc.)}
#'   \item{output_info}{List describing the output structure for each column}
#'   \item{output_dimensions}{Total number of columns in transformed data}
#'   \item{mode_specific}{Whether mode-specific normalization was used}
#'   \item{n_modes}{Number of GMM modes (if mode_specific = TRUE)}
#'   \item{continuous_transform}{The resolved continuous representation}
#'   \item{categorical_encoding}{The categorical representation}
#'   \item{n_quantiles}{Size of the quantile reference table}
#'   \item{last_snap_stats}{Named list of cos/sin snap diagnostics from the most
#'     recent `inverse_transform()` call}
#' }
#'
#' ## Integration with RGAN
#'
#' The transformer integrates seamlessly with RGAN's training and sampling functions:
#'
#' ```r
#' # 1. Create and fit transformer
#' transformer <- data_transformer$new()
#' transformer$fit(data, discrete_columns = c("category_col"))
#'
#' # 2. Transform data for training
#' transformed_data <- transformer$transform(data)
#'
#' # 3. Train GAN
#' trained_gan <- gan_trainer(transformed_data)
#'
#' # 4. Sample and inverse transform
#' synthetic_data <- sample_synthetic_data(trained_gan, transformer)
#' ```
#'
#' For mode-specific normalization with categorical columns, use `TabularGenerator`
#' with Gumbel-Softmax for differentiable sampling (see `gan_trainer` with `output_info`).
#'
#' @references
#' Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
#' Modeling tabular data using conditional GAN. Advances in Neural Information
#' Processing Systems, 32.
#'
#' Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023).
#' TabDDPM: Modelling tabular data with diffusion models. ICML. (Source of the
#' Gaussian quantile transform used here and in TabSyn.)
#'
#' Hsi, J., et al. (2025). TabRep: Training tabular diffusion models with a
#' simple and effective continuous representation. arXiv:2504.04798.
#'
#' @return An R6 class object for transforming tabular data
#' @export
#'
#' @examples
#' \dontrun{
#' # ============================================================
#' # Example 1: Basic usage with standard normalization
#' # ============================================================
#'
#' # Load sample data
#' data <- sample_toydata()
#'
#' # Create and fit transformer
#' transformer <- data_transformer$new()
#' transformer$fit(data)
#'
#' # Transform data
#' transformed_data <- transformer$transform(data)
#' cat("Original dimensions:", dim(data), "\n")
#' cat("Transformed dimensions:", dim(transformed_data), "\n")
#'
#' # Train GAN and generate synthetic data
#' trained_gan <- gan_trainer(transformed_data, epochs = 50)
#' synthetic_data <- sample_synthetic_data(trained_gan, transformer)
#'
#' # Compare distributions
#' par(mfrow = c(1, 2))
#' plot(data, main = "Original Data")
#' plot(synthetic_data, main = "Synthetic Data")
#'
#' # ============================================================
#' # Example 2: Mode-specific normalization for complex distributions
#' # ============================================================
#'
#' # Create data with multiple modes
#' set.seed(42)
#' bimodal_data <- data.frame(
#'   x = c(rnorm(500, mean = -3), rnorm(500, mean = 3)),
#'   y = c(rnorm(500, mean = 0), rnorm(500, mean = 5))
#' )
#'
#' # Fit with mode-specific normalization
#' transformer_gmm <- data_transformer$new()
#' transformer_gmm$fit(bimodal_data, mode_specific = TRUE, n_modes = 5)
#'
#' # Check output dimensions (more columns due to mode indicators)
#' transformed_gmm <- transformer_gmm$transform(bimodal_data)
#' cat("Original columns:", ncol(bimodal_data), "\n")
#' cat("Transformed columns:", ncol(transformed_gmm), "\n")
#'
#' # ============================================================
#' # Example 3: Mixed continuous and categorical columns
#' # ============================================================
#'
#' # Create mixed data
#' mixed_data <- data.frame(
#'   age = rnorm(1000, mean = 40, sd = 15),
#'   income = rexp(1000, rate = 0.00002),
#'   gender = sample(c("M", "F"), 1000, replace = TRUE),
#'   education = sample(c("HS", "BA", "MA", "PhD"), 1000, replace = TRUE)
#' )
#'
#' # Fit transformer specifying categorical columns
#' transformer_mixed <- data_transformer$new()
#' transformer_mixed$fit(
#'   mixed_data,
#'   discrete_columns = c("gender", "education"),
#'   mode_specific = TRUE,  # GMM for continuous columns
#'   n_modes = 5
#' )
#'
#' # Transform
#' transformed_mixed <- transformer_mixed$transform(mixed_data)
#' cat("Output dimensions:", transformer_mixed$output_dimensions, "\n")
#'
#' # Inverse transform preserves types
#' reconstructed <- transformer_mixed$inverse_transform(transformed_mixed)
#' str(reconstructed)
#'
#' # ============================================================
#' # Example 4: Verifying inverse transform accuracy
#' # ============================================================
#'
#' data <- sample_toydata()
#' transformer <- data_transformer$new()
#' transformer$fit(data)
#'
#' # Round-trip transformation
#' transformed <- transformer$transform(data)
#' reconstructed <- transformer$inverse_transform(transformed)
#'
#' # Check reconstruction error (should be very small)
#' max_error <- max(abs(as.matrix(data) - as.matrix(reconstructed)))
#' cat("Maximum reconstruction error:", max_error, "\n")
#' }
data_transformer <- R6::R6Class(
  "data_transformer",
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Create a new data_transformer object
    initialize = function() {
      self$last_snap_stats <- list()
      NULL
    },

    #' @description Fit parameters for a continuous column (internal method)
    #' @param column Column name or index
    #' @param data Column data as a single-column matrix
    #' @param mode_specific Whether to use GMM-based normalization. Kept for
    #'   backward compatibility; ignored when `continuous_transform` is given.
    #' @param n_modes Number of GMM components
    #' @param continuous_transform One of `"zscore"`, `"mode_specific"` or
    #'   `"quantile"`. When NULL it is derived from `mode_specific`.
    #' @param n_quantiles Size of the empirical quantile reference table used by
    #'   the `"quantile"` transform.
    fit_continuous = function(column = NULL, data = NULL, mode_specific = FALSE,
                              n_modes = 10, continuous_transform = NULL,
                              n_quantiles = 1000) {
      if (is.null(continuous_transform)) {
        continuous_transform <- if (isTRUE(mode_specific)) "mode_specific" else "zscore"
      }
      data <- data[, 1]
      data <- data[!is.na(data)]

      if (continuous_transform == "quantile") {
        # Gaussian quantile transform (Kotelnikov et al. 2023; TabDDPM/TabSyn).
        # The forward map is the empirical CDF followed by qnorm(); it is stored
        # as a table of at most `n_quantiles` empirical quantiles so that both
        # directions are a linear interpolation over the same monotone table.
        n_quantiles <- max(2L, as.integer(n_quantiles))
        probs <- seq(0, 1, length.out = n_quantiles)

        if (length(data) == 0) {
          # Degenerate column: behave like a constant column at 0.
          return(list(
            name = column,
            mode_specific = FALSE,
            continuous_transform = "quantile",
            constant = TRUE,
            constant_value = NA_real_,
            output_info = list(1, "quantile"),
            output_dimensions = 1
          ))
        }

        qs <- as.numeric(stats::quantile(data, probs = probs, names = FALSE,
                                         na.rm = TRUE, type = 7))

        # Collapse duplicated quantile values (heavy ties, discrete-ish columns)
        # to a strictly increasing table; the probability of a repeated value is
        # the mean of the probabilities that map to it.
        agg <- stats::aggregate(list(p = probs), by = list(v = qs), FUN = mean)
        ord <- order(agg$v)
        values_u <- as.numeric(agg$v)[ord]
        probs_u <- as.numeric(agg$p)[ord]

        if (length(values_u) < 2) {
          # Constant column: no interpolation possible.
          return(list(
            name = column,
            mode_specific = FALSE,
            continuous_transform = "quantile",
            constant = TRUE,
            constant_value = values_u[1],
            output_info = list(1, "quantile"),
            output_dimensions = 1
          ))
        }

        return(list(
          name = column,
          mode_specific = FALSE,
          continuous_transform = "quantile",
          constant = FALSE,
          quantile_values = values_u,
          quantile_probs = probs_u,
          n_quantiles = n_quantiles,
          output_info = list(1, "quantile"),
          output_dimensions = 1
        ))
      }

      if (continuous_transform == "mode_specific") {
        # Fit GMM using EM algorithm
        gmm_result <- private$fit_gmm(data, n_modes)

        return(
          list(
            name = column,
            mode_specific = TRUE,
            continuous_transform = "mode_specific",
            n_modes = gmm_result$n_modes,
            means = gmm_result$means,
            stds = gmm_result$stds,
            weights = gmm_result$weights,
            output_info = list(gmm_result$n_modes + 1, "mode_specific"),
            output_dimensions = gmm_result$n_modes + 1
          )
        )
      } else {
        # Standard z-transformation
        mean <- mean(data, na.rm = TRUE)
        std <- sd(data, na.rm = TRUE)
        if (std == 0) std <- 1  # Avoid division by zero

        return(
          list(
            name = column,
            mode_specific = FALSE,
            continuous_transform = "zscore",
            mean = mean,
            std = std,
            output_info = list(1, "linear"),
            output_dimensions = 1
          )
        )
      }
    },

    #' @description Fit parameters for a discrete/categorical column (internal method)
    #' @param column Column name or index
    #' @param data Column data as a single-column matrix
    #' @param categorical_encoding One of `"onehot"` (default), `"cossin"` or
    #'   `"embedding_ready"`.
    fit_discrete = function(column = NULL, data = NULL, categorical_encoding = "onehot") {
      column <- column
      data <- factor(data[, 1])
      levs <- levels(data)
      categories <- length(levs)

      if (categorical_encoding == "cossin") {
        # TabRep (Hsi et al. 2025): place the K levels equidistantly on the
        # unit circle. Two output dimensions regardless of cardinality.
        return(list(
          name = column,
          levs = levs,
          categorical_encoding = "cossin",
          K = categories,
          output_info = list(2, "cossin"),
          output_dimensions = 2
        ))
      }

      if (categorical_encoding == "embedding_ready") {
        # Data path only: the column stays a 1..K integer index so that a later
        # learned-embedding generator can consume it. There is no generator head
        # for this type.
        return(list(
          name = column,
          levs = levs,
          categorical_encoding = "embedding_ready",
          K = categories,
          output_info = list(1, "index"),
          output_dimensions = 1
        ))
      }

      return(list(
        name = column,
        levs = levs,
        categorical_encoding = "onehot",
        output_info = list(categories, "softmax"),
        output_dimensions = categories
      ))

    },
    #' @description
    #' Fit the transformer to learn normalization parameters from data.
    #'
    #' This method analyzes each column in the data and stores the parameters
    #' needed for transformation (means, standard deviations, category levels, etc.).
    #' Must be called before `transform()` or `inverse_transform()`.
    #'
    #' @param data A data.frame, matrix, or array containing the training data.
    #'   Column names are preserved and used for tracking transformations.
    #' @param discrete_columns Character or integer vector specifying which columns
    #'   contain categorical/discrete values. These columns will be one-hot encoded.
    #'   Can be column names (character) or column indices (integer). Columns not
    #'   listed here are treated as continuous. Defaults to NULL (all continuous).
    #' @param mode_specific Logical. If TRUE, use mode-specific normalization (GMM)
    #'   for continuous columns. This fits a Gaussian Mixture Model to each
    #'   continuous column and normalizes values within their assigned mode.
    #'   Recommended for columns with multi-modal or skewed distributions.
    #'   Defaults to FALSE (standard z-score normalization).
    #' @param n_modes Integer. Maximum number of Gaussian components for GMM fitting.
    #'   The actual number may be lower if modes with weight < 0.01 are pruned.
    #'   Only used when `mode_specific = TRUE`. Defaults to 10.
    #' @param continuous_transform Character, one of `"zscore"`,
    #'   `"mode_specific"` or `"quantile"`. Selects the representation of the
    #'   continuous columns and supersedes `mode_specific`. When `NULL` (the
    #'   default) it is derived from `mode_specific` so that existing code keeps
    #'   its behaviour. If both are supplied and disagree,
    #'   `continuous_transform` wins and a warning is emitted.
    #'   `"quantile"` maps each column through its empirical CDF and `qnorm()`,
    #'   the Gaussian quantile transform used by TabDDPM and TabSyn
    #'   (Kotelnikov et al. 2023). It produces one output column with
    #'   `output_info` type `"quantile"`.
    #' @param categorical_encoding Character, one of `"onehot"` (the default and
    #'   the previous behaviour, `output_info` type `"softmax"`), `"cossin"`
    #'   (TabRep, Hsi et al. 2025: level `c` of `K` becomes
    #'   `(cos(2*pi*(c-1)/K), sin(2*pi*(c-1)/K))`, two output columns,
    #'   `output_info` type `"cossin"`) or `"embedding_ready"` (a single integer
    #'   index column, `output_info` type `"index"`, intended as the data path
    #'   for a learned-embedding generator; no generator head exists for it).
    #' @param n_quantiles Integer. Size of the empirical quantile reference table
    #'   stored per column when `continuous_transform = "quantile"`.
    #'   Defaults to 1000.
    #'
    #' @return The transformer object (invisibly), allowing method chaining.
    #'
    #' @examples
    #' # Standard normalization
    #' data <- sample_toydata()
    #' transformer <- data_transformer$new()
    #' transformer$fit(data)
    #'
    #' # Mode-specific normalization
    #' transformer$fit(data, mode_specific = TRUE, n_modes = 10)
    #'
    #' # With categorical columns
    #' transformer$fit(data, discrete_columns = c("category"))
    fit = function(data, discrete_columns = NULL, mode_specific = FALSE, n_modes = 10,
                   continuous_transform = NULL, categorical_encoding = "onehot",
                   n_quantiles = 1000) {
      valid_ct <- c("zscore", "mode_specific", "quantile")
      if (is.null(continuous_transform)) {
        continuous_transform <- if (isTRUE(mode_specific)) "mode_specific" else "zscore"
      } else {
        if (length(continuous_transform) != 1 ||
            !continuous_transform %in% valid_ct) {
          stop(sprintf("continuous_transform must be one of: %s",
                       paste(valid_ct, collapse = ", ")))
        }
        implied <- isTRUE(mode_specific)
        if (implied != (continuous_transform == "mode_specific")) {
          warning(sprintf(
            paste0("continuous_transform = '%s' conflicts with mode_specific = %s; ",
                   "continuous_transform wins (mode_specific is deprecated)."),
            continuous_transform, implied))
        }
      }
      mode_specific <- (continuous_transform == "mode_specific")

      valid_ce <- c("onehot", "cossin", "embedding_ready")
      if (length(categorical_encoding) != 1 ||
          !categorical_encoding %in% valid_ce) {
        stop(sprintf("categorical_encoding must be one of: %s",
                     paste(valid_ce, collapse = ", ")))
      }

      self$output_info <- list()
      self$output_dimensions <- 0
      self$mode_specific <- mode_specific
      self$n_modes <- n_modes
      self$continuous_transform <- continuous_transform
      self$categorical_encoding <- categorical_encoding
      self$n_quantiles <- n_quantiles
      self$last_snap_stats <- list()

      self$meta <- list()

      if (is.null(colnames(data))) {
        col_ids <- 1:ncol(data)
      } else {
        col_ids <- colnames(data)
      }

      for (column in col_ids) {
        column_data <- data[, which(column == col_ids), drop = F]
        if (column %in% discrete_columns) {
          meta <- self$fit_discrete(column, column_data, categorical_encoding)
        } else {
          meta <- self$fit_continuous(column, column_data, mode_specific, n_modes,
                                      continuous_transform, n_quantiles)
        }
        self$output_info[[length(self$output_info) + 1]] <-
          meta$output_info
        self$output_dimensions <-
          self$output_dimensions + meta$output_dimensions
        self$meta[[length(self$meta) + 1]] <- meta
      }
      invisible(self)
    },

    #' @description Transform a continuous column (internal method)
    #' @param column_meta Metadata for this column from fit_continuous
    #' @param data Vector of values to transform
    transform_continuous = function(column_meta, data) {
      if (identical(column_meta$continuous_transform, "quantile")) {
        # Empirical CDF position by linear interpolation over the stored
        # reference table, clipped away from 0/1 so that qnorm() stays finite.
        data <- as.numeric(data)
        out <- rep(NA_real_, length(data))
        ok <- !is.na(data)
        if (any(ok)) {
          if (isTRUE(column_meta$constant)) {
            # A constant column carries no rank information; map it to the
            # centre of the normal.
            out[ok] <- 0
          } else {
            p <- stats::approx(x = column_meta$quantile_values,
                               y = column_meta$quantile_probs,
                               xout = data[ok], rule = 2)$y
            p <- pmin(pmax(p, 1e-4), 1 - 1e-4)
            out[ok] <- stats::qnorm(p)
          }
        }
        return(matrix(out, ncol = 1))
      }

      if (isTRUE(column_meta$mode_specific)) {
        # Mode-specific transformation
        n_modes <- column_meta$n_modes
        means <- column_meta$means
        stds <- column_meta$stds
        weights <- column_meta$weights

        data <- as.numeric(data)
        n <- length(data)
        # Output: n_modes columns for mode probabilities + 1 column for normalized value
        result <- matrix(0, nrow = n, ncol = n_modes + 1)

        # Vectorised over rows: the only loop left runs over the (at most
        # n_modes) mixture components, which is orders of magnitude cheaper
        # than the previous per-row loop.
        ok <- !is.na(data)
        if (any(!ok)) {
          result[!ok, ] <- NA
        }

        if (any(ok)) {
          x <- data[ok]
          rows <- which(ok)

          # Log of the unnormalised mode responsibilities. Normalising the
          # responsibilities does not change the argmax, so it is skipped.
          # Working in log space is numerically equivalent to
          # `weights * dnorm(...)` but does not underflow to an all-zero row
          # for values sitting far in the tail of every component.
          log_probs <- matrix(0, nrow = length(x), ncol = n_modes)
          for (k in seq_len(n_modes)) {
            log_probs[, k] <- log(weights[k]) +
              stats::dnorm(x, mean = means[k], sd = stds[k], log = TRUE)
          }

          # Assign each value to its most likely mode. max.col() with
          # ties.method = "first" reproduces which.max(), which also returns
          # the first maximum.
          mode_idx <- max.col(log_probs, ties.method = "first")

          # One-hot encoding of the assigned mode (first n_modes columns)
          result[cbind(rows, mode_idx)] <- 1

          # Normalize within the selected mode (last column).
          # Clip to [-4, 4] standard deviations as in CTGAN, then scale to
          # roughly [-1, 1].
          normalized <- (x - means[mode_idx]) / stds[mode_idx]
          normalized <- pmax(-4, pmin(4, normalized))
          result[rows, n_modes + 1] <- normalized / 4
        }

        return(result)
      } else {
        # Standard z-transformation
        mean <- column_meta$mean
        std <- column_meta$std

        z <- (data - mean) / std

        return(z)
      }
    },

    #' @description Transform a discrete column to one-hot encoding (internal method)
    #' @param column_meta Metadata for this column from fit_discrete
    #' @param data Vector of values to transform
    transform_discrete = function(column_meta, data) {
      encoding <- column_meta$categorical_encoding
      if (is.null(encoding)) encoding <- "onehot"

      if (encoding %in% c("cossin", "embedding_ready")) {
        idx <- as.integer(factor(data, levels = column_meta$levs))
        K <- column_meta$K

        if (encoding == "embedding_ready") {
          out <- matrix(as.numeric(idx), ncol = 1)
          colnames(out) <- as.character(column_meta$name)
          return(out)
        }

        angle <- 2 * pi * (idx - 1) / K
        out <- cbind(cos(angle), sin(angle))
        out[is.na(idx), ] <- NA
        colnames(out) <- paste0(column_meta$name, c("_cos", "_sin"))
        return(out)
      }

      oh <- model.matrix( ~ 0 + factor(data, levels = column_meta$levs))
      colnames(oh) <- column_meta$levs
      oh_na <- array(NA, dim = c(length(data), ncol(oh)))
      oh_na[!is.na(data), ] <- oh
      colnames(oh_na) <- colnames(oh)
      return(oh_na)
    },
    #' @description
    #' Transform data from original format to normalized format for GAN training.
    #'
    #' Applies the learned transformations to convert data into a format suitable
    #' for neural network training:
    #' - Continuous columns: z-score normalization or mode-specific normalization
    #' - Categorical columns: one-hot encoding
    #'
    #' The transformer must be fitted before calling this method.
    #'
    #' @param data A data.frame, matrix, or array with the same columns as the
    #'   data used for fitting. Column order and names must match.
    #'
    #' @return A numeric matrix with transformed data. The number of columns
    #'   depends on the transformation:
    #'   - Standard normalization: same number of columns as input
    #'   - Mode-specific: (n_modes + 1) columns per continuous column
    #'   - Categorical: one column per category level
    #'
    #'   Use `transformer$output_dimensions` to check the total output columns.
    #'
    #' @examples
    #' data <- sample_toydata()
    #' transformer <- data_transformer$new()
    #' transformer$fit(data)
    #' transformed_data <- transformer$transform(data)
    #' cat("Output dimensions:", dim(transformed_data))
    transform = function(data) {
      values <- list()
      for (meta in self$meta) {
        column_data <- data[, meta$name]
        if ("levs" %in% names(meta)) {
          values[[length(values) + 1]] <-
            self$transform_discrete(meta, column_data)
        } else {
          values[[length(values) + 1]] <-
            self$transform_continuous(meta, column_data)
        }
      }

      return(do.call(cbind, values))

    },

    #' @description Inverse transform a continuous column (internal method)
    #' @param meta Metadata for this column
    #' @param data Transformed data to inverse transform
    inverse_transform_continuous = function(meta, data) {
      if (identical(meta$continuous_transform, "quantile")) {
        z <- as.numeric(data)
        out <- rep(NA_real_, length(z))
        ok <- !is.na(z)
        if (any(ok)) {
          if (isTRUE(meta$constant)) {
            out[ok] <- meta$constant_value
          } else {
            p <- stats::pnorm(z[ok])
            p <- pmin(pmax(p, 1e-4), 1 - 1e-4)
            out[ok] <- stats::approx(x = meta$quantile_probs,
                                     y = meta$quantile_values,
                                     xout = p, rule = 2)$y
          }
        }
        return(out)
      }

      if (isTRUE(meta$mode_specific)) {
        # Mode-specific inverse transformation
        n_modes <- meta$n_modes
        means <- meta$means
        stds <- meta$stds

        # data has n_modes columns for mode selection + 1 for normalized value
        if (is.null(dim(data))) {
          # Single row case
          mode_probs <- data[1:n_modes]
          normalized_value <- data[n_modes + 1]

          # Select mode with highest probability
          mode_idx <- which.max(mode_probs)

          # Denormalize: reverse the /4 scaling, then apply mode mean/std
          value <- normalized_value * 4 * stds[mode_idx] + means[mode_idx]
          return(value)
        } else {
          mode_probs <- data[, 1:n_modes, drop = FALSE]
          normalized_values <- data[, n_modes + 1]

          # Select mode with highest probability for each row
          mode_indices <- max.col(mode_probs, ties.method = "first")

          # Denormalize every value at once using its selected mode
          # (vectorised replacement for the former per-row loop).
          values <- as.numeric(normalized_values) * 4 * stds[mode_indices] +
            means[mode_indices]

          return(values)
        }
      } else {
        # Standard inverse z-transformation
        mean <- meta$mean
        std <- meta$std

        u <- data

        column <- u * std + mean

        return(column)
      }
    },

    #' @description Inverse transform a discrete column from one-hot (internal method)
    #' @param meta Metadata for this column
    #' @param data One-hot encoded data to inverse transform
    inverse_transform_discrete = function(meta, data) {
      levs <- meta$levs
      encoding <- meta$categorical_encoding
      if (is.null(encoding)) encoding <- "onehot"

      if (encoding == "cossin") {
        K <- meta$K
        angle <- atan2(as.numeric(data[, 2]), as.numeric(data[, 1]))
        level_angles <- 2 * pi * (seq_len(K) - 1) / K

        # Circular distance to every level angle, wrapped to [0, pi].
        d <- abs(outer(angle, level_angles, "-")) %% (2 * pi)
        d <- pmin(d, 2 * pi - d)

        max_index <- max.col(-d, ties.method = "first")
        snap <- d[cbind(seq_along(angle), max_index)]

        # Snap distance in units of the inter-level angle (2*pi/K); for K = 1
        # every point maps to the single level and the full circle is the unit.
        snap <- snap / (2 * pi / K)
        self$last_snap_stats[[as.character(meta$name)]] <- list(
          mean = mean(snap, na.rm = TRUE),
          q90 = as.numeric(stats::quantile(snap, probs = 0.9, na.rm = TRUE,
                                           names = FALSE)),
          n = sum(!is.na(snap)),
          K = K
        )

        column <- levs[max_index]
        numeric_levs <- suppressWarnings(as.numeric(levs))
        if (!any(is.na(numeric_levs))) column <- as.numeric(column)
        return(column)
      }

      if (encoding == "embedding_ready") {
        idx <- round(as.numeric(data[, 1]))
        idx <- pmin(pmax(idx, 1), length(levs))
        column <- levs[idx]
        numeric_levs <- suppressWarnings(as.numeric(levs))
        if (!any(is.na(numeric_levs))) column <- as.numeric(column)
        return(column)
      }

      # Get the index of the maximum value in each row (the selected category)
      max_index <- max.col(data, ties.method = "random")

      # Map indices to category levels
      column <- levs[max_index]

      # Try to convert to numeric if the levels are numeric
      # Otherwise keep as character
      numeric_levs <- suppressWarnings(as.numeric(levs))
      if (!any(is.na(numeric_levs))) {
        # All levels are numeric, convert result to numeric
        column <- as.numeric(column)
      }

      return(column)
    },
    #' @description
    #' Inverse transform data from normalized format back to original scale.
    #'
    #' Reverses the transformations applied by `transform()`:
    #' - Continuous columns: denormalized using stored means/stds
    #' - Mode-specific: selects mode with highest probability, then denormalizes
    #' - Categorical columns: selects category with highest value (argmax)
    #'
    #' This is typically used to convert GAN-generated samples back to the
    #' original data format for analysis and use.
    #'
    #' @param data A numeric matrix in the transformed format. Must have the
    #'   same number of columns as `transformer$output_dimensions`.
    #'
    #' @return A data.frame with columns in the original format:
    #'   - Continuous columns as numeric
    #'   - Categorical columns as character (or numeric if original levels were numeric)
    #'
    #' @examples
    #' data <- sample_toydata()
    #' transformer <- data_transformer$new()
    #' transformer$fit(data)
    #'
    #' # Round-trip transformation
    #' transformed_data <- transformer$transform(data)
    #' reconstructed_data <- transformer$inverse_transform(transformed_data)
    #'
    #' # Use with GAN output
    #' # synthetic_raw <- trained_gan$generator(noise)
    #' # synthetic_data <- transformer$inverse_transform(as_array(synthetic_raw))
    inverse_transform = function(data) {
      start <- 1
      # Snap diagnostics are reported for the most recent call only.
      self$last_snap_stats <- list()
      output <- list()
      column_names <- c()
      for (meta in self$meta) {
        dimensions <- meta$output_dimensions
        columns_data <- data[, start:(start + dimensions - 1), drop = FALSE]

        if ("levs" %in% names(meta)) {
          inverted <- self$inverse_transform_discrete(meta, columns_data)
        } else {
          inverted <- self$inverse_transform_continuous(meta, columns_data)
        }
        output[[meta$name]] <- inverted
        column_names <- c(column_names, meta$name)
        start <- start + dimensions
      }

      # Use data.frame to preserve column types (numeric vs character)
      result <- as.data.frame(output, stringsAsFactors = FALSE)
      colnames(result) <- column_names

      return(result)
    }
  ),
  private = list(
    # Fit a Gaussian Mixture Model using EM algorithm
    # @param data Numeric vector of data points
    # @param n_modes Maximum number of modes to fit
    # @return List with means, stds, weights, and actual n_modes used
    fit_gmm = function(data, n_modes) {
      data <- data[!is.na(data)]
      n <- length(data)

      # Reduce n_modes if we don't have enough data
      n_modes <- min(n_modes, floor(n / 2))
      if (n_modes < 1) n_modes <- 1

      # Initialize using k-means
      if (n_modes == 1) {
        means <- mean(data)
        stds <- max(sd(data), 1e-6)
        weights <- 1
      } else {
        # Use k-means for initialization
        km <- tryCatch({
          stats::kmeans(data, centers = n_modes, nstart = 10)
        }, error = function(e) {
          # Fall back to quantile-based initialization
          NULL
        })

        if (!is.null(km)) {
          means <- as.numeric(km$centers)
          # Compute std for each cluster
          stds <- numeric(n_modes)
          weights <- numeric(n_modes)
          for (k in 1:n_modes) {
            cluster_data <- data[km$cluster == k]
            if (length(cluster_data) > 1) {
              stds[k] <- sd(cluster_data)
            } else {
              stds[k] <- sd(data) / n_modes
            }
            weights[k] <- length(cluster_data) / n
          }
        } else {
          # Quantile-based initialization
          quantiles <- stats::quantile(data, probs = seq(0, 1, length.out = n_modes + 2)[-c(1, n_modes + 2)])
          means <- as.numeric(quantiles)
          stds <- rep(sd(data) / sqrt(n_modes), n_modes)
          weights <- rep(1/n_modes, n_modes)
        }
      }

      # Ensure minimum std to avoid numerical issues
      stds <- pmax(stds, 1e-6)

      # Run EM algorithm for a few iterations to refine
      for (iter in 1:20) {
        # E-step: compute responsibilities
        resp <- matrix(0, nrow = n, ncol = n_modes)
        for (k in 1:n_modes) {
          resp[, k] <- weights[k] * stats::dnorm(data, mean = means[k], sd = stds[k])
        }
        resp_sum <- rowSums(resp)
        resp_sum[resp_sum == 0] <- 1e-10
        resp <- resp / resp_sum

        # M-step: update parameters
        Nk <- colSums(resp)
        Nk[Nk == 0] <- 1e-10

        new_weights <- Nk / n
        new_means <- colSums(resp * data) / Nk

        new_stds <- numeric(n_modes)
        for (k in 1:n_modes) {
          diff_sq <- (data - new_means[k])^2
          new_stds[k] <- sqrt(sum(resp[, k] * diff_sq) / Nk[k])
        }
        new_stds <- pmax(new_stds, 1e-6)

        # Check for convergence
        if (max(abs(new_means - means)) < 1e-6 && max(abs(new_stds - stds)) < 1e-6) {
          break
        }

        means <- new_means
        stds <- new_stds
        weights <- new_weights
      }

      # Remove modes with very low weight
      valid_modes <- weights > 0.01
      if (sum(valid_modes) < n_modes && sum(valid_modes) >= 1) {
        means <- means[valid_modes]
        stds <- stds[valid_modes]
        weights <- weights[valid_modes]
        weights <- weights / sum(weights)  # Renormalize
        n_modes <- sum(valid_modes)
      }

      return(list(
        means = means,
        stds = stds,
        weights = weights,
        n_modes = n_modes
      ))
    }
  )
)
