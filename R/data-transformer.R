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
#' ## Categorical Columns
#'
#' Categorical (discrete) columns are one-hot encoded. Each category becomes a
#' separate binary column. The inverse transform selects the category with the
#' highest value (argmax).
#'
#' ## Fields
#'
#' After fitting, the transformer stores:
#' \describe{
#'   \item{meta}{List of metadata for each column (means, stds, levels, etc.)}
#'   \item{output_info}{List describing the output structure for each column}
#'   \item{output_dimensions}{Total number of columns in transformed data}
#'   \item{mode_specific}{Whether mode-specific normalization was used}
#'   \item{n_modes}{Number of GMM modes (if mode_specific = TRUE)}
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
      NULL
    },

    #' @description Fit parameters for a continuous column (internal method)
    #' @param column Column name or index
    #' @param data Column data as a single-column matrix
    #' @param mode_specific Whether to use GMM-based normalization
    #' @param n_modes Number of GMM components
    fit_continuous = function(column = NULL, data = NULL, mode_specific = FALSE, n_modes = 10) {
      data <- data[, 1]
      data <- data[!is.na(data)]

      if (mode_specific) {
        # Fit GMM using EM algorithm
        gmm_result <- private$fit_gmm(data, n_modes)

        return(
          list(
            name = column,
            mode_specific = TRUE,
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
    fit_discrete = function(column = NULL, data = NULL) {
      column <- column
      values <- if (is.data.frame(data)) data[[1]] else data[, 1]
      # Remember the column's class so inverse_transform() can restore it
      # exactly; guessing from the level strings turns codes such as "01"
      # into the number 1.
      original_class <- class(values)[1]
      data <- factor(values)
      levs <- levels(data)
      categories <- length(levs)

      return(list(
        name = column,
        levs = levs,
        original_class = original_class,
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
    #'   The actual number may be lower: modes backed by fewer than
    #'   `max(10, 0.01 * n)` records are pruned, and spreads are floored at 1\%
    #'   of the column's standard deviation, so that no mode reproduces a single
    #'   record. This limits memorisation but is not a privacy guarantee; see
    #'   `fit_public()`.
    #'   Only used when `mode_specific = TRUE`. Defaults to 10.
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
    #' # With categorical columns (by name or by index)
    #' mixed <- data.frame(x = rnorm(100), category = sample(c("a", "b"), 100, TRUE))
    #' transformer$fit(mixed, discrete_columns = c("category"))
    #' transformer$fit(mixed, discrete_columns = 2)
    fit = function(data, discrete_columns = NULL, mode_specific = FALSE, n_modes = 10) {
      # Fitted on the data: its statistics are not differentially private.
      self$public <- FALSE
      self$output_info <- list()
      self$output_dimensions <- 0
      self$mode_specific <- mode_specific
      self$n_modes <- n_modes

      self$meta <- list()

      if (is.null(colnames(data))) {
        col_ids <- 1:ncol(data)
      } else {
        col_ids <- colnames(data)
      }

      # discrete_columns may be given as names or as column indices. Indices
      # are mapped to names when the data has names; matching them against
      # the names directly would silently treat every column as continuous.
      if (!is.null(discrete_columns)) {
        if (is.numeric(discrete_columns)) {
          bad <- discrete_columns[discrete_columns < 1 |
                                    discrete_columns > ncol(data) |
                                    discrete_columns != round(discrete_columns)]
          if (length(bad) > 0) {
            stop(sprintf(
              "discrete_columns indices out of range (data has %d columns): %s",
              ncol(data), paste(bad, collapse = ", ")
            ), call. = FALSE)
          }
          discrete_columns <- col_ids[discrete_columns]
        } else {
          unknown <- setdiff(discrete_columns, col_ids)
          if (length(unknown) > 0) {
            stop(sprintf(
              "discrete_columns not found in the data: %s",
              paste(unknown, collapse = ", ")
            ), call. = FALSE)
          }
        }
      }

      for (column in col_ids) {
        column_data <- data[, which(column == col_ids), drop = F]
        if (column %in% discrete_columns) {
          meta <- self$fit_discrete(column, column_data)
        } else {
          meta <- self$fit_continuous(column, column_data, mode_specific, n_modes)
        }
        self$output_info[[length(self$output_info) + 1]] <-
          meta$output_info
        self$output_dimensions <-
          self$output_dimensions + meta$output_dimensions
        self$meta[[length(self$meta) + 1]] <- meta
      }
      invisible(self)
    },

    #' @description
    #' Fit the transformer from public information only, without looking at
    #' any data.
    #'
    #' A transformer fitted with `fit()` stores statistics of the data it was
    #' fitted on (means, standard deviations, category levels, mixture
    #' components). Because the same transformer is needed to decode synthetic
    #' samples, those statistics are released with a model, outside any
    #' differential-privacy guarantee. `fit_public()` builds the transformer
    #' from a schema of public bounds and category domains instead, so it
    #' spends no privacy budget and reveals nothing about the records.
    #'
    #' Continuous columns are clipped to their bounds and scaled linearly into
    #' `[-1, 1]`; the inverse transform clips back to the bounds. Categorical
    #' columns are one-hot encoded over the given levels. Mode-specific
    #' normalisation is not available, since it needs a mixture fitted to data.
    #'
    #' @param columns A named list, in the order of the data's columns, whose
    #'   elements are built with [public_continuous()] or [public_categorical()].
    #'   The names must match the data's column names.
    #'
    #' @return The transformer object (invisibly).
    #'
    #' @examples
    #' transformer <- data_transformer$new()
    #' transformer$fit_public(list(
    #'   age = public_continuous(18, 100),
    #'   sex = public_categorical(c("F", "M"))
    #' ))
    #' transformer$transform(data.frame(age = c(30, 120), sex = c("M", "F")))
    fit_public = function(columns) {
      if (!is.list(columns) || length(columns) == 0 ||
          is.null(names(columns)) || any(names(columns) == "")) {
        stop("columns must be a non-empty named list of public_continuous() ",
             "or public_categorical() specifications.", call. = FALSE)
      }
      self$public <- TRUE
      self$mode_specific <- FALSE
      self$n_modes <- NULL
      self$output_info <- list()
      self$output_dimensions <- 0
      self$meta <- list()

      for (column in names(columns)) {
        spec <- columns[[column]]
        if (inherits(spec, "rgan_public_continuous")) {
          meta <- list(
            name = column,
            mode_specific = FALSE,
            mean = (spec$lower + spec$upper) / 2,
            std = (spec$upper - spec$lower) / 2,
            bounds = c(spec$lower, spec$upper),
            output_info = list(1, "linear"),
            output_dimensions = 1
          )
        } else if (inherits(spec, "rgan_public_categorical")) {
          meta <- list(
            name = column,
            levs = spec$levels,
            original_class = spec$class,
            output_info = list(length(spec$levels), "softmax"),
            output_dimensions = length(spec$levels)
          )
        } else {
          stop(sprintf(paste0(
            "Column '%s': use public_continuous() or public_categorical() to ",
            "describe it."), column), call. = FALSE)
        }
        self$output_info[[length(self$output_info) + 1]] <- meta$output_info
        self$output_dimensions <- self$output_dimensions + meta$output_dimensions
        self$meta[[length(self$meta) + 1]] <- meta
      }
      invisible(self)
    },

    #' @description Transform a continuous column (internal method)
    #' @param column_meta Metadata for this column from fit_continuous
    #' @param data Vector of values to transform
    transform_continuous = function(column_meta, data) {
      if (isTRUE(column_meta$mode_specific)) {
        # Mode-specific transformation
        n_modes <- column_meta$n_modes
        means <- column_meta$means
        stds <- column_meta$stds
        weights <- column_meta$weights

        n <- length(data)
        # Output: n_modes columns for mode probabilities + 1 column for normalized value
        result <- matrix(0, nrow = n, ncol = n_modes + 1)

        for (i in seq_len(n)) {
          if (is.na(data[i])) {
            result[i, ] <- NA
            next
          }

          # Compute probability of each mode for this data point
          probs <- weights * stats::dnorm(data[i], mean = means, sd = stds)
          probs <- probs / sum(probs)

          # Assign to most likely mode
          mode_idx <- which.max(probs)

          # Create one-hot encoding for mode (first n_modes columns)
          result[i, mode_idx] <- 1

          # Normalize within the selected mode (last column)
          # Clip to [-4, 4] standard deviations as in CTGAN
          normalized <- (data[i] - means[mode_idx]) / stds[mode_idx]
          normalized <- max(-4, min(4, normalized))
          # Scale to roughly [-1, 1]
          result[i, n_modes + 1] <- normalized / 4
        }

        return(result)
      } else {
        # Standard z-transformation (or, for a public schema, a linear map of
        # the clipped value into [-1, 1])
        mean <- column_meta$mean
        std <- column_meta$std

        if (!is.null(column_meta$bounds)) {
          data <- pmin(pmax(data, column_meta$bounds[1]), column_meta$bounds[2])
        }

        z <- (data - mean) / std

        return(z)
      }
    },

    #' @description Transform a discrete column to one-hot encoding (internal method)
    #' @param column_meta Metadata for this column from fit_discrete
    #' @param data Vector of values to transform
    transform_discrete = function(column_meta, data) {
      if (is.factor(data)) data <- as.character(data)
      unseen <- setdiff(unique(as.character(data[!is.na(data)])), column_meta$levs)
      if (length(unseen) > 0) {
        stop(sprintf(
          "Column '%s' contains levels not seen in fit(): %s. Refit the transformer on data that includes them.",
          column_meta$name, paste(utils::head(unseen, 10), collapse = ", ")
        ), call. = FALSE)
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

          # Denormalize each value using its selected mode
          values <- numeric(nrow(data))
          for (i in seq_len(nrow(data))) {
            mode_idx <- mode_indices[i]
            values[i] <- normalized_values[i] * 4 * stds[mode_idx] + means[mode_idx]
          }

          return(values)
        }
      } else {
        # Standard inverse z-transformation
        mean <- meta$mean
        std <- meta$std

        u <- data

        column <- u * std + mean

        if (!is.null(meta$bounds)) {
          column <- pmin(pmax(column, meta$bounds[1]), meta$bounds[2])
        }

        return(column)
      }
    },

    #' @description Inverse transform a discrete column from one-hot (internal method)
    #' @param meta Metadata for this column
    #' @param data One-hot encoded data to inverse transform
    inverse_transform_discrete = function(meta, data) {
      levs <- meta$levs

      # Get the index of the maximum value in each row (the selected category)
      max_index <- max.col(data, ties.method = "random")

      # Map indices to category levels
      column <- levs[max_index]

      # Restore the class the column had at fit() time.
      original_class <- meta$original_class
      if (is.null(original_class)) {
        # Transformers fitted before original_class was recorded: keep the
        # old behaviour of converting all-numeric levels to numbers.
        numeric_levs <- suppressWarnings(as.numeric(levs))
        if (!any(is.na(numeric_levs))) {
          column <- as.numeric(column)
        }
        return(column)
      }

      column <- switch(
        original_class,
        "numeric" = as.numeric(column),
        "integer" = as.integer(column),
        "logical" = as.logical(column),
        "factor" = factor(column, levels = levs),
        "ordered" = factor(column, levels = levs, ordered = TRUE),
        column
      )

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
    #'   - Categorical columns with the class they had when the transformer was
    #'     fitted (character, factor, numeric, integer or logical)
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

      # Hardening against single-record modes. A mode backed by only a handful
      # of records, with a spread at an absolute floor, reproduces those
      # records in every synthetic draw from it. Modes must be backed by at
      # least `min_mode_size` records (effective count), and spreads are
      # floored relative to the column's scale. This limits memorisation; it
      # is not a privacy guarantee (use fit_public() for that).
      min_mode_size <- max(10, 0.01 * n)
      data_sd <- if (n > 1) stats::sd(data) else 0
      std_floor <- max(1e-6, 0.01 * data_sd)
      n_modes <- max(1, min(n_modes, floor(n / min_mode_size)))

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
      stds <- pmax(stds, std_floor)

      # EM refinement
      run_em <- function(means, stds, weights) {
        k_modes <- length(means)
        for (iter in 1:20) {
          # E-step: compute responsibilities
          resp <- matrix(0, nrow = n, ncol = k_modes)
          for (k in 1:k_modes) {
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

          new_stds <- numeric(k_modes)
          for (k in 1:k_modes) {
            diff_sq <- (data - new_means[k])^2
            new_stds[k] <- sqrt(sum(resp[, k] * diff_sq) / Nk[k])
          }
          new_stds <- pmax(new_stds, std_floor)

          converged <- max(abs(new_means - means)) < 1e-6 &&
            max(abs(new_stds - stds)) < 1e-6
          means <- new_means
          stds <- new_stds
          weights <- new_weights
          if (converged) break
        }
        list(means = means, stds = stds, weights = weights)
      }

      fit <- run_em(means, stds, weights)

      # Remove modes backed by too few records, then refit so that the
      # remaining modes absorb their records. If none qualifies, keep the
      # heaviest mode only.
      for (pass in 1:2) {
        valid_modes <- fit$weights * n >= min_mode_size
        if (!any(valid_modes)) {
          valid_modes <- seq_along(fit$weights) == which.max(fit$weights)
        }
        if (all(valid_modes)) break
        w <- fit$weights[valid_modes]
        fit <- run_em(fit$means[valid_modes], fit$stds[valid_modes], w / sum(w))
      }
      # A final pass can still leave a light mode; drop it without refitting.
      valid_modes <- fit$weights * n >= min_mode_size
      if (!any(valid_modes)) {
        valid_modes <- seq_along(fit$weights) == which.max(fit$weights)
      }
      means <- fit$means[valid_modes]
      stds <- fit$stds[valid_modes]
      weights <- fit$weights[valid_modes] / sum(fit$weights[valid_modes])
      n_modes <- length(means)

      return(list(
        means = means,
        stds = stds,
        weights = weights,
        n_modes = n_modes
      ))
    }
  )
)


#' @title Public Specification of a Continuous Column
#'
#' @description Describes a continuous column for
#'   `data_transformer$fit_public()` by public bounds, chosen without looking at
#'   the data (e.g. from domain knowledge or a codebook). Values outside the
#'   bounds are clipped.
#'
#' @param lower,upper Finite public bounds, `lower < upper`.
#' @return An object of class `rgan_public_continuous`.
#' @export
#' @examples
#' public_continuous(0, 120)
public_continuous <- function(lower, upper) {
  if (!is.numeric(lower) || !is.numeric(upper) || length(lower) != 1 ||
      length(upper) != 1 || !is.finite(lower) || !is.finite(upper) ||
      lower >= upper) {
    stop("public_continuous() needs finite bounds with lower < upper.",
         call. = FALSE)
  }
  structure(list(lower = lower, upper = upper),
            class = "rgan_public_continuous")
}

#' @title Public Specification of a Categorical Column
#'
#' @description Describes a categorical column for
#'   `data_transformer$fit_public()` by its public domain of levels.
#'
#' @param levels The possible values, as a vector. Levels must be chosen
#'   without looking at the data; a level that occurs in the data but is not
#'   listed raises an error at `transform()`.
#' @param class The class to restore at `inverse_transform()`: one of
#'   "character" (default), "factor", "ordered", "numeric", "integer" or
#'   "logical".
#' @return An object of class `rgan_public_categorical`.
#' @export
#' @examples
#' public_categorical(c("low", "medium", "high"), class = "ordered")
public_categorical <- function(levels, class = "character") {
  class <- match.arg(class, c("character", "factor", "ordered", "numeric",
                              "integer", "logical"))
  levels <- unique(as.character(levels))
  if (length(levels) < 1 || anyNA(levels)) {
    stop("public_categorical() needs at least one non-missing level.",
         call. = FALSE)
  }
  structure(list(levels = levels, class = class),
            class = "rgan_public_categorical")
}
