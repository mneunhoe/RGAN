#' @title Data Transformer
#'
#' @description Provides a class to transform data for RGAN.
#'   Method `$new()` initializes a new transformer, method `$fit(data)` learns
#'   the parameters for the transformation from data (e.g. means and sds).
#'   Methods `$transform()` and `$inverse_transform()` can be used to transform
#'   and back transform a data set based on the learned parameters.
#'
#'   DataTransformer supports two normalization methods for continuous columns:
#'   \itemize{
#'     \item \strong{Standard (default):} z-transformation using mean and standard deviation.
#'     \item \strong{Mode-specific:} Fits a Gaussian Mixture Model (GMM) to handle
#'       multi-modal distributions. Each value is normalized within its assigned mode,
#'       and a mode indicator is included. This is the approach used by CTGAN and
#'       significantly improves GAN performance on columns with skewed or multi-modal
#'       distributions.
#'   }
#'
#'   Categorical features are one-hot encoded.
#'
#' @return A class to transform (normalize or one hot encode) tabular data for RGAN
#' @export
#' @examples
#' \dontrun{
#' # Before running the first time the torch backend needs to be installed
#' torch::install_torch()
#' # Load data
#' data <- sample_toydata()
#' # Build new transformer
#' transformer <- data_transformer$new()
#' # Fit transformer to data (standard normalization)
#' transformer$fit(data)
#' # Transform data and store as new object
#' transformed_data <-  transformer$transform(data)
#'
#' # Or use mode-specific normalization for better handling of multi-modal data
#' transformer_gmm <- data_transformer$new()
#' transformer_gmm$fit(data, mode_specific = TRUE, n_modes = 10)
#' transformed_data_gmm <- transformer_gmm$transform(data)
#'
#' # Train the default GAN
#' trained_gan <- gan_trainer(transformed_data)
#' # Sample synthetic data from the trained GAN
#' synthetic_data <- sample_synthetic_data(trained_gan, transformer)
#' # Plot the results
#' GAN_update_plot(data = data,
#' synth_data = synthetic_data,
#' main = "Real and Synthetic Data after Training")
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
    fit_discrete = function(column = NULL, data = NULL) {
      column <- column
      data <- factor(data[, 1])
      levs <- levels(data)
      categories <- length(levs)

      return(list(
        name = column,
        levs = levs,
        output_info = list(categories, "softmax"),
        output_dimensions = categories
      ))

    },
    #' @description
    #' Fit a data_transformer to data.
    #' @param data The data set to transform
    #' @param discrete_columns Column ids for columns with discrete/nominal values to be one hot encoded.
    #' @param mode_specific If TRUE, use mode-specific normalization (GMM) for continuous columns.
    #'   This helps with multi-modal distributions. Defaults to FALSE.
    #' @param n_modes Number of modes (Gaussian components) to fit for mode-specific normalization.
    #'   Defaults to 10. Only used when mode_specific = TRUE.
    #' @examples
    #' data <- sample_toydata()
    #' transformer <- data_transformer$new()
    #' transformer$fit(data)
    #' # Or with mode-specific normalization
    #' transformer$fit(data, mode_specific = TRUE, n_modes = 10)
    fit = function(data, discrete_columns = NULL, mode_specific = FALSE, n_modes = 10) {
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
        # Standard z-transformation
        mean <- column_meta$mean
        std <- column_meta$std

        z <- (data - mean) / std

        return(z)
      }
    },
    transform_discrete = function(column_meta, data) {
      oh <- model.matrix( ~ 0 + factor(data, levels = column_meta$levs))
      colnames(oh) <- column_meta$levs
      oh_na <- array(NA, dim = c(length(data), ncol(oh)))
      oh_na[!is.na(data), ] <- oh
      colnames(oh_na) <- colnames(oh)
      return(oh_na)
    },
    #' @description
    #' Transform data using a fitted data_transformer. (From original format to transformed format.)
    #' @param data The data set to transform
    #' @examples
    #' data <- sample_toydata()
    #' transformer <- data_transformer$new()
    #' transformer$fit(data)
    #' transformed_data <- transformer$transform(data)
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

        return(column)
      }
    },
    inverse_transform_discrete = function(meta, data) {
      levs <- meta$levs

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
    #' Inverse Transform data using a fitted data_transformer. (From transformed format to original format.)
    #' @param data The data set to transform
    #' @examples
    #' data <- sample_toydata()
    #' transformer <- data_transformer$new()
    #' transformer$fit(data)
    #' transformed_data <- transformer$transform(data)
    #' reconstructed_data <- transformer$inverse_transform(transformed_data)
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
    #' Fit a Gaussian Mixture Model using EM algorithm
    #' @param data Numeric vector of data points
    #' @param n_modes Maximum number of modes to fit
    #' @return List with means, stds, weights, and actual n_modes used
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
