#' @title Data Transformer
#'
#' @description Provides a class to transform data for RGAN.
#'   Method `$new()` initializes a new transformer, method `$fit(data)` learns
#'   the parameters for the transformation from data (e.g. means and sds).
#'   Methods `$transform()` and `$inverse_transform()` can be used to transform
#'   and back transform a data set based on the learned parameters.
#'   Currently, DataTransformer supports z-transformation (a.k.a. normalization)
#'   for numerical features/variables and one hot encoding for categorical
#'   features/variables. In your call to fit you just need to indicate which
#'   columns contain discrete features.
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
data_transformer <- R6::R6Class(
  "data_transformer",
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Create a new data_transformer object
    initialize = function() {
      NULL
    },

    fit_continuous = function(column = NULL, data = NULL) {
      data <- data[, 1]
      mean <- mean(data, na.rm = TRUE)
      std <- sd(data, na.rm = TRUE)

      return(
        list(
          name = column,
          z_transform = NULL,
          mean = mean,
          std = std,
          output_info = list(1, "linear"),
          output_dimensions = 1
        )
      )
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
    #' @examples
    #' data <- sample_toydata()
    #' transformer <- data_transformer$new()
    #' transformer$fit(data)
    fit = function(data, discrete_columns = NULL) {
      self$output_info <- list()
      self$output_dimensions <- 0

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
          meta <- self$fit_continuous(column, column_data)
        }
        self$output_info[[length(self$output_info) + 1]] <-
          meta$output_info
        self$ouput_dimensions <-
          self$output_dimensions + meta$output_dimensions
        self$meta[[length(self$meta) + 1]] <- meta
      }
      invisible(self)
    },
    transform_continuous = function(column_meta, data) {
      mean <- column_meta$mean
      std <- column_meta$std

      z <- (data - mean) / std

      return(z)
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
      mean <- meta$mean
      std <- meta$std

      u <- data

      column <- u * std + mean

      return(column)
    },
    inverse_transform_discrete = function(meta, data) {
      levs <- meta$levs
      #column <- factor(round(data) %*% 1:length(levs))
      #column <- factor(t(apply(data, 1, function(x){
      #ranks <- rank(x, ties.method = "random")
      #ranks == max(ranks)
      #})*1) %*% 1:length(levs))
      max_index <- max.col(data, ties.method = "random")
      row_col_index <-
        stack(setNames(max_index, seq_along(max_index)))
      max_matrix <-
        Matrix::sparseMatrix(
          as.numeric(row_col_index[, 2]),
          row_col_index[, 1],
          x = 1,
          dims = c(max(as.numeric(row_col_index[, 2])), length(levs))
        )

      column <- factor(as.matrix(max_matrix) %*% 1:length(levs))
      levels(column) <- levs
      column <- as.numeric(as.character(column))
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
      column_names <- list()
      for (meta in self$meta) {
        dimensions <- meta$output_dimensions
        columns_data <- data[, start:(start + dimensions - 1)]

        if ("levs" %in% names(meta)) {
          inverted <- self$inverse_transform_discrete(meta, columns_data)
        } else {
          inverted <- self$inverse_transform_continuous(meta, columns_data)
        }
        output[[length(output) + 1]] <- inverted
        column_names[[length(column_names) + 1]] <- meta$name
        start <- start + dimensions
      }
      output <- do.call("cbind", output)
      colnames(output) <- do.call("c", column_names)

      return(output)
    }
  )
)
