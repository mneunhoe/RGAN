#' @title Sample Synthetic Data with explicit noise input
#'
#' @description Provides a function that makes it easy to sample synthetic data from a Generator
#'
#' @param g_net A torch::nn_module with a Generator
#' @param z A noise vector
#' @param device The device on which synthetic data should be sampled (cpu or cuda)
#' @param eval_dropout Should dropout be applied during inference
#'
#' @return Synthetic data
#' @export
expert_sample_synthetic_data <-
  function(g_net,
           z,
           device,
           eval_dropout = FALSE) {
    # Pass the noise through the Generator to create fake data

    if (eval_dropout) {
      fake_data <-  g_net(z)
    } else {
      g_net$eval()
      fake_data <-  g_net(z)
      g_net$train()
    }
    # Create an R array/matrix from the torch_tensor
    synth_data <- torch::as_array(fake_data$detach()$cpu())
    return(synth_data)
  }


#' @title Build the sampling condition of a trained conditional RGAN
#'
#' @description Internal helper shared by [sample_synthetic_data()] and
#'   [impute_synthetic()]. Resolves the `condition` argument into the
#'   `(n_samples x (n_blocks + data_dim))` condition matrix, following the
#'   rules documented in [sample_synthetic_data()].
#'
#' @param trained_gan A trained RGAN object.
#' @param transformer The transformer used to pre-process the data.
#' @param n_samples Number of rows to generate.
#' @param condition NULL or a data.frame in the original column space.
#'
#' @return NULL for an unconditional model, otherwise a list with `mask`,
#'   `values`, `cond` and (for a user condition) `observed`.
#' @keywords internal
resolve_sampling_condition <- function(trained_gan, transformer, n_samples,
                                       condition = NULL) {
  meta <- trained_gan$conditioning_meta
  if (is.null(meta)) {
    if (!is.null(condition)) {
      stop("this model was trained without `conditioning`, so `condition` cannot be used")
    }
    return(NULL)
  }
  if (!is.null(condition)) {
    return(condition_matrix_from_df(condition, transformer, meta, n_samples))
  }
  type <- trained_gan$settings$conditioning$type
  if (identical(type, "single_column")) {
    # CTGAN's unconditional sampling: a condition per row from the (column,
    # category) table, at the empirical frequency.
    return(sample_single_column_condition(meta, n_samples))
  }
  # random_mask: the all-unobserved condition, the mode cond_dropout trains.
  zero_condition(meta, n_samples)
}


#' @title Sample Synthetic Data from a trained RGAN
#'
#' @description Provides a function that makes it easy to sample synthetic data from a Generator
#'
#' @param trained_gan A trained RGAN object of class "trained_RGAN"
#' @param transformer The transformer object used to pre-process the data
#' @param n_samples The number of synthetic samples to generate. Defaults to NULL, which will use `trained_gan$settings$synthetic_examples`.
#' @param condition Optional condition for a model trained with the
#'   `conditioning` argument of [gan_trainer()]. A data.frame in the
#'   **original** column space, where `NA` means "not observed": one row is
#'   broadcast to all `n_samples`, otherwise it must have `n_samples` rows.
#'   Columns that are not present count as unobserved. The data.frame is
#'   transformed with `transformer$transform()` after the `NA` cells have been
#'   filled with a placeholder, and the blocks of the `NA` cells are then
#'   masked out, so the placeholder never reaches the generator.
#'
#'   With `condition = NULL` the behaviour depends on how the model was
#'   trained:
#'   \itemize{
#'     \item unconditional model: plain sampling, exactly as before;
#'     \item `conditioning$type = "random_mask"`: the all-unobserved mask, the
#'       unconditional mode that `cond_dropout` trains;
#'     \item `conditioning$type = "single_column"`: CTGAN's sampling
#'       procedure, a `(column, category)` condition per row drawn from the
#'       **empirical** frequency table (not the log-frequency table used
#'       during training), so the marginal of the conditioned column is
#'       preserved.
#'   }
#'
#' @return A data.frame of synthetic data (a matrix if no `transformer` is given).
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
sample_synthetic_data <-
  function(trained_gan, transformer = NULL, n_samples = NULL,
           condition = NULL) {

    if(is.null(n_samples)) {
      n_samples <- trained_gan$settings$synthetic_examples
    }
    z <- trained_gan$settings$sample_noise(c(
      n_samples,
      trained_gan$settings$noise_dim
    ))$to(device = trained_gan$settings$device)

    cnd <- resolve_sampling_condition(trained_gan, transformer, n_samples,
                                      condition)
    if (!is.null(cnd)) {
      cond_t <- torch::torch_tensor(cnd$cond)$to(device = z$device)$to(dtype = z$dtype)
      z <- torch::torch_cat(list(z, cond_t), dim = 2)
    }

    if (trained_gan$settings$eval_dropout) {
      fake_data <-  trained_gan$generator(z)
    } else {
      trained_gan$generator$eval()
      fake_data <- trained_gan$generator(z)
      trained_gan$generator$train()
    }
    # Create an R array/matrix from the torch_tensor
    synth_data <- torch::as_array(fake_data$detach()$cpu())
    if (!is.null(transformer)) {
      synth_data <- transformer$inverse_transform(synth_data)
    }
    return(synth_data)
  }


#' @title Impute missing cells with a conditionally trained RGAN
#'
#' @description Fills the `NA` cells of a data.frame with values drawn from a
#'   generator that was trained with the `conditioning` argument of
#'   [gan_trainer()]. Every row is generated conditionally on its own observed
#'   cells (the any-order conditioning of TabMT / TabularARGN, report item T3),
#'   and the observed cells are then copied back unchanged, so the returned
#'   data.frame differs from the input only where the input was `NA`.
#'
#'   A model trained with `conditioning$type = "random_mask"` is the natural
#'   choice here, because it has seen arbitrary observed subsets; a
#'   `"single_column"` model has only ever seen one observed column at a time.
#'
#' @param trained_gan A trained RGAN object of class "trained_RGAN", trained
#'   with `conditioning`.
#' @param transformer The transformer object used to pre-process the data.
#' @param data_with_NA A data.frame (or matrix) in the original column space
#'   with `NA` in the cells to impute.
#' @param n_draws Number of independent generator draws per row. With the
#'   default 1 the imputation is a single draw (proper multiple-imputation
#'   semantics: call the function repeatedly for several completed data sets).
#'   With `n_draws > 1` numeric cells are averaged and discrete cells take the
#'   modal draw, which lowers imputation error but is no longer a draw from
#'   the conditional distribution.
#'
#' @return A data.frame of the same shape as `data_with_NA`, with the observed
#'   cells kept exactly and the `NA` cells filled from the generator.
#' @export
#' @examples
#' \dontrun{
#' trained_gan <- gan_trainer(transformed_data, output_info = transformer$output_info,
#'                            conditioning = list(type = "random_mask"))
#' completed <- impute_synthetic(trained_gan, transformer, data_with_NA)
#' }
impute_synthetic <- function(trained_gan, transformer, data_with_NA,
                             n_draws = 1) {
  if (is.null(trained_gan$conditioning_meta)) {
    stop("impute_synthetic() needs a model trained with the `conditioning` argument of gan_trainer()")
  }
  if (is.null(transformer)) {
    stop("impute_synthetic() needs the `transformer` that was used to fit the data")
  }
  if (!is.data.frame(data_with_NA)) data_with_NA <- as.data.frame(data_with_NA)
  if (!is.numeric(n_draws) || length(n_draws) != 1 || n_draws < 1) {
    stop("n_draws must be a positive integer")
  }
  n <- nrow(data_with_NA)
  if (n == 0) return(data_with_NA)

  draws <- vector("list", as.integer(n_draws))
  for (k in seq_len(as.integer(n_draws))) {
    draws[[k]] <- sample_synthetic_data(trained_gan, transformer,
                                        n_samples = n,
                                        condition = data_with_NA)
  }
  synth <- draws[[1]]
  if (length(draws) > 1) {
    for (nm in colnames(synth)) {
      cols <- lapply(draws, function(d) d[[nm]])
      if (is.numeric(synth[[nm]])) {
        synth[[nm]] <- rowMeans(do.call(cbind, cols))
      } else {
        m <- do.call(cbind, lapply(cols, as.character))
        synth[[nm]] <- apply(m, 1, function(r) names(sort(table(r), decreasing = TRUE))[1])
      }
    }
  }

  out <- data_with_NA
  for (nm in colnames(out)) {
    if (!nm %in% colnames(synth)) next
    miss <- is.na(out[[nm]])
    if (!any(miss)) next
    filled <- synth[[nm]][miss]
    if (is.factor(out[[nm]])) {
      lv <- levels(out[[nm]])
      chr <- as.character(out[[nm]])
      chr[miss] <- as.character(filled)
      out[[nm]] <- factor(chr, levels = lv)
    } else if (is.numeric(out[[nm]])) {
      out[[nm]][miss] <- as.numeric(filled)
    } else {
      out[[nm]][miss] <- as.character(filled)
    }
  }
  out
}
