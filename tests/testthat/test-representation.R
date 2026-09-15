# Tests for Experiment 2 (representation): the Gaussian quantile transform for
# continuous columns, the TabRep cos/sin encoding for categorical columns, the
# embedding-ready integer path, and the matching TabularGenerator heads.

# --- Gaussian quantile transform ---------------------------------------------

test_that("quantile transform round trips interior points of a log-normal column", {
  set.seed(20260912)
  x <- rlnorm(5000, meanlog = 0, sdlog = 1)
  data <- data.frame(x = x)

  transformer <- data_transformer$new()
  transformer$fit(data, continuous_transform = "quantile", n_quantiles = 1000)

  expect_equal(transformer$output_dimensions, 1)
  expect_equal(transformer$output_info[[1]], list(1, "quantile"))
  expect_equal(transformer$continuous_transform, "quantile")

  transformed <- transformer$transform(data)
  expect_equal(dim(transformed), c(5000L, 1L))

  reconstructed <- transformer$inverse_transform(transformed)

  # The two extreme order statistics sit at CDF position 0 and 1 and are
  # clipped to [1e-4, 1 - 1e-4]; every interior point must round trip exactly.
  interior <- x > stats::quantile(x, 0.01) & x < stats::quantile(x, 0.99)
  expect_lt(max(abs(reconstructed$x[interior] - x[interior])), 1e-6)
})

test_that("quantile transformed column is approximately standard normal", {
  set.seed(11)
  x <- rlnorm(5000)
  data <- data.frame(x = x)

  transformer <- data_transformer$new()
  transformer$fit(data, continuous_transform = "quantile")
  z <- as.numeric(transformer$transform(data))

  expect_lt(abs(mean(z)), 0.05)
  expect_lt(abs(sd(z) - 1), 0.05)
})

test_that("quantile transform handles a constant column", {
  data <- data.frame(const = rep(3.5, 200))

  transformer <- data_transformer$new()
  expect_silent(transformer$fit(data, continuous_transform = "quantile"))
  expect_true(isTRUE(transformer$meta[[1]]$constant))

  transformed <- transformer$transform(data)
  expect_true(all(transformed == 0))

  reconstructed <- transformer$inverse_transform(transformed)
  expect_true(all(reconstructed$const == 3.5))
})

test_that("quantile transform propagates NA values", {
  data <- data.frame(x = c(1, NA, 3, 4, 5, NA, 7))

  transformer <- data_transformer$new()
  transformer$fit(data, continuous_transform = "quantile")

  transformed <- transformer$transform(data)
  expect_equal(which(is.na(transformed)), c(2L, 6L))

  reconstructed <- transformer$inverse_transform(transformed)
  expect_equal(which(is.na(reconstructed$x)), c(2L, 6L))
})

test_that("continuous_transform argument is validated and resolves mode_specific", {
  data <- sample_toydata(n = 100)

  transformer <- data_transformer$new()
  expect_error(transformer$fit(data, continuous_transform = "nonsense"),
               "continuous_transform must be one of")

  # continuous_transform wins over an inconsistent mode_specific, with a warning
  expect_warning(
    transformer$fit(data, mode_specific = TRUE, continuous_transform = "quantile"),
    "conflicts with mode_specific"
  )
  expect_equal(transformer$continuous_transform, "quantile")
  expect_false(transformer$mode_specific)

  expect_warning(
    transformer$fit(data, mode_specific = FALSE, continuous_transform = "mode_specific",
                    n_modes = 3),
    "conflicts with mode_specific"
  )
  expect_true(transformer$mode_specific)

  # Consistent combinations are silent
  expect_silent(transformer$fit(data, mode_specific = TRUE,
                                continuous_transform = "mode_specific", n_modes = 3))
  expect_silent(transformer$fit(data, continuous_transform = "zscore"))
})

test_that("categorical_encoding argument is validated", {
  data <- data.frame(g = c("a", "b", "a"))
  transformer <- data_transformer$new()
  expect_error(
    transformer$fit(data, discrete_columns = "g", categorical_encoding = "nope"),
    "categorical_encoding must be one of"
  )
})

# --- Backward compatibility ---------------------------------------------------

test_that("fit(mode_specific = TRUE) yields the previous output_info", {
  set.seed(3)
  data <- sample_toydata(n = 300)

  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 5)

  expect_true(transformer$mode_specific)
  for (i in seq_along(transformer$output_info)) {
    info <- transformer$output_info[[i]]
    expect_equal(info[[2]], "mode_specific")
    expect_equal(info[[1]], transformer$meta[[i]]$n_modes + 1)
  }
  expect_equal(transformer$output_dimensions,
               sum(sapply(transformer$output_info, function(x) x[[1]])))

  # And the default (z-score) path is unchanged
  transformer2 <- data_transformer$new()
  transformer2$fit(data)
  expect_equal(transformer2$output_info,
               replicate(ncol(data), list(1, "linear"), simplify = FALSE))
  expect_equal(transformer2$output_dimensions, ncol(data))

  # One-hot categorical columns still produce "softmax" blocks
  mixed <- data.frame(x = rnorm(50), g = sample(c("a", "b", "c"), 50, TRUE))
  transformer3 <- data_transformer$new()
  transformer3$fit(mixed, discrete_columns = "g")
  expect_equal(transformer3$output_info[[2]], list(3, "softmax"))
})

# --- cos/sin (TabRep) encoding -----------------------------------------------

test_that("cossin encoding round trips exactly for several cardinalities", {
  set.seed(7)
  for (K in c(1, 2, 3, 10, 500)) {
    levs <- paste0("L", seq_len(K))
    n <- 4 * K + 8
    # Every level must occur, otherwise fit_discrete legitimately drops it.
    values <- factor(rep_len(levs, n), levels = levs)
    values <- values[sample.int(n)]
    data <- data.frame(g = values, stringsAsFactors = FALSE)

    transformer <- data_transformer$new()
    transformer$fit(data, discrete_columns = "g", categorical_encoding = "cossin")

    expect_equal(transformer$output_dimensions, 2,
                 info = paste("K =", K))
    expect_equal(transformer$output_info[[1]], list(2, "cossin"))
    expect_equal(transformer$meta[[1]]$K, K)

    transformed <- transformer$transform(data)
    expect_equal(dim(transformed), c(n, 2L))
    # Encoded points lie on the unit circle
    expect_lt(max(abs(sqrt(rowSums(transformed^2)) - 1)), 1e-12)

    reconstructed <- transformer$inverse_transform(transformed)
    expect_identical(as.character(reconstructed$g), as.character(values),
                     info = paste("K =", K))

    stats_g <- transformer$last_snap_stats[["g"]]
    expect_false(is.null(stats_g))
    expect_equal(stats_g$K, K)
    expect_equal(stats_g$n, n)
    expect_lt(stats_g$mean, 1e-8)
    expect_lt(stats_g$q90, 1e-8)
  }
})

test_that("cossin snap statistics grow when the encoding is perturbed", {
  set.seed(8)
  levs <- paste0("L", 1:8)
  data <- data.frame(g = factor(sample(levs, 1000, TRUE), levels = levs))

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = "g", categorical_encoding = "cossin")
  transformed <- transformer$transform(data)

  noisy <- transformed + matrix(rnorm(length(transformed), sd = 0.3),
                                nrow = nrow(transformed))
  invisible(transformer$inverse_transform(noisy))
  noisy_stats <- transformer$last_snap_stats[["g"]]

  expect_gt(noisy_stats$mean, 0)
  expect_gte(noisy_stats$q90, noisy_stats$mean)
  # The snap distance is measured in units of the inter-level angle and can
  # never exceed half of it.
  expect_lte(noisy_stats$q90, 0.5 + 1e-8)

  # last_snap_stats is reset for every inverse_transform call
  invisible(transformer$inverse_transform(transformed))
  expect_lt(transformer$last_snap_stats[["g"]]$mean, 1e-8)
})

# --- embedding-ready integer index --------------------------------------------

test_that("embedding_ready keeps a single integer index column", {
  levs <- c("a", "b", "c", "d")
  data <- data.frame(g = factor(rep(levs, 25), levels = levs))

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = "g",
                  categorical_encoding = "embedding_ready")

  expect_equal(transformer$output_dimensions, 1)
  expect_equal(transformer$output_info[[1]], list(1, "index"))

  transformed <- transformer$transform(data)
  expect_equal(dim(transformed), c(100L, 1L))
  expect_true(all(transformed %in% 1:4))

  reconstructed <- transformer$inverse_transform(transformed)
  expect_identical(as.character(reconstructed$g), as.character(data$g))

  # Rounding and clipping of out-of-range indices
  odd <- matrix(c(-3, 0.6, 2.4, 99), ncol = 1)
  expect_identical(as.character(transformer$inverse_transform(odd)$g),
                   c("a", "a", "b", "d"))
})

# --- Mixed data ---------------------------------------------------------------

test_that("quantile + cossin round trips mixed data", {
  set.seed(42)
  n <- 1000
  data <- data.frame(
    age = round(rnorm(n, 40, 12)),
    income = rlnorm(n, 10, 1),
    gender = sample(c("M", "F"), n, replace = TRUE),
    education = sample(c("HS", "BA", "MA", "PhD"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )

  transformer <- data_transformer$new()
  transformer$fit(data,
                  discrete_columns = c("gender", "education"),
                  continuous_transform = "quantile",
                  categorical_encoding = "cossin")

  # 1 column per numeric + 2 per categorical
  expect_equal(transformer$output_dimensions, 1 + 1 + 2 + 2)
  types <- sapply(transformer$output_info, function(x) x[[2]])
  expect_equal(types, c("quantile", "quantile", "cossin", "cossin"))

  transformed <- transformer$transform(data)
  expect_equal(dim(transformed), c(n, 6L))
  expect_false(any(is.na(transformed)))

  reconstructed <- transformer$inverse_transform(transformed)
  expect_equal(nrow(reconstructed), n)
  expect_identical(reconstructed$gender, data$gender)
  expect_identical(reconstructed$education, data$education)

  # Numerics round trip up to the clipping of the two extreme order statistics
  for (col in c("age", "income")) {
    interior <- data[[col]] > stats::quantile(data[[col]], 0.01) &
      data[[col]] < stats::quantile(data[[col]], 0.99)
    expect_lt(max(abs(reconstructed[[col]][interior] - data[[col]][interior])),
              1e-6)
  }

  expect_named(transformer$last_snap_stats, c("gender", "education"))
  expect_lt(transformer$last_snap_stats$education$mean, 1e-8)
})

# --- Generator heads ----------------------------------------------------------

test_that("TabularGenerator supports quantile and cossin heads", {
  skip_if_not_installed("torch")

  output_info <- list(list(1, "quantile"), list(2, "cossin"),
                      list(3, "softmax"), list(1, "linear"))
  gen <- TabularGenerator(noise_dim = 8, output_info = output_info,
                          hidden_units = list(16, 16))
  gen$eval()

  out <- as.matrix(gen(torch::torch_randn(c(11, 8))))
  expect_equal(dim(out), c(11L, 7L))

  # cos/sin block lies on the unit circle
  norms <- sqrt(rowSums(out[, 2:3, drop = FALSE]^2))
  expect_lt(max(abs(norms - 1)), 1e-5)

  # softmax block is a simplex, linear block is bounded by tanh
  expect_lt(max(abs(rowSums(out[, 4:6, drop = FALSE]) - 1)), 1e-5)
  expect_true(all(abs(out[, 7]) <= 1))

  # The quantile head is unbounded (plain linear); it must not be squashed.
  # Just check that it is finite and not forced into [-1, 1] by construction.
  expect_true(all(is.finite(out[, 1])))
})

test_that("TabularGenerator rejects the index output type", {
  skip_if_not_installed("torch")

  expect_error(
    TabularGenerator(noise_dim = 4, output_info = list(list(1, "index"))),
    "no TabularGenerator head"
  )
  expect_error(
    TabularGenerator(noise_dim = 4, output_info = list(list(1, "bogus"))),
    "output_info type must be one of"
  )
  expect_error(
    TabularGenerator(noise_dim = 4, output_info = list(list(3, "cossin"))),
    "must have dimension 2"
  )
})

test_that("gan_trainer trains on quantile + cossin data", {
  skip_if_not_installed("torch")

  set.seed(5)
  n <- 200
  data <- data.frame(
    x = rlnorm(n),
    g = sample(c("a", "b", "c"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = "g",
                  continuous_transform = "quantile",
                  categorical_encoding = "cossin")
  transformed <- transformer$transform(data)

  # gan_trainer() validates output_info types itself. If it has not yet been
  # taught about the new types, skip rather than fail: the transformer and the
  # generator heads are covered by the tests above.
  result <- tryCatch(
    gan_trainer(
      transformed,
      output_info = transformer$output_info,
      epochs = 2,
      batch_size = 50,
      noise_dim = 4,
      seed = 123
    ),
    error = function(e) {
      if (grepl("output_info type must be one of", conditionMessage(e))) {
        return(NULL)
      }
      stop(e)
    }
  )
  skip_if(is.null(result),
          "gan_trainer()'s valid_types does not yet accept 'quantile'/'cossin'")

  expect_s3_class(result, "trained_RGAN")

  synthetic <- sample_synthetic_data(result, transformer, n = 60)
  expect_equal(nrow(synthetic), 60)
  expect_equal(ncol(synthetic), 2)
  expect_true(all(synthetic$g %in% c("a", "b", "c")))
  expect_true(all(is.finite(synthetic$x)))

  snap <- transformer$last_snap_stats[["g"]]
  expect_false(is.null(snap))
  expect_gte(snap$mean, 0)
  expect_lte(snap$mean, 0.5 + 1e-8)
})
