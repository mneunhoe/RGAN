# Tests for the vectorised mode-specific transform / inverse transform.
# These pin down the behaviour that the row-loop implementation had, plus the
# underflow case that the row loop got wrong.

# Reference (row-loop) implementation, kept here only as an oracle.
ref_transform_continuous <- function(meta, data) {
  n_modes <- meta$n_modes
  means <- meta$means
  stds <- meta$stds
  weights <- meta$weights
  n <- length(data)
  result <- matrix(0, nrow = n, ncol = n_modes + 1)
  for (i in seq_len(n)) {
    if (is.na(data[i])) {
      result[i, ] <- NA
      next
    }
    probs <- weights * stats::dnorm(data[i], mean = means, sd = stds)
    probs <- probs / sum(probs)
    if (all(!is.finite(probs)) || sum(probs, na.rm = TRUE) == 0) {
      # the row loop silently produced an all-zero mode indicator here
      next
    }
    mode_idx <- which.max(probs)
    result[i, mode_idx] <- 1
    normalized <- (data[i] - means[mode_idx]) / stds[mode_idx]
    normalized <- max(-4, min(4, normalized))
    result[i, n_modes + 1] <- normalized / 4
  }
  result
}

make_meta <- function() {
  list(
    name = "x",
    mode_specific = TRUE,
    n_modes = 3L,
    means = c(-2, 0, 5),
    stds = c(0.5, 1, 2),
    weights = c(0.2, 0.5, 0.3),
    output_info = list(4, "mode_specific"),
    output_dimensions = 4
  )
}

test_that("vectorised mode-specific transform matches the row-loop reference", {
  set.seed(42)
  meta <- make_meta()
  x <- c(rnorm(300, -2, 0.5), rnorm(300, 0, 1), rnorm(300, 5, 2))
  tr <- data_transformer$new()

  got <- tr$transform_continuous(meta, x)
  want <- ref_transform_continuous(meta, x)

  expect_equal(got, want)
})

test_that("NA rows are propagated as all-NA rows", {
  meta <- make_meta()
  tr <- data_transformer$new()
  x <- c(1, NA, -2, NA, 5)

  got <- tr$transform_continuous(meta, x)

  expect_equal(dim(got), c(5L, 4L))
  expect_true(all(is.na(got[2, ])))
  expect_true(all(is.na(got[4, ])))
  expect_false(any(is.na(got[c(1, 3, 5), ])))
  expect_equal(got[c(1, 3, 5), , drop = FALSE],
               ref_transform_continuous(meta, x)[c(1, 3, 5), , drop = FALSE])
})

test_that("values are clipped to [-4, 4] SDs and scaled by 1/4", {
  meta <- make_meta()
  tr <- data_transformer$new()
  # 0 + 10 * sd(=1) is far beyond the clip on mode 2 but mode 3 is closer;
  # use an explicit single-mode meta to isolate the clipping.
  single <- list(name = "x", mode_specific = TRUE, n_modes = 1L,
                 means = 0, stds = 1, weights = 1,
                 output_info = list(2, "mode_specific"), output_dimensions = 2)

  got <- tr$transform_continuous(single, c(-100, -2, 0, 2, 100))

  expect_equal(got[, 1], rep(1, 5))
  expect_equal(got[, 2], c(-1, -0.5, 0, 0.5, 1))
})

test_that("mode assignment survives density underflow (old loop assigned no mode)", {
  # A near-degenerate mode (CTGAN routinely produces these, e.g. Adult's
  # capital_gain point mass at 0) makes every density underflow to 0 for a
  # far-away value. Every row must still get exactly one mode indicator.
  meta <- list(name = "x", mode_specific = TRUE, n_modes = 2L,
               means = c(0, 15024), stds = c(1e-6, 1e-6), weights = c(0.9, 0.1),
               output_info = list(3, "mode_specific"), output_dimensions = 3)
  tr <- data_transformer$new()

  got <- tr$transform_continuous(meta, c(0, 15024, 99999))

  expect_equal(rowSums(got[, 1:2, drop = FALSE]), rep(1, 3))
  # 99999 is nearest (in log-density) to the 15024 mode
  expect_equal(got[3, 2], 1)
})

test_that("vectorised inverse transform matches the row-loop reference", {
  set.seed(7)
  meta <- make_meta()
  tr <- data_transformer$new()
  x <- rnorm(500, 1, 3)
  tx <- tr$transform_continuous(meta, x)

  got <- tr$inverse_transform_continuous(meta, tx)

  mode_idx <- max.col(tx[, 1:3, drop = FALSE], ties.method = "first")
  want <- vapply(seq_along(x), function(i) {
    tx[i, 4] * 4 * meta$stds[mode_idx[i]] + meta$means[mode_idx[i]]
  }, numeric(1))

  expect_equal(got, want)
})

test_that("inverse transform still handles a single row supplied as a vector", {
  meta <- make_meta()
  tr <- data_transformer$new()

  got <- tr$inverse_transform_continuous(meta, c(0, 1, 0, 0.25))

  expect_equal(got, 0 + 0.25 * 4 * 1)
})

test_that("full round trip on mixed data is unchanged end to end", {
  set.seed(11)
  df <- data.frame(
    a = c(rnorm(400, -3), rnorm(400, 4)),
    b = rlnorm(800),
    g = sample(c("x", "y", "z"), 800, replace = TRUE),
    stringsAsFactors = FALSE
  )
  tr <- data_transformer$new()
  tr$fit(df, discrete_columns = "g", mode_specific = TRUE, n_modes = 5)
  tx <- tr$transform(df)

  expect_equal(nrow(tx), 800)
  expect_equal(ncol(tx), tr$output_dimensions)

  back <- tr$inverse_transform(tx)
  expect_equal(nrow(back), 800)
  expect_gt(cor(df$a, back$a), 0.95)
  expect_gt(cor(df$b, back$b), 0.9)
  expect_true(all(back$g == df$g))
})
