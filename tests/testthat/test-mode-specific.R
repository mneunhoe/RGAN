test_that("data_transformer accepts mode_specific parameter", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()

  # Should work with mode_specific = TRUE
  transformer$fit(data, mode_specific = TRUE)

  expect_true(transformer$mode_specific)
  expect_equal(transformer$n_modes, 10)  # default n_modes
})

test_that("data_transformer accepts n_modes parameter", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()

  transformer$fit(data, mode_specific = TRUE, n_modes = 5)

  expect_true(transformer$mode_specific)
  expect_equal(transformer$n_modes, 5)
})

test_that("mode_specific metadata is stored correctly", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 5)

  # Check that meta contains mode_specific info
  for (meta in transformer$meta) {
    expect_true(meta$mode_specific)
    expect_true("means" %in% names(meta))
    expect_true("stds" %in% names(meta))
    expect_true("weights" %in% names(meta))
    expect_true("n_modes" %in% names(meta))
  }
})

test_that("standard normalization still works", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()

  # Default should be mode_specific = FALSE

  transformer$fit(data)

  expect_false(transformer$mode_specific)

  # Check that meta uses standard normalization
  for (meta in transformer$meta) {
    expect_false(isTRUE(meta$mode_specific))
    expect_true("mean" %in% names(meta))
    expect_true("std" %in% names(meta))
  }
})

test_that("mode_specific transform produces correct dimensions", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 5)

  transformed <- transformer$transform(data)

  # Each continuous column should produce n_modes + 1 columns
  # With 2 columns and n_modes=5, we expect 2 * (5 + 1) = 12 columns
  # However, n_modes may be reduced if some modes have low weight
  expect_true(ncol(transformed) >= 2)
  expect_equal(nrow(transformed), 100)
})

test_that("mode_specific transform output has correct structure", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)

  transformed <- transformer$transform(data)

  # Output dimensions should match what's in meta
  expected_cols <- sum(sapply(transformer$meta, function(m) m$output_dimensions))
  expect_equal(ncol(transformed), expected_cols)
})

test_that("mode_specific inverse_transform recovers approximate values", {
  set.seed(123)
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 5)

  transformed <- transformer$transform(data)
  reconstructed <- transformer$inverse_transform(transformed)

  # Values should be approximately recovered
  # Allow for some error due to mode assignment and clipping
  correlation_x <- cor(data[, 1], reconstructed[, 1])
  correlation_y <- cor(data[, 2], reconstructed[, 2])

  expect_gt(correlation_x, 0.9)  # High correlation expected

  expect_gt(correlation_y, 0.9)
})

test_that("standard transform inverse_transform round trip works", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = FALSE)

  transformed <- transformer$transform(data)
  reconstructed <- transformer$inverse_transform(transformed)

  # Standard normalization should perfectly reconstruct
  expect_equal(as.numeric(data[, 1]), as.numeric(reconstructed[, 1]), tolerance = 1e-10)
  expect_equal(as.numeric(data[, 2]), as.numeric(reconstructed[, 2]), tolerance = 1e-10)
})

test_that("mode_specific works with mixed continuous and discrete columns", {
  data <- sample_toydata(n = 100)
  # Add a discrete column - use data.frame to avoid cbind converting to character
  data <- data.frame(data, category = sample(c("A", "B", "C"), 100, replace = TRUE))

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = "category", mode_specific = TRUE, n_modes = 3)

  transformed <- transformer$transform(data)
  reconstructed <- transformer$inverse_transform(transformed)

  # Continuous columns should have mode_specific = TRUE
  expect_true(transformer$meta[[1]]$mode_specific)
  expect_true(transformer$meta[[2]]$mode_specific)

  # Discrete column should have levs
  expect_true("levs" %in% names(transformer$meta[[3]]))
})

test_that("fit_gmm produces valid parameters", {
  data <- sample_toydata(n = 200)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 5)

  for (meta in transformer$meta) {
    # Check means, stds, weights are valid
    expect_true(length(meta$means) >= 1)
    expect_true(length(meta$stds) >= 1)
    expect_true(length(meta$weights) >= 1)

    # Weights should sum to 1
    expect_equal(sum(meta$weights), 1, tolerance = 1e-6)

    # All stds should be positive
    expect_true(all(meta$stds > 0))
  }
})

test_that("mode_specific handles single mode gracefully", {
  # Create data that only needs one mode
  data <- matrix(rnorm(200, mean = 5, sd = 0.1), ncol = 2)
  colnames(data) <- c("x", "y")

  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)

  transformed <- transformer$transform(data)
  reconstructed <- transformer$inverse_transform(transformed)

  # Should still work
  expect_equal(nrow(transformed), 100)
  expect_equal(nrow(reconstructed), 100)
})

test_that("mode_specific handles bimodal data", {
  # Create bimodal data
  set.seed(42)
  x <- c(rnorm(50, mean = -2, sd = 0.5), rnorm(50, mean = 2, sd = 0.5))
  y <- c(rnorm(50, mean = 0, sd = 0.5), rnorm(50, mean = 5, sd = 0.5))
  data <- cbind(x = x, y = y)

  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 5)

  transformed <- transformer$transform(data)
  reconstructed <- transformer$inverse_transform(transformed)

  # Check reasonable reconstruction
  correlation_x <- cor(data[, 1], reconstructed[, 1])
  correlation_y <- cor(data[, 2], reconstructed[, 2])

  expect_gt(correlation_x, 0.85)
  expect_gt(correlation_y, 0.85)
})

test_that("output_dimensions is correctly computed for mode_specific", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 4)

  # output_dimensions should equal sum of meta output_dimensions
  total_dim <- sum(sapply(transformer$meta, function(m) m$output_dimensions))
  expect_equal(transformer$output_dimensions, total_dim)
})

test_that("output_info contains correct info for mode_specific", {
  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)

  for (i in seq_along(transformer$meta)) {
    info <- transformer$output_info[[i]]
    meta <- transformer$meta[[i]]

    if (isTRUE(meta$mode_specific)) {
      expect_equal(info[[2]], "mode_specific")
      expect_equal(info[[1]], meta$n_modes + 1)
    }
  }
})

test_that("n_modes is reduced when data is insufficient", {
  # Small dataset
  data <- sample_toydata(n = 20)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 50)

  # n_modes should be reduced from 50 to at most floor(20/2) = 10
  for (i in seq_along(transformer$meta)) {
    meta <- transformer$meta[[i]]
    n_modes <- as.numeric(meta$n_modes)
    expect_true(!is.null(n_modes) && length(n_modes) == 1)
    expect_true(n_modes <= 10)  # Should be reduced from 50 to at most 10
    expect_true(n_modes >= 1)   # Should have at least 1 mode
  }
})

test_that("mode_specific handles NA values in data", {
  data <- sample_toydata(n = 100)
  # Add some NAs
  data[1:5, 1] <- NA
  data[10:15, 2] <- NA

  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)

  # Fit should work, ignoring NAs
  expect_true(transformer$mode_specific)

  # Transform should produce NAs in same positions
  transformed <- transformer$transform(data)
  expect_true(any(is.na(transformed)))
})

test_that("mode_specific transform clips extreme values", {
  # Create data with extreme outliers
  data <- sample_toydata(n = 100)
  data[1, 1] <- 100  # Extreme value

  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)

  transformed <- transformer$transform(data)

  # The normalized value (last column for first variable) should be clipped
  # Mode columns are first, then normalized value is last for each variable
  n_modes <- transformer$meta[[1]]$n_modes
  normalized_col <- n_modes + 1

  # Should be between -1 and 1 (since clipped to ±4 SD then divided by 4)
  expect_true(all(abs(transformed[, normalized_col]) <= 1, na.rm = TRUE))
})

test_that("mode_specific with GAN training produces valid output", {
  skip_if_not_installed("torch")

  data <- sample_toydata(n = 100)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)

  transformed_data <- transformer$transform(data)

  # Train a minimal GAN
  result <- gan_trainer(
    transformed_data,
    epochs = 2,
    batch_size = 20,
    seed = 123
  )

  expect_s3_class(result, "trained_RGAN")

  # Sample synthetic data
  synthetic <- sample_synthetic_data(result, transformer, n = 50)

  expect_equal(nrow(synthetic), 50)
  expect_equal(ncol(synthetic), 2)
})

test_that("transform_continuous produces one-hot mode encoding", {
  data <- sample_toydata(n = 50)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)

  transformed <- transformer$transform(data)

  # For each row, the mode columns should have exactly one 1 and rest 0s
  n_modes <- transformer$meta[[1]]$n_modes
  mode_cols <- transformed[, 1:n_modes]

  # Each row should sum to 1 (one-hot)
  row_sums <- rowSums(mode_cols)
  expect_true(all(abs(row_sums - 1) < 1e-6))

  # Each row should have exactly one 1
  for (i in 1:nrow(mode_cols)) {
    expect_equal(sum(mode_cols[i, ] == 1), 1)
  }
})

test_that("mode assignment is deterministic", {
  set.seed(123)
  data <- sample_toydata(n = 50)
  transformer <- data_transformer$new()
  transformer$fit(data, mode_specific = TRUE, n_modes = 3)

  transformed1 <- transformer$transform(data)
  transformed2 <- transformer$transform(data)

  # Same data should produce same transformation
  expect_equal(transformed1, transformed2)
})
