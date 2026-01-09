test_that("data_transformer can be created", {
  transformer <- data_transformer$new()
  expect_s3_class(transformer, "data_transformer")
})

test_that("data_transformer fit works with continuous data", {
  data <- matrix(rnorm(100), ncol = 2)
  colnames(data) <- c("x", "y")

  transformer <- data_transformer$new()
  transformer$fit(data)

  expect_equal(length(transformer$meta), 2)
  expect_equal(transformer$meta[[1]]$name, "x")
  expect_equal(transformer$meta[[2]]$name, "y")
})

test_that("data_transformer transform produces z-scores", {
  set.seed(123)
  data <- matrix(rnorm(1000, mean = 10, sd = 5), ncol = 2)
  colnames(data) <- c("x", "y")


  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed <- transformer$transform(data)

  # Transformed data should have mean ~0 and sd ~1
 expect_equal(mean(transformed[, 1]), 0, tolerance = 0.1)
  expect_equal(sd(transformed[, 1]), 1, tolerance = 0.1)
})

test_that("data_transformer inverse_transform recovers original data", {
  set.seed(123)
  data <- matrix(rnorm(100, mean = 5, sd = 2), ncol = 2)
  colnames(data) <- c("x", "y")

  transformer <- data_transformer$new()
  transformer$fit(data)
  transformed <- transformer$transform(data)
  recovered <- transformer$inverse_transform(transformed)

  expect_equal(recovered[, 1], data[, 1], tolerance = 1e-10)
  expect_equal(recovered[, 2], data[, 2], tolerance = 1e-10)
})

test_that("data_transformer handles discrete columns", {
  set.seed(123)
  data <- cbind(
    x = rnorm(100),
    category = sample(1:3, 100, replace = TRUE)
  )

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = "category")
  transformed <- transformer$transform(data)

  # Should have 1 continuous + 3 one-hot columns = 4 columns
 expect_equal(ncol(transformed), 4)

  # One-hot columns should sum to 1 for each row
  one_hot_sum <- rowSums(transformed[, 2:4])
  expect_true(all(one_hot_sum == 1))
})

test_that("data_transformer roundtrip works with mixed data", {
  set.seed(123)
  data <- cbind(
    x = rnorm(100),
    y = rnorm(100, mean = 10),
    category = sample(1:3, 100, replace = TRUE)
  )

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = "category")
  transformed <- transformer$transform(data)
  recovered <- transformer$inverse_transform(transformed)

  # Continuous columns should be recovered exactly
  expect_equal(recovered[, "x"], data[, "x"], tolerance = 1e-10)
  expect_equal(recovered[, "y"], data[, "y"], tolerance = 1e-10)

  # Discrete column should match (allowing for ties in one-hot decoding)
 expect_equal(as.numeric(recovered[, "category"]), as.numeric(data[, "category"]))
})

test_that("data_transformer handles string categorical columns", {
  set.seed(123)
  data <- data.frame(
    x = rnorm(100),
    workclass = sample(c("Private", "Self-emp", "Gov", "Other"), 100, replace = TRUE),
    education = sample(c("HS-grad", "Bachelors", "Masters"), 100, replace = TRUE),
    stringsAsFactors = FALSE
  )

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = c("workclass", "education"))
  transformed <- transformer$transform(data)

  # Should have 1 continuous + 4 workclass + 3 education = 8 columns
  expect_equal(ncol(transformed), 8)

  # Inverse transform
  recovered <- transformer$inverse_transform(transformed)

  # Continuous column should be recovered exactly
  expect_equal(recovered[, "x"], data[, "x"], tolerance = 1e-10)

  # String categorical columns should be recovered as strings, not NA
  expect_false(any(is.na(recovered[, "workclass"])))
  expect_false(any(is.na(recovered[, "education"])))

  # Should match original values
  expect_equal(recovered[, "workclass"], data$workclass)
  expect_equal(recovered[, "education"], data$education)
})

test_that("data_transformer handles mixed numeric and string categoricals", {
  set.seed(123)
  data <- data.frame(
    x = rnorm(50),
    num_cat = sample(1:3, 50, replace = TRUE),
    str_cat = sample(c("A", "B", "C"), 50, replace = TRUE),
    stringsAsFactors = FALSE
  )

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = c("num_cat", "str_cat"))
  transformed <- transformer$transform(data)
  recovered <- transformer$inverse_transform(transformed)

  # Numeric categorical should be numeric
  expect_true(is.numeric(recovered[, "num_cat"]))
  expect_equal(recovered[, "num_cat"], data$num_cat)

  # String categorical should be character
  expect_true(is.character(recovered[, "str_cat"]))
  expect_equal(recovered[, "str_cat"], data$str_cat)
})
