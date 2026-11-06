# Tests for the column-token architectures of Experiment 4 (2026 plan):
# TokenDiscriminator, TokenGenerator and the shared output-head helper.

mixed_output_info <- function() {
  list(
    list(1, "linear"),
    list(5, "mode_specific"),   # 4 modes + 1 value
    list(3, "softmax"),
    list(1, "quantile"),
    list(2, "cossin")
  )
}

oi_dim <- function(oi) sum(vapply(oi, function(x) as.numeric(x[[1]]), numeric(1)))


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------

test_that("TokenDiscriminator returns one score per pack", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  dd <- oi_dim(oi)

  d_net <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 2)
  out <- d_net(torch::torch_randn(c(8, dd)))

  expect_equal(out$shape, c(8, 1))
  expect_equal(d_net$data_dim, dd)
  expect_equal(d_net$n_tokens, length(oi))
})

test_that("TokenDiscriminator sigmoid head is bounded", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  d_net <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1,
                              sigmoid = TRUE)
  out <- as.numeric(d_net(torch::torch_randn(c(5, oi_dim(oi))))$to(dtype = torch::torch_float()))
  expect_true(all(out > 0 & out < 1))
})

test_that("TokenDiscriminator handles the pac = 10 path", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  dd <- oi_dim(oi)

  d_net <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1, pac = 10)
  # exactly the tensor pack_samples() produces for a batch of 40
  raw <- torch::torch_randn(c(40, dd))
  packed <- raw$view(c(4, 10 * dd))
  out <- d_net(packed)

  expect_equal(out$shape, c(4, 1))
  # pac slots are distinguished by a learned pack-slot embedding
  expect_false(is.null(d_net$pac_embedding))

  # the unpacked width is rejected
  expect_error(d_net(raw), "expects 120 input columns")
})

test_that("TokenDiscriminator handles the extra_inputs = 1 path", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  dd <- oi_dim(oi)

  d_net <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1,
                              extra_inputs = 1)
  expect_equal(d_net$n_tokens, length(oi) + 1L)
  out <- d_net(torch::torch_randn(c(7, dd + 1)))
  expect_equal(out$shape, c(7, 1))

  # combined with packing
  d_pac <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1,
                              extra_inputs = 1, pac = 5)
  out2 <- d_pac(torch::torch_randn(c(3, 5 * (dd + 1))))
  expect_equal(out2$shape, c(3, 1))

  # the extra feature actually changes the score
  x <- torch::torch_randn(c(4, dd + 1))
  x2 <- x$clone()
  x2[, dd + 1] <- x[, dd + 1] + 5
  s1 <- as.numeric(d_net(x)$to(dtype = torch::torch_float()))
  s2 <- as.numeric(d_net(x2)$to(dtype = torch::torch_float()))
  expect_false(isTRUE(all.equal(s1, s2)))
})

test_that("both attention variants and row attention run", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  dd <- oi_dim(oi)
  x <- torch::torch_randn(c(9, dd))

  for (att in c("l2", "dot")) {
    d_net <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 2,
                                attention = att)
    out <- d_net(x)
    expect_equal(out$shape, c(9, 1))
    expect_true(as.logical(torch::torch_isfinite(out)$all()))

    d_row <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 2,
                                attention = att, row_attention = TRUE)
    out_row <- d_row(x)
    expect_equal(out_row$shape, c(9, 1))
    expect_true(as.logical(torch::torch_isfinite(out_row)$all()))

    g_net <- TokenGenerator(8, oi, d_model = 16, n_heads = 2, n_layers = 2,
                            attention = att)
    expect_equal(g_net(torch::torch_randn(c(9, 8)))$shape, c(9, dd))
  }
})

test_that("row attention couples the rows of a batch", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  dd <- oi_dim(oi)

  d_net <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1,
                              row_attention = TRUE)
  d_net$eval()
  x <- torch::torch_randn(c(6, dd))
  base <- as.numeric(d_net(x)$to(dtype = torch::torch_float()))

  # changing row 6 must change the score of row 1: that is the whole point of
  # inter-sample attention (and is false for the plain column-attention critic)
  x2 <- x$clone()
  x2[6, ] <- torch::torch_randn(c(dd))
  perturbed <- as.numeric(d_net(x2)$to(dtype = torch::torch_float()))
  expect_false(isTRUE(all.equal(base[1], perturbed[1], tolerance = 1e-6)))

  d_plain <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1,
                                row_attention = FALSE)
  d_plain$eval()
  b1 <- as.numeric(d_plain(x)$to(dtype = torch::torch_float()))
  b2 <- as.numeric(d_plain(x2)$to(dtype = torch::torch_float()))
  expect_equal(b1[1], b2[1], tolerance = 1e-5)
})


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------

test_that("gradients flow to all TokenDiscriminator parameters", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  d_net <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 2,
                              row_attention = TRUE, pac = 2, extra_inputs = 1)
  out <- d_net(torch::torch_randn(c(6, 2 * (oi_dim(oi) + 1))))
  loss <- out$mean()
  loss$backward()

  pars <- d_net$parameters
  expect_gt(length(pars), 10)
  missing <- names(pars)[vapply(pars, function(p) is.null(p$grad), logical(1))]
  expect_equal(missing, character(0))
})

test_that("gradients flow to all TokenGenerator parameters", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  g_net <- TokenGenerator(8, oi, d_model = 16, n_heads = 2, n_layers = 2)
  out <- g_net(torch::torch_randn(c(6, 8)))
  out$mean()$backward()

  pars <- g_net$parameters
  missing <- names(pars)[vapply(pars, function(p) is.null(p$grad), logical(1))]
  expect_equal(missing, character(0))
})

test_that("TokenDiscriminator is twice differentiable (R1 penalty path)", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  d_net <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 2,
                              attention = "l2")

  pen <- r1_penalty(d_net, torch::torch_randn(c(6, oi_dim(oi))), device = "cpu")
  expect_true(as.logical(torch::torch_isfinite(pen)))
  pen$backward()

  # the output bias legitimately receives no gradient from an R1 term
  pars <- d_net$parameters
  with_grad <- sum(vapply(pars, function(p) !is.null(p$grad), logical(1)))
  expect_gt(with_grad, length(pars) - 2)
})


# --------------------------------------------------------------------------
# permutation equivariance
# --------------------------------------------------------------------------

test_that("TokenDiscriminator is permutation equivariant in the columns", {
  skip_if_not_installed("torch")
  # Three blocks of equal width so the tokenisers can be permuted by copying.
  oi <- list(list(2, "cossin"), list(2, "cossin"), list(2, "cossin"))
  dd <- oi_dim(oi)
  perm <- c(3L, 1L, 2L)   # new position i holds old block perm[i]

  d1 <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 2)
  d2 <- TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 2)
  d2$load_state_dict(d1$state_dict())
  d1$eval(); d2$eval()

  torch::with_no_grad({
    # permute the tokenisers of d2 and zero both column-ID embeddings, which
    # is the "permute the embeddings too" condition in vectorised form
    w <- lapply(seq_along(perm), function(i) d1$tokenisers[[perm[i]]]$weight$clone())
    b <- lapply(seq_along(perm), function(i) d1$tokenisers[[perm[i]]]$bias$clone())
    for (i in seq_along(perm)) {
      d2$tokenisers[[i]]$weight$copy_(w[[i]])
      d2$tokenisers[[i]]$bias$copy_(b[[i]])
    }
    d1$col_embedding$weight$zero_()
    d2$col_embedding$weight$zero_()
  })

  x <- torch::torch_randn(c(5, dd))
  idx <- unlist(lapply(perm, function(j) ((j - 1) * 2 + 1):((j - 1) * 2 + 2)))
  x_perm <- x[, idx]

  o1 <- as.numeric(d1(x)$to(dtype = torch::torch_float()))
  o2 <- as.numeric(d2(x_perm)$to(dtype = torch::torch_float()))
  expect_equal(o1, o2, tolerance = 1e-5)
})


# --------------------------------------------------------------------------
# output heads
# --------------------------------------------------------------------------

test_that("TokenGenerator respects the output-head constraints", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  dd <- oi_dim(oi)
  g_net <- TokenGenerator(8, oi, d_model = 16, n_heads = 2, n_layers = 2)
  g_net$eval()

  out <- g_net(torch::torch_randn(c(32, 8)))
  expect_equal(out$shape, c(32, dd))

  m <- as.matrix(out$detach()$to(dtype = torch::torch_float()))

  # block 1: linear -> tanh
  expect_true(all(abs(m[, 1]) <= 1 + 1e-6))
  # block 2: mode_specific, columns 2:5 are a simplex, column 6 a tanh value
  expect_equal(rowSums(m[, 2:5]), rep(1, 32), tolerance = 1e-5)
  expect_true(all(abs(m[, 6]) <= 1 + 1e-6))
  # block 3: softmax over columns 7:9
  expect_equal(rowSums(m[, 7:9]), rep(1, 32), tolerance = 1e-5)
  # block 5: cossin on the unit circle
  expect_equal(sqrt(m[, 11]^2 + m[, 12]^2), rep(1, 32), tolerance = 1e-5)

  # hard = TRUE makes the categorical blocks one-hot
  hard_out <- as.matrix(g_net(torch::torch_randn(c(16, 8)), hard = TRUE)$detach()$
                          to(dtype = torch::torch_float()))
  expect_true(all(abs(hard_out[, 7:9] - round(hard_out[, 7:9])) < 1e-6))
  expect_equal(rowSums(hard_out[, 7:9]), rep(1, 16), tolerance = 1e-5)
})

test_that("TokenGenerator rejects the head-less index type and honours cond_dim", {
  skip_if_not_installed("torch")
  expect_error(
    TokenGenerator(8, list(list(1, "index"))),
    "has no TokenGenerator head"
  )
  expect_error(
    TokenGenerator(8, list(list(3, "cossin"))),
    "must have dimension 2"
  )

  oi <- mixed_output_info()
  g_cond <- TokenGenerator(8, oi, d_model = 16, n_heads = 2, n_layers = 1,
                           cond_dim = 4)
  expect_error(g_cond(torch::torch_randn(c(4, 8))), "expects an input of width 12")
  expect_equal(g_cond(torch::torch_randn(c(4, 12)))$shape, c(4, oi_dim(oi)))
})

test_that("apply_output_heads reproduces the TabularGenerator behaviour", {
  skip_if_not_installed("torch")
  oi <- mixed_output_info()
  dd <- oi_dim(oi)
  raw <- torch::torch_randn(c(6, dd))

  soft <- as.matrix(apply_output_heads(raw, oi, tau = 0.2, hard = FALSE,
                                       training = TRUE)$detach()$
                      to(dtype = torch::torch_float()))
  expect_equal(rowSums(soft[, 7:9]), rep(1, 6), tolerance = 1e-5)
  expect_equal(sqrt(soft[, 11]^2 + soft[, 12]^2), rep(1, 6), tolerance = 1e-5)
  # the quantile head is the identity
  expect_equal(soft[, 10],
               as.matrix(raw$to(dtype = torch::torch_float()))[, 10],
               tolerance = 1e-6)

  # TabularGenerator still produces valid blocks after the refactor
  tg <- TabularGenerator(noise_dim = 8, output_info = oi,
                         hidden_units = list(32), normalization = "layer")
  tg$eval()
  tout <- as.matrix(tg(torch::torch_randn(c(6, 8)))$detach()$
                      to(dtype = torch::torch_float()))
  expect_equal(dim(tout), c(6L, as.integer(dd)))
  expect_equal(rowSums(tout[, 7:9]), rep(1, 6), tolerance = 1e-5)
  expect_equal(sqrt(tout[, 11]^2 + tout[, 12]^2), rep(1, 6), tolerance = 1e-5)
})


# --------------------------------------------------------------------------
# integration with gan_trainer
# --------------------------------------------------------------------------

toy_tabular <- function(n = 120) {
  set.seed(42)
  data.frame(
    a = stats::rnorm(n),
    b = stats::runif(n),
    g = sample(c("x", "y", "z"), n, replace = TRUE),
    h = sample(c("u", "v"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

test_that("gan_trainer runs with the token architectures", {
  skip_if_not_installed("torch")
  skip_on_cran()

  df <- toy_tabular()
  tr <- data_transformer$new()
  tr$fit(df, discrete_columns = c("g", "h"), continuous_transform = "quantile",
         categorical_encoding = "cossin")
  x <- tr$transform(df)
  oi <- tr$output_info
  dd <- ncol(x)

  # (a) TokenDiscriminator + default generator
  fit_a <- gan_trainer(
    x, discriminator = TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1),
    output_info = oi, noise_dim = 8, batch_size = 20, epochs = 2, plot_progress = FALSE
  )
  expect_s3_class(fit_a, "trained_RGAN")

  # (b) TokenGenerator + default discriminator
  fit_b <- gan_trainer(
    x, generator = TokenGenerator(8, oi, d_model = 16, n_heads = 2, n_layers = 1),
    output_info = oi, noise_dim = 8, batch_size = 20, epochs = 2, plot_progress = FALSE
  )
  expect_s3_class(fit_b, "trained_RGAN")
  synth <- sample_synthetic_data(fit_b, tr)
  expect_equal(ncol(synth), ncol(df))

  # (c) both
  fit_c <- gan_trainer(
    x,
    generator = TokenGenerator(8, oi, d_model = 16, n_heads = 2, n_layers = 1),
    discriminator = TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1),
    output_info = oi, noise_dim = 8, batch_size = 20, epochs = 2, plot_progress = FALSE
  )
  expect_s3_class(fit_c, "trained_RGAN")

  # (d) both, with pac = 10
  fit_d <- gan_trainer(
    x,
    generator = TokenGenerator(8, oi, d_model = 16, n_heads = 2, n_layers = 1),
    discriminator = TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1,
                                       pac = 10),
    output_info = oi, noise_dim = 8, batch_size = 20, epochs = 2, pac = 10,
    plot_progress = FALSE
  )
  expect_s3_class(fit_d, "trained_RGAN")

  # (e) rpgan + zero-centred R1/R2 penalties (double backward through the critic)
  fit_e <- gan_trainer(
    x,
    discriminator = TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1,
                                       attention = "l2"),
    output_info = oi, noise_dim = 8, batch_size = 20, epochs = 2,
    value_function = "rpgan", regularization = list(type = "r1r2", gamma = 10),
    plot_progress = FALSE
  )
  expect_s3_class(fit_e, "trained_RGAN")
  expect_equal(dd, sum(vapply(oi, function(o) as.numeric(o[[1]]), numeric(1))))
})

test_that("gan_trainer runs with row attention", {
  skip_if_not_installed("torch")
  skip_on_cran()

  df <- toy_tabular()
  tr <- data_transformer$new()
  tr$fit(df, discrete_columns = c("g", "h"), continuous_transform = "quantile",
         categorical_encoding = "cossin")
  x <- tr$transform(df)
  oi <- tr$output_info

  fit <- gan_trainer(
    x,
    discriminator = TokenDiscriminator(oi, d_model = 16, n_heads = 2, n_layers = 1,
                                       row_attention = TRUE),
    output_info = oi, noise_dim = 8, batch_size = 20, epochs = 2,
    value_function = "rpgan", regularization = list(type = "r1", gamma = 1),
    plot_progress = FALSE
  )
  expect_s3_class(fit, "trained_RGAN")
})
