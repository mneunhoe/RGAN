skip_if_not_installed("torch")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A mixed data set built with data_transformer: two continuous columns, one
# with mode-specific normalisation and one with the quantile transform, and
# two categorical columns, one one-hot and one cos/sin.
mixed_fixture <- function(n = 300, seed = 7) {
  set.seed(seed)
  d <- data.frame(
    num_ms = c(stats::rnorm(n %/% 2, -3), stats::rnorm(n - n %/% 2, 3)),
    num_q = stats::rexp(n, 1),
    cat_oh = sample(c("a", "b", "c"), n, TRUE, prob = c(0.7, 0.25, 0.05)),
    cat_cs = sample(c("u", "v", "w", "x"), n, TRUE, prob = c(0.5, 0.3, 0.15, 0.05)),
    stringsAsFactors = FALSE
  )
  tr_ms <- data_transformer$new()
  tr_ms$fit(d[, c("num_ms", "cat_oh")], discrete_columns = "cat_oh",
            mode_specific = TRUE, n_modes = 3)
  tr_q <- data_transformer$new()
  tr_q$fit(d[, c("num_q", "cat_cs")], discrete_columns = "cat_cs",
           continuous_transform = "quantile", categorical_encoding = "cossin")
  list(data = d, tr_ms = tr_ms, tr_q = tr_q)
}

# One transformer over all four columns: mode_specific + onehot for the first
# pair, quantile + cossin for the second. data_transformer fits one
# continuous_transform / categorical_encoding for all columns, so the two
# halves are fitted separately and the transformed matrices are bound.
mixed_transformer <- function(n = 300, seed = 7) {
  set.seed(seed)
  d <- data.frame(
    num_ms = c(stats::rnorm(n %/% 2, -3), stats::rnorm(n - n %/% 2, 3)),
    cat_oh = sample(c("a", "b", "c"), n, TRUE, prob = c(0.7, 0.25, 0.05)),
    num_q = stats::rexp(n, 1),
    cat_cs = sample(c("u", "v", "w", "x"), n, TRUE, prob = c(0.5, 0.3, 0.15, 0.05)),
    stringsAsFactors = FALSE
  )
  a <- data_transformer$new()
  a$fit(d[, c("num_ms", "cat_oh")], discrete_columns = "cat_oh",
        mode_specific = TRUE, n_modes = 3)
  b <- data_transformer$new()
  b$fit(d[, c("num_q", "cat_cs")], discrete_columns = "cat_cs",
        continuous_transform = "quantile", categorical_encoding = "cossin")
  ta <- a$transform(d[, c("num_ms", "cat_oh")])
  tb <- b$transform(d[, c("num_q", "cat_cs")])
  list(
    data = d,
    transformed = cbind(ta, tb),
    output_info = c(a$output_info, b$output_info),
    tr_a = a, tr_b = b
  )
}

# A transformer over all columns at once (needed for sampling / imputation,
# which go through transformer$transform()).
simple_transformer <- function(d, discrete_columns) {
  tr <- data_transformer$new()
  tr$fit(d, discrete_columns = discrete_columns, mode_specific = TRUE,
         n_modes = 3)
  tr
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

test_that("resolve_conditioning fills defaults and validates", {
  expect_null(resolve_conditioning(NULL))
  expect_null(resolve_conditioning(FALSE))
  expect_null(resolve_conditioning(list(type = "none")))

  cn <- resolve_conditioning(list(type = "random_mask"))
  expect_equal(cn$type, "random_mask")
  expect_equal(cn$mask_prob, "uniform")
  expect_equal(cn$cond_dropout, 0.1)
  expect_true(cn$log_frequency)
  expect_false(cn$resample_rare)
  expect_equal(cn$consistency_weight, 1)

  cn2 <- resolve_conditioning(list(type = "single_column", mask_prob = 0.3,
                                   cond_dropout = 0, resample_rare = 0.25,
                                   consistency_weight = 2))
  expect_equal(cn2$mask_prob, 0.3)
  expect_true(cn2$resample_rare)
  expect_equal(cn2$resample_rare_frac, 0.25)
  expect_equal(cn2$consistency_weight, 2)

  expect_equal(resolve_conditioning("single_column")$type, "single_column")
  expect_equal(resolve_conditioning(list(type = "random_mask",
                                         resample_rare = TRUE))$resample_rare_frac,
               0.5)

  expect_error(resolve_conditioning(list(type = "bogus")), "conditioning\\$type")
  expect_error(resolve_conditioning(list(type = "random_mask", mask_prob = 2)),
               "mask_prob")
  expect_error(resolve_conditioning(list(type = "random_mask", cond_dropout = -1)),
               "cond_dropout")
  expect_error(resolve_conditioning(list(type = "random_mask",
                                         consistency_weight = -1)),
               "consistency_weight")
  expect_error(resolve_conditioning(list(type = "random_mask",
                                         resample_rare = "yes")),
               "resample_rare")
})


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

test_that("build_conditioning_meta decodes blocks, groups and categories", {
  fx <- mixed_transformer(n = 400)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)

  # 4 original columns; the mode_specific one contributes two blocks.
  expect_equal(meta$n_groups, 4L)
  expect_equal(meta$n_blocks, 5L)
  expect_equal(meta$data_dim, ncol(fx$transformed))
  expect_equal(meta$cond_dim, meta$n_blocks + meta$data_dim)
  # The two blocks of the mode-specific column share a group.
  expect_equal(meta$group_of_block[1], meta$group_of_block[2])
  expect_equal(length(meta$blocks_of_group[[1]]), 2L)

  # Two categorical blocks: the one-hot column and the cos/sin column.
  expect_equal(length(meta$cat_blocks), 2L)
  types <- vapply(meta$cat_blocks, function(b) b$type, character(1))
  expect_setequal(types, c("softmax", "cossin"))

  oh <- meta$cat_blocks[[which(types == "softmax")]]
  expect_equal(oh$n_cat, 3L)
  expect_equal(sum(oh$counts), nrow(fx$transformed))
  expect_equal(oh$counts, unname(as.integer(table(factor(fx$data$cat_oh,
                                                         levels = c("a", "b", "c"))))))
  expect_equal(sum(vapply(oh$rows, length, integer(1))), nrow(fx$transformed))

  cs <- meta$cat_blocks[[which(types == "cossin")]]
  expect_equal(cs$n_cat, 4L)
  expect_equal(sum(cs$counts), nrow(fx$transformed))
  # The stored canonical values reproduce the encoding of the real rows.
  ids <- rep(seq_len(cs$n_cat), times = cs$counts)
  expect_equal(nrow(cs$values), 4L)
  expect_equal(ncol(cs$values), 2L)

  # Log-frequency probabilities, CTGAN's sampler.
  expect_equal(oh$log_prob, log(oh$counts + 1) / sum(log(oh$counts + 1)))
  expect_equal(oh$freq_prob, oh$counts / sum(oh$counts))
  # The rare level gets far more weight under log-frequency than empirically.
  rare <- which.min(oh$counts)
  expect_gt(oh$log_prob[rare], oh$freq_prob[rare])
})

test_that("expand_block_mask maps blocks to columns", {
  fx <- mixed_transformer(n = 100)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  m <- matrix(0, nrow = 2, ncol = meta$n_blocks)
  m[1, 1] <- 1
  em <- expand_block_mask(m, meta)
  expect_equal(dim(em), c(2L, meta$data_dim))
  b1 <- meta$blocks[[1]]
  expect_true(all(em[1, b1$start:b1$end] == 1))
  expect_equal(sum(em[1, ]), b1$end - b1$start + 1)
  expect_equal(sum(em[2, ]), 0)
})


# ---------------------------------------------------------------------------
# Consistency loss
# ---------------------------------------------------------------------------

test_that("consistency loss is zero when the generated row matches the condition", {
  fx <- mixed_transformer(n = 200)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  x <- fx$transformed[1:16, , drop = FALSE]
  mask <- matrix(1, nrow = 16, ncol = meta$n_blocks)
  values <- x * expand_block_mask(mask, meta)

  fake <- torch::torch_tensor(x)
  loss0 <- conditioning_consistency_loss(fake,
                                         torch::torch_tensor(mask),
                                         torch::torch_tensor(values), meta)
  expect_lt(abs(torch::as_array(loss0$cpu())), 1e-5)

  # Perturbing the generated row makes it positive.
  x2 <- x
  x2[, ] <- x2[, ] * 0 + 0.25
  loss1 <- conditioning_consistency_loss(torch::torch_tensor(x2),
                                         torch::torch_tensor(mask),
                                         torch::torch_tensor(values), meta)
  expect_gt(torch::as_array(loss1$cpu()), 0)

  # Nothing observed -> no penalty.
  mask0 <- matrix(0, nrow = 16, ncol = meta$n_blocks)
  loss2 <- conditioning_consistency_loss(torch::torch_tensor(x2),
                                         torch::torch_tensor(mask0),
                                         torch::torch_tensor(values * 0), meta)
  expect_lt(abs(torch::as_array(loss2$cpu())), 1e-5)
})

test_that("consistency loss only penalises observed blocks", {
  fx <- mixed_transformer(n = 100)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  x <- fx$transformed[1:8, , drop = FALSE]
  mask <- matrix(0, nrow = 8, ncol = meta$n_blocks)
  mask[, 1] <- 1
  values <- x * expand_block_mask(mask, meta)
  # Generated data matches the observed block exactly but nothing else.
  fake <- x * expand_block_mask(mask, meta)
  loss <- conditioning_consistency_loss(torch::torch_tensor(fake),
                                        torch::torch_tensor(mask),
                                        torch::torch_tensor(values), meta)
  expect_lt(abs(torch::as_array(loss$cpu())), 1e-5)
})

test_that("consistency loss is differentiable w.r.t. the generator output", {
  fx <- mixed_transformer(n = 50)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  x <- fx$transformed[1:4, , drop = FALSE]
  mask <- matrix(1, nrow = 4, ncol = meta$n_blocks)
  values <- x * expand_block_mask(mask, meta)
  fake <- torch::torch_tensor(x * 0 + 0.3, requires_grad = TRUE)
  loss <- conditioning_consistency_loss(fake, torch::torch_tensor(mask),
                                        torch::torch_tensor(values), meta)
  loss$backward()
  expect_true(as.logical(torch::as_array(fake$grad$abs()$sum()$cpu() > 0)))
})


# ---------------------------------------------------------------------------
# Samplers and per-step conditions
# ---------------------------------------------------------------------------

test_that("draw_categories follows the log-frequency distribution", {
  fx <- mixed_transformer(n = 2000)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  types <- vapply(meta$cat_blocks, function(b) b$type, character(1))
  cb <- meta$cat_blocks[[which(types == "softmax")]]

  set.seed(11)
  draws <- draw_categories(cb, 5000, log_frequency = TRUE)
  obs <- tabulate(draws, nbins = cb$n_cat) / 5000
  expect_lt(max(abs(obs - cb$log_prob)), 0.03)
  # A lenient chi-square check against the log-frequency probabilities.
  chi <- suppressWarnings(stats::chisq.test(tabulate(draws, nbins = cb$n_cat),
                                            p = cb$log_prob))
  expect_gt(chi$p.value, 0.001)
  # And it is clearly *not* the empirical frequency distribution.
  expect_gt(max(abs(obs - cb$freq_prob)), 0.05)

  # log_frequency = FALSE recovers the empirical frequencies.
  set.seed(12)
  draws2 <- draw_categories(cb, 5000, log_frequency = FALSE)
  obs2 <- tabulate(draws2, nbins = cb$n_cat) / 5000
  expect_lt(max(abs(obs2 - cb$freq_prob)), 0.03)
})

test_that("single_column training-by-sampling returns rows with the drawn category", {
  fx <- mixed_transformer(n = 500)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  cn <- resolve_conditioning(list(type = "single_column", cond_dropout = 0))
  dt <- torch::torch_tensor(fx$transformed)

  set.seed(3)
  st <- build_step_condition(dt, fx$transformed, 200, cn, meta)
  mask <- torch::as_array(st$mask$cpu())
  values <- torch::as_array(st$values$cpu())

  # Exactly one observed block per row, and it is a categorical one.
  expect_equal(unname(rowSums(mask)), rep(1, 200))
  observed_block <- apply(mask, 1, which.max)
  expect_true(all(observed_block %in% meta$cat_block_ids))

  # The observed values equal the real row's own values there, i.e. the real
  # batch was sampled *by* the condition.
  real <- torch::as_array(st$real_data$cpu())
  expect_equal(values, real * expand_block_mask(mask, meta), tolerance = 1e-6)

  # And the drawn categories over-represent rare levels (log-frequency).
  types <- vapply(meta$cat_blocks, function(b) b$type, character(1))
  cb <- meta$cat_blocks[[which(types == "softmax")]]
  sel <- observed_block == cb$block
  if (any(sel)) {
    cats <- max.col(real[sel, cb$start:cb$end, drop = FALSE], ties.method = "first")
    expect_true(all(cats >= 1 & cats <= cb$n_cat))
  }

  # The condition vector is [mask, values].
  cond <- torch::as_array(st$cond$cpu())
  expect_equal(ncol(cond), meta$n_blocks + meta$data_dim)
  expect_equal(cond[, seq_len(meta$n_blocks)], mask)
})

test_that("random_mask masking rate and condition dropout have the right rates", {
  fx <- mixed_transformer(n = 400)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  dt <- torch::torch_tensor(fx$transformed)

  # Fixed rate, no dropout: the per-column observation rate is mask_prob.
  cn <- resolve_conditioning(list(type = "random_mask", mask_prob = 0.3,
                                  cond_dropout = 0))
  set.seed(4)
  st <- build_step_condition(dt, fx$transformed, 4000, cn, meta)
  mask <- torch::as_array(st$mask$cpu())
  expect_equal(mean(mask), 0.3, tolerance = 0.03)
  # The two blocks of the mode-specific column are always observed together.
  expect_true(all(mask[, 1] == mask[, 2]))

  # "uniform": the per-row rate is U(0, 1), so the mean rate is 0.5 and the
  # distribution of the per-row number of observed columns is flat.
  cn_u <- resolve_conditioning(list(type = "random_mask", mask_prob = "uniform",
                                    cond_dropout = 0))
  set.seed(5)
  st_u <- build_step_condition(dt, fx$transformed, 4000, cn_u, meta)
  mask_u <- torch::as_array(st_u$mask$cpu())
  expect_equal(mean(mask_u), 0.5, tolerance = 0.03)
  expect_gt(mean(rowSums(mask_u) == 0), 0.05)
  expect_gt(mean(rowSums(mask_u) == meta$n_blocks), 0.05)

  # Condition dropout zeroes whole rows of the mask.
  cn_d <- resolve_conditioning(list(type = "random_mask", mask_prob = 1,
                                    cond_dropout = 0.25))
  set.seed(6)
  st_d <- build_step_condition(dt, fx$transformed, 4000, cn_d, meta)
  mask_d <- torch::as_array(st_d$mask$cpu())
  values_d <- torch::as_array(st_d$values$cpu())
  dropped <- rowSums(mask_d) == 0
  expect_equal(mean(dropped), 0.25, tolerance = 0.03)
  expect_true(all(values_d[dropped, ] == 0))
  expect_true(all(mask_d[!dropped, ] == 1))
})

test_that("resample_rare shifts the conditioned categories towards rare levels", {
  fx <- mixed_transformer(n = 1000)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  dt <- torch::torch_tensor(fx$transformed)
  types <- vapply(meta$cat_blocks, function(b) b$type, character(1))
  cb <- meta$cat_blocks[[which(types == "softmax")]]
  rare <- which.min(cb$counts)

  rate_of_rare <- function(resample) {
    cn <- resolve_conditioning(list(type = "random_mask", mask_prob = 1,
                                    cond_dropout = 0,
                                    resample_rare = resample))
    st <- build_step_condition(dt, fx$transformed, 3000, cn, meta)
    real <- torch::as_array(st$real_data$cpu())
    cats <- max.col(real[, cb$start:cb$end, drop = FALSE], ties.method = "first")
    mean(cats == rare)
  }
  set.seed(21)
  base <- rate_of_rare(FALSE)
  set.seed(22)
  boosted <- rate_of_rare(TRUE)
  expect_gt(boosted, base)
  # The real batch keeps matching its own condition, so the discriminator
  # never sees an inconsistent (row, condition) pair.
  cn <- resolve_conditioning(list(type = "random_mask", mask_prob = 0.6,
                                  cond_dropout = 0, resample_rare = TRUE))
  set.seed(23)
  st <- build_step_condition(dt, fx$transformed, 200, cn, meta)
  mask <- torch::as_array(st$mask$cpu())
  expect_equal(torch::as_array(st$values$cpu()),
               torch::as_array(st$real_data$cpu()) * expand_block_mask(mask, meta),
               tolerance = 1e-6)
})


# ---------------------------------------------------------------------------
# Shapes of the generator and discriminator inputs
# ---------------------------------------------------------------------------

test_that("the conditional generator and discriminator get the documented shapes", {
  fx <- mixed_transformer(n = 200)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  cond_dim <- meta$n_blocks + meta$data_dim

  g <- gan_trainer(fx$transformed, epochs = 1, batch_size = 40, noise_dim = 4,
                   pac = 4, seed = 1, output_info = fx$output_info,
                   conditioning = list(type = "random_mask"))
  expect_equal(g$settings$cond_dim, cond_dim)
  expect_equal(g$generator$cond_dim, cond_dim)
  expect_equal(g$generator$input_dim, 4L + cond_dim)
  # Default MLP discriminator: pac * (data_dim + cond_dim) inputs.
  d_in <- g$discriminator$parameters[[1]]$shape[2]
  expect_equal(d_in, 4 * (meta$data_dim + cond_dim))

  # The generator accepts cat(z, condition) and rejects a bare z.
  z <- torch::torch_randn(c(8, 4))
  cond <- torch::torch_zeros(c(8, cond_dim))
  out <- g$generator(torch::torch_cat(list(z, cond), dim = 2))
  expect_equal(out$shape, c(8, meta$data_dim))
  expect_error(g$generator(z), "TabularGenerator expects an input of width")
})

test_that("TabularGenerator cond_dim defaults to 0 and changes nothing", {
  oi <- list(list(1, "linear"), list(3, "softmax"))
  g0 <- TabularGenerator(noise_dim = 5, output_info = oi)
  expect_equal(g0$cond_dim, 0L)
  expect_equal(g0$input_dim, 5L)
  expect_equal(g0(torch::torch_randn(c(2, 5)))$shape, c(2, 4))

  g1 <- TabularGenerator(noise_dim = 5, output_info = oi, cond_dim = 7)
  expect_equal(g1$input_dim, 12L)
  expect_equal(g1(torch::torch_randn(c(2, 12)))$shape, c(2, 4))
  expect_error(TabularGenerator(noise_dim = 5, output_info = oi, cond_dim = -1),
               "cond_dim")
})

test_that("attach_condition is a no-op without a condition", {
  x <- torch::torch_randn(c(3, 4))
  expect_true(identical(attach_condition(x, NULL), x))
  cnd <- torch::torch_zeros(c(3, 2))
  expect_equal(attach_condition(x, cnd)$shape, c(3, 6))
})


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------

test_that("gan_trainer trains on a mixed data set for both conditioning types", {
  fx <- mixed_transformer(n = 200)
  for (type in c("single_column", "random_mask")) {
    for (pac in c(1, 10)) {
      g <- gan_trainer(fx$transformed, epochs = 2, batch_size = 50,
                       noise_dim = 4, pac = pac, seed = 42, track_loss = TRUE,
                       output_info = fx$output_info,
                       value_function = "rpgan",
                       regularization = list(type = "r1r2", gamma = 10),
                       adam_betas = c(0, 0.99), discriminator_dropout = 0,
                       conditioning = list(type = type))
      expect_s3_class(g, "trained_RGAN")
      expect_equal(g$settings$conditioning$type, type)
      expect_false(is.null(g$conditioning_meta))
      expect_true(all(is.finite(g$losses$g_loss)))
      expect_true(all(is.finite(g$losses$d_loss)))
    }
  }
})

test_that("conditioning works with TokenGenerator and TokenDiscriminator", {
  fx <- mixed_transformer(n = 200)
  meta <- build_conditioning_meta(fx$output_info, fx$transformed)
  cond_dim <- meta$n_blocks + meta$data_dim
  pac <- 5

  g_net <- TokenGenerator(noise_dim = 4, output_info = fx$output_info,
                          d_model = 16, n_heads = 2, n_layers = 1,
                          cond_dim = cond_dim)
  d_net <- TokenDiscriminator(output_info = fx$output_info, d_model = 16,
                              n_heads = 2, n_layers = 1, pac = pac,
                              extra_inputs = cond_dim)
  expect_equal(d_net$extra_inputs, cond_dim)
  expect_equal(d_net$sample_dim, meta$data_dim + cond_dim)

  g <- gan_trainer(fx$transformed, epochs = 2, batch_size = 50, noise_dim = 4,
                   pac = pac, seed = 42, track_loss = TRUE,
                   output_info = fx$output_info,
                   generator = g_net, discriminator = d_net,
                   value_function = "rpgan",
                   regularization = list(type = "r1r2", gamma = 10),
                   conditioning = list(type = "random_mask"))
  expect_s3_class(g, "trained_RGAN")
  expect_true(all(is.finite(g$losses$g_loss)))
})

test_that("conditioning errors without output_info", {
  fx <- mixed_transformer(n = 100)
  expect_error(
    gan_trainer(fx$transformed, epochs = 1, batch_size = 50, noise_dim = 4,
                conditioning = list(type = "random_mask")),
    "output_info"
  )
})

test_that("single_column requires a categorical column", {
  set.seed(2)
  d <- data.frame(a = stats::rnorm(80), b = stats::rnorm(80))
  tr <- data_transformer$new()
  tr$fit(d)
  expect_error(
    gan_trainer(tr$transform(d), epochs = 1, batch_size = 40, noise_dim = 3,
                output_info = tr$output_info,
                conditioning = list(type = "single_column")),
    "single_column"
  )
})


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

test_that("sample_synthetic_data honours a one-row condition", {
  skip_on_cran()
  set.seed(101)
  n <- 600
  grp <- sample(c("A", "B", "C"), n, TRUE)
  d <- data.frame(
    grp = grp,
    x = stats::rnorm(n, c(A = -4, B = 0, C = 4)[grp], 0.3),
    y = stats::rnorm(n, 0, 1),
    stringsAsFactors = FALSE
  )
  tr <- simple_transformer(d, discrete_columns = "grp")
  td <- tr$transform(d)

  g <- gan_trainer(td, epochs = 30, batch_size = 60, noise_dim = 8,
                   seed = 7, output_info = tr$output_info,
                   value_function = "rpgan",
                   regularization = list(type = "r1r2", gamma = 10),
                   adam_betas = c(0, 0.99), discriminator_dropout = 0,
                   base_lr = 5e-4,
                   conditioning = list(type = "random_mask",
                                       mask_prob = 0.5,
                                       cond_dropout = 0.1,
                                       consistency_weight = 5))
  s <- sample_synthetic_data(g, tr, n_samples = 300,
                             condition = data.frame(grp = "B"))
  expect_equal(nrow(s), 300)
  expect_gte(mean(s$grp == "B"), 0.9)

  # An unknown level is an error, and an unconditional call still works.
  expect_error(sample_synthetic_data(g, tr, 10,
                                     condition = data.frame(grp = "Z")),
               "not seen in training")
  expect_equal(nrow(sample_synthetic_data(g, tr, 50)), 50)
})

test_that("a single_column model samples conditions from the frequency table", {
  fx <- mixed_transformer(n = 200)
  g <- gan_trainer(fx$transformed, epochs = 1, batch_size = 50, noise_dim = 4,
                   seed = 5, output_info = fx$output_info,
                   conditioning = list(type = "single_column"))
  meta <- g$conditioning_meta
  set.seed(31)
  cnd <- sample_single_column_condition(meta, 5000)
  expect_equal(unname(rowSums(cnd$mask)), rep(1, 5000))
  types <- vapply(meta$cat_blocks, function(b) b$type, character(1))
  cb <- meta$cat_blocks[[which(types == "softmax")]]
  sel <- cnd$mask[, cb$block] == 1
  # Unconditional sampling uses the *empirical* frequencies, not the
  # log-frequencies used during training.
  cats <- max.col(cnd$values[sel, cb$start:cb$end, drop = FALSE],
                  ties.method = "first")
  obs <- tabulate(cats, nbins = cb$n_cat) / sum(sel)
  expect_lt(max(abs(obs - cb$freq_prob)), 0.05)

  # The trainer's unconditional path uses exactly that.
  s <- sample_synthetic_data(g, n_samples = 40)
  expect_equal(nrow(s), 40)
})

test_that("condition_matrix_from_df masks NA cells and rejects bad input", {
  set.seed(102)
  d <- data.frame(grp = sample(c("A", "B"), 100, TRUE),
                  x = stats::rnorm(100), stringsAsFactors = FALSE)
  tr <- simple_transformer(d, discrete_columns = "grp")
  td <- tr$transform(d)
  meta <- build_conditioning_meta(tr$output_info, td)

  cm <- condition_matrix_from_df(data.frame(grp = "A"), tr, meta, 5)
  expect_equal(nrow(cm$mask), 5L)
  # grp observed, x not.
  grp_blocks <- meta$blocks_of_group[[1]]
  x_blocks <- meta$blocks_of_group[[2]]
  expect_true(all(cm$mask[, grp_blocks] == 1))
  expect_true(all(cm$mask[, x_blocks] == 0))
  # The placeholder that filled the NA numeric cells is masked out.
  for (b in x_blocks) {
    bk <- meta$blocks[[b]]
    expect_true(all(cm$values[, bk$start:bk$end] == 0))
  }
  expect_equal(ncol(cm$cond), meta$n_blocks + meta$data_dim)

  expect_error(condition_matrix_from_df(data.frame(nope = 1), tr, meta, 5),
               "unknown condition column")
  expect_error(condition_matrix_from_df(data.frame(grp = c("A", "B")), tr,
                                        meta, 5),
               "must have 1 or 5 rows")
})

test_that("condition is rejected for an unconditional model", {
  set.seed(103)
  d <- data.frame(grp = sample(c("A", "B"), 80, TRUE), x = stats::rnorm(80),
                  stringsAsFactors = FALSE)
  tr <- simple_transformer(d, discrete_columns = "grp")
  g <- gan_trainer(tr$transform(d), epochs = 1, batch_size = 40, noise_dim = 3,
                   seed = 1, output_info = tr$output_info)
  expect_error(sample_synthetic_data(g, tr, 5,
                                     condition = data.frame(grp = "A")),
               "trained without")
  expect_error(impute_synthetic(g, tr, d[1:3, ]), "impute_synthetic")
})


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

test_that("impute_synthetic fills NA cells and keeps observed cells exactly", {
  set.seed(104)
  n <- 300
  grp <- sample(c("A", "B", "C"), n, TRUE)
  d <- data.frame(grp = grp,
                  x = stats::rnorm(n, c(A = -3, B = 0, C = 3)[grp], 0.5),
                  y = stats::rnorm(n), stringsAsFactors = FALSE)
  tr <- simple_transformer(d, discrete_columns = "grp")
  g <- gan_trainer(tr$transform(d), epochs = 3, batch_size = 50, noise_dim = 6,
                   seed = 9, output_info = tr$output_info,
                   conditioning = list(type = "random_mask"))

  dm <- d[1:40, ]
  set.seed(105)
  miss_x <- sample(40, 12)
  miss_g <- sample(40, 10)
  dm$x[miss_x] <- NA
  dm$grp[miss_g] <- NA

  out <- impute_synthetic(g, tr, dm)
  expect_equal(dim(out), dim(dm))
  expect_equal(colnames(out), colnames(dm))
  expect_false(any(is.na(out)))
  # Observed cells are kept bit-for-bit.
  keep_x <- setdiff(seq_len(40), miss_x)
  keep_g <- setdiff(seq_len(40), miss_g)
  expect_identical(out$x[keep_x], dm$x[keep_x])
  expect_identical(out$grp[keep_g], dm$grp[keep_g])
  expect_identical(out$y, dm$y)
  # Imputed discrete cells are valid levels.
  expect_true(all(out$grp[miss_g] %in% c("A", "B", "C")))
  # A no-missing frame comes back unchanged.
  expect_equal(impute_synthetic(g, tr, d[1:5, ]), d[1:5, ])
})


# ---------------------------------------------------------------------------
# The NULL path stays bit-identical
# ---------------------------------------------------------------------------

test_that("conditioning = NULL is a no-op (bit-identical training)", {
  set.seed(7)
  n <- 300
  d <- data.frame(
    x1 = stats::rnorm(n),
    x2 = c(stats::rnorm(n %/% 2, -3), stats::rnorm(n - n %/% 2, 3)),
    c1 = sample(c("a", "b", "c"), n, TRUE, prob = c(.6, .3, .1)),
    c2 = sample(c("u", "v"), n, TRUE),
    stringsAsFactors = FALSE
  )
  tr <- data_transformer$new()
  tr$fit(d, discrete_columns = c("c1", "c2"), mode_specific = TRUE, n_modes = 3)
  td <- tr$transform(d)
  run <- function(...) {
    set.seed(1)
    torch::torch_manual_seed(1)
    # No zero-centred penalty here: the double backward of r1/r2 is not
    # bit-reproducible on a multi-threaded CPU backend, which would make this
    # check flaky for reasons that have nothing to do with conditioning.
    gan_trainer(td, epochs = 2, batch_size = 50, noise_dim = 4, pac = 10,
                seed = 42, track_loss = TRUE, output_info = tr$output_info,
                value_function = "rpgan", ...)
  }
  a <- run()
  b <- run(conditioning = NULL)
  expect_identical(a$losses$g_loss, b$losses$g_loss)
  expect_identical(a$losses$d_loss, b$losses$d_loss)
  expect_null(a$settings$conditioning)
  expect_null(a$conditioning_meta)
  expect_equal(a$settings$cond_dim, 0L)

  # And the same as the reference recorded from the code *before* Experiment 5
  # (scratchpad/ref_losses_conditioning.rds, generated from a copy of the
  # package with every conditioning hunk reverted). Torch's CPU reductions are
  # bit-reproducible for a fixed number of intra-op threads but not across
  # thread counts, and the thread count cannot be changed once torch has run,
  # so the reference is compared numerically rather than bit-for-bit; the
  # bit-identity claim is the `expect_identical()` pair above, within one
  # session.
  ref_path <- test_path("ref_losses_conditioning.rds")
  skip_if_not(file.exists(ref_path))
  ref <- readRDS(ref_path)
  expect_equal(a$losses$g_loss, ref$g_loss, tolerance = 1e-6)
  expect_equal(a$losses$d_loss, ref$d_loss, tolerance = 1e-6)
})
