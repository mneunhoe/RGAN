test_that("flow_time_embedding has the right shape and is bounded", {
  t <- torch::torch_rand(7)
  e <- flow_time_embedding(t, 16)
  expect_equal(dim(e), c(7, 16))
  expect_lte(as.numeric(torch::torch_max(torch::torch_abs(e))), 1 + 1e-5)

  # odd widths are zero padded in the last position
  e_odd <- flow_time_embedding(t, 9)
  expect_equal(dim(e_odd), c(7, 9))
  expect_equal(as.numeric(torch::torch_sum(torch::torch_abs(e_odd[, 9]))), 0)

  # a column vector of times is accepted
  e2 <- flow_time_embedding(t$unsqueeze(2), 16)
  expect_equal(dim(e2), c(7, 16))

  # different times give different embeddings
  e3 <- flow_time_embedding(torch::torch_tensor(c(0, 0.5, 1)), 8)
  expect_gt(as.numeric(torch::torch_sum(torch::torch_abs(e3[1, ] - e3[3, ]))), 0)
})


test_that("FlowVectorField returns (batch, data_dim) and is differentiable", {
  v <- FlowVectorField(data_dim = 6, hidden_units = list(32, 16),
                       time_embed_dim = 8)
  x <- torch::torch_randn(5, 6)
  t <- torch::torch_rand(5)
  out <- v(x, t)
  expect_equal(dim(out), c(5, 6))
  expect_true(all(is.finite(as.matrix(out))))

  # plain numeric times and a scalar time both work
  expect_equal(dim(v(x, runif(5))), c(5, 6))
  expect_equal(dim(v(x, torch::torch_tensor(0.3))), c(5, 6))

  # gradients reach every parameter
  loss <- torch::torch_mean(v(x, t)^2)
  loss$backward()
  grads <- vapply(v$parameters, function(p) !is.null(p$grad), logical(1))
  expect_true(all(grads))

  # equal widths use an identity shortcut, unequal widths a linear one
  v2 <- FlowVectorField(data_dim = 3, hidden_units = list(8, 8),
                        time_embed_dim = 4, activation = "relu", dropout = 0.1)
  expect_equal(dim(v2(torch::torch_randn(2, 3), torch::torch_rand(2))), c(2, 3))
})


test_that("flow_matching_trainer runs and the loss decreases on average", {
  set.seed(42)
  x <- cbind(rnorm(600), rnorm(600, 2, 0.5))
  fm <- flow_matching_trainer(
    x, epochs = 3, batch_size = 100, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(2, list(64, 64), time_embed_dim = 16))

  expect_s3_class(fm, "trained_RFlow")
  expect_equal(length(fm$losses), 3)
  expect_true(all(is.finite(fm$losses)))
  expect_equal(fm$settings$data_dim, 2)
  expect_equal(fm$settings$steps, 3 * 6)
  expect_output(print(fm), "Trained flow-matching model")

  # over a longer run the loss trend must be downwards
  fm2 <- flow_matching_trainer(
    x, epochs = 40, batch_size = 100, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(2, list(64, 64), time_embed_dim = 16))
  first <- mean(fm2$losses[1:5])
  last <- mean(fm2$losses[36:40])
  expect_lt(last, first)
  expect_lt(coef(stats::lm(fm2$losses ~ seq_along(fm2$losses)))[2], 0)
})


test_that("both paths and both time samplings train", {
  set.seed(3)
  x <- matrix(rnorm(200), 100, 2)
  for (pth in c("ot", "vp")) {
    for (ts in c("uniform", "logit_normal")) {
      fm <- flow_matching_trainer(
        x, epochs = 2, batch_size = 50, path = pth, time_sampling = ts,
        lr_schedule = "constant", ema_decay = 0, seed = 1, verbose = FALSE,
        vector_field = FlowVectorField(2, list(16), time_embed_dim = 4))
      expect_true(all(is.finite(fm$losses)))
      expect_equal(fm$settings$path, pth)
    }
  }
})


test_that("sample_flow shapes, solvers, determinism and trajectories", {
  set.seed(7)
  x <- matrix(rnorm(200), 100, 2)
  fm <- flow_matching_trainer(
    x, epochs = 3, batch_size = 50, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(2, list(32), time_embed_dim = 8))

  s_e <- sample_flow(fm, n_samples = 17, n_steps = 5, solver = "euler")
  s_h <- sample_flow(fm, n_samples = 17, n_steps = 5, solver = "heun")
  expect_true(is.matrix(s_e) && is.matrix(s_h))
  expect_equal(dim(s_e), c(17, 2))
  expect_equal(dim(s_h), c(17, 2))
  expect_true(all(is.finite(s_e)) && all(is.finite(s_h)))

  # deterministic given z, and the two solvers differ
  z <- torch::torch_randn(11, 2)
  a1 <- sample_flow(fm, n_steps = 7, solver = "heun", z = z)
  a2 <- sample_flow(fm, n_steps = 7, solver = "heun", z = z)
  expect_equal(a1, a2)
  b1 <- sample_flow(fm, n_steps = 7, solver = "euler", z = z)
  expect_false(isTRUE(all.equal(a1, b1)))

  # a numeric matrix is accepted as z too
  zm <- matrix(rnorm(22), 11, 2)
  expect_equal(dim(sample_flow(fm, n_steps = 3, z = zm)), c(11, 2))

  tr <- sample_flow(fm, n_samples = 4, n_steps = 6, return_trajectory = TRUE)
  expect_named(tr, c("x", "trajectory", "t"))
  expect_equal(length(tr$trajectory), 7)
  expect_equal(dim(tr$trajectory[[1]]), c(4, 2))
  expect_equal(tr$trajectory[[7]], tr$x)
  expect_equal(length(tr$t), 7)
})


test_that("flow_pairs returns matched (z, x) matrices in batches", {
  set.seed(11)
  x <- matrix(rnorm(200), 100, 2)
  fm <- flow_matching_trainer(
    x, epochs = 2, batch_size = 50, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(2, list(16), time_embed_dim = 4))

  p <- flow_pairs(fm, n_pairs = 25, n_steps = 4, batch_size = 10, seed = 5)
  expect_named(p, c("z", "x"))
  expect_equal(dim(p$z), c(25, 2))
  expect_equal(dim(p$x), c(25, 2))
  expect_true(all(is.finite(p$z)) && all(is.finite(p$x)))

  # the pair map is the teacher ODE: re-integrating z reproduces x
  again <- sample_flow(fm, n_steps = 4, solver = "heun", z = p$z)
  expect_equal(again, p$x, tolerance = 1e-5)

  # same seed, same pairs
  p2 <- flow_pairs(fm, n_pairs = 25, n_steps = 4, batch_size = 10, seed = 5)
  expect_equal(p2$z, p$z)
  expect_equal(p2$x, p$x)
})


test_that("per_type_weight builds mean-1 block weights and trains", {
  # two numeric columns, a 4-level one-hot block, and a cos/sin pair
  output_info <- list(list(1, "quantile"), list(1, "linear"),
                      list(4, "softmax"), list(2, "cossin"))
  data_dim <- 8
  w <- flow_loss_weights(output_info, data_dim)
  expect_equal(length(w), data_dim)
  expect_equal(mean(w), 1)
  # within a block all weights are equal, and wide blocks get small weights
  expect_equal(w[3], w[6])
  expect_lt(w[3], w[1])
  expect_equal(w[3] * 4, w[1], tolerance = 1e-8)
  # each block contributes the same total weight
  totals <- c(w[1], w[2], sum(w[3:6]) / 1, sum(w[7:8]))
  expect_equal(unname(totals[1]), unname(totals[3]), tolerance = 1e-8)
  expect_equal(unname(totals[1]), unname(totals[4]), tolerance = 1e-8)

  # unusable output_info falls back to uniform weights
  expect_equal(flow_loss_weights(NULL, 5), rep(1, 5))
  expect_equal(flow_loss_weights(list(list(3, "softmax")), 5), rep(1, 5))

  set.seed(13)
  X <- matrix(rnorm(80 * data_dim), 80, data_dim)
  fm <- flow_matching_trainer(
    X, output_info = output_info, per_type_weight = TRUE,
    epochs = 2, batch_size = 40, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(data_dim, list(32), time_embed_dim = 8))
  expect_true(all(is.finite(fm$losses)))
  expect_equal(fm$settings$loss_weights, w)

  fm_off <- flow_matching_trainer(
    X, output_info = output_info, per_type_weight = FALSE,
    epochs = 1, batch_size = 40, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(data_dim, list(32), time_embed_dim = 8))
  expect_equal(fm_off$settings$loss_weights, rep(1, data_dim))
})


test_that("the teacher recovers a 2D toy distribution", {
  skip_on_cran()
  x <- scale(sample_toydata(n = 1000, seed = 20211111))
  x <- matrix(as.numeric(x), ncol = 2)

  t0 <- Sys.time()
  fm <- flow_matching_trainer(
    x, epochs = 300, batch_size = 100, base_lr = 1e-3, seed = 1,
    verbose = FALSE,
    vector_field = FlowVectorField(2, list(128, 128, 128), time_embed_dim = 32))
  train_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  expect_lt(train_seconds, 300)

  syn <- sample_flow(fm, n_samples = 1000, n_steps = 50, solver = "heun")

  # mean absolute difference of two 20x20 normalised 2D histograms on a
  # common grid. The Experiment 1 GAN sanity check reached 0.0016
  # (rpgan + R1R2) and 0.0039 (WGAN-GP) with the same statistic.
  rng1 <- range(c(x[, 1], syn[, 1]))
  rng2 <- range(c(x[, 2], syn[, 2]))
  br1 <- seq(rng1[1], rng1[2], length.out = 21)
  br2 <- seq(rng2[1], rng2[2], length.out = 21)
  hist2d <- function(m) {
    i <- cut(m[, 1], br1, include.lowest = TRUE)
    j <- cut(m[, 2], br2, include.lowest = TRUE)
    tab <- table(i, j)
    as.numeric(tab) / sum(tab)
  }
  mad2d <- mean(abs(hist2d(x) - hist2d(syn)))
  message(sprintf("flow teacher 2D histogram MAD = %.5f (train %.1fs)",
                  mad2d, train_seconds))
  expect_lt(mad2d, 0.006)
})


test_that("settings$device is a character label, not a torch_device", {
  set.seed(101)
  x <- matrix(rnorm(200), 100, 2)
  fm <- flow_matching_trainer(
    x, epochs = 2, batch_size = 50, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(2, list(16), time_embed_dim = 4))

  expect_type(fm$settings$device, "character")
  expect_identical(fm$settings$device, "cpu")

  # the label survives a plain RDS round trip (the module does not: see the
  # save_flow()/load_flow() test below)
  f <- tempfile(fileext = ".rds")
  saveRDS(fm$settings, f)
  expect_identical(readRDS(f)$device, "cpu")
  unlink(f)

  # helpers
  expect_identical(flow_device_label("cpu"), "cpu")
  expect_identical(flow_device_label(torch::torch_device("cpu")), "cpu")
  expect_s3_class(flow_resolve_device("cpu"), "torch_device")
  expect_s3_class(flow_resolve_device(NULL), "torch_device")
  expect_identical(flow_device_label(flow_resolve_device(NULL)), "cpu")
  # a device that is not available on this machine degrades to the CPU with a
  # warning rather than erroring
  if (!torch::cuda_is_available()) {
    expect_warning(d <- flow_resolve_device("cuda"))
    expect_identical(flow_device_label(d), "cpu")
  }
})


test_that("save_flow/load_flow round trip: the same z gives the same sample", {
  set.seed(202)
  x <- scale(sample_toydata(n = 400))
  fm <- flow_matching_trainer(
    x, epochs = 2, batch_size = 100, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(2, list(32), time_embed_dim = 8))

  z <- matrix(rnorm(40), 20, 2)
  before <- sample_flow(fm, n_steps = 8, solver = "heun", z = z)
  before_euler <- sample_flow(fm, n_steps = 8, solver = "euler", z = z)

  path <- file.path(tempdir(), paste0("rflow_", as.integer(runif(1, 1, 1e6))))
  expect_invisible(save_flow(fm, path))
  expect_true(file.exists(paste0(path, "_vector_field.pt")))
  expect_true(file.exists(paste0(path, "_settings.rds")))

  fm2 <- load_flow(path)
  expect_s3_class(fm2, "trained_RFlow")
  expect_identical(fm2$settings$data_dim, fm$settings$data_dim)
  expect_identical(fm2$settings$device, "cpu")
  expect_equal(fm2$losses, fm$losses)

  after <- sample_flow(fm2, n_steps = 8, solver = "heun", z = z)
  expect_equal(after, before, tolerance = 1e-6)
  expect_equal(sample_flow(fm2, n_steps = 8, solver = "euler", z = z),
               before_euler, tolerance = 1e-6)

  # flow_pairs works on the reloaded object and reproduces the teacher map
  p <- flow_pairs(fm2, n_pairs = 12, n_steps = 5, batch_size = 5, seed = 3)
  expect_equal(dim(p$x), c(12, 2))
  expect_equal(sample_flow(fm2, n_steps = 5, solver = "heun", z = p$z), p$x,
               tolerance = 1e-5)

  # the device override is honoured on both entry points
  expect_equal(sample_flow(fm2, n_steps = 8, solver = "heun", z = z,
                           device = "cpu"), before, tolerance = 1e-6)
  expect_equal(dim(flow_pairs(fm2, n_pairs = 4, n_steps = 3, seed = 3,
                              device = "cpu")$x), c(4, 2))

  unlink(list.files(dirname(path), pattern = basename(path), full.names = TRUE))
})


test_that("a saveRDS'd trained_RFlow is diagnosed, not silently broken", {
  # Documents the torch 0.17 behaviour that motivates save_flow(): nn_module
  # parameters are external pointers and do not survive saveRDS/readRDS.
  # sample_flow() must therefore still fail loudly on such an object rather
  # than return garbage -- save_flow()/load_flow() is the supported route.
  set.seed(303)
  x <- matrix(rnorm(200), 100, 2)
  fm <- flow_matching_trainer(
    x, epochs = 2, batch_size = 50, seed = 1, verbose = FALSE,
    vector_field = FlowVectorField(2, list(16), time_embed_dim = 4))

  f <- tempfile(fileext = ".rds")
  saveRDS(fm, f)
  fm_rds <- readRDS(f)
  # settings survive (that is the part this fix repaired) ...
  expect_identical(fm_rds$settings$device, "cpu")
  expect_identical(fm_rds$settings$data_dim, fm$settings$data_dim)
  # ... the network does not, and the failure is a loud error rather than
  # silently wrong numbers.
  res <- tryCatch(sample_flow(fm_rds, n_samples = 3, n_steps = 2),
                  error = function(e) structure(conditionMessage(e),
                                                class = "flow_rds_error"))
  expect_s3_class(res, "flow_rds_error")
  expect_match(as.character(res), "external pointer")

  rm(fm_rds); gc(verbose = FALSE)
  unlink(f)
})
