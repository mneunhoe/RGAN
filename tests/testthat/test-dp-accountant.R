# Correctness of the RDP accountant for the Sampled Gaussian Mechanism.
#
# The load-bearing test is the quadrature cross-check: it evaluates the same
# definition by a different route, so it checks the algebra rather than the code
# against itself. The property tests below are necessary but would NOT have
# caught the 0.2.0 defect on their own - that formula satisfied all of them.

# log E_{z ~ N(0, sigma^2)}[((1-q) + q exp((2z-1)/(2 sigma^2)))^alpha] / (alpha-1),
# by direct numerical integration of the density ratio.
rdp_sgm_quadrature <- function(q, sigma, alpha, lo = -40, hi = 40, n = 400001) {
  z <- seq(lo, hi, length.out = n)
  h <- z[2] - z[1]
  log_phi <- -z^2 / (2 * sigma^2) - log(sigma * sqrt(2 * pi))
  ratio <- (1 - q) + q * exp((2 * z - 1) / (2 * sigma^2))
  log_sum_exp(log_phi + alpha * log(ratio) + log(h)) / (alpha - 1)
}

# The formula shipped in 0.2.0, for the regression test below.
legacy_rdp <- function(q, sigma, alpha) {
  base <- alpha / (2 * sigma^2)
  if (q < 0.01) return(q^2 * base)
  max(0, log((1 - q) + q * exp((alpha - 1) * base)) / (alpha - 1))
}


test_that("rdp_sgm matches direct numerical integration", {
  # The tolerance is set by the quadrature grid, not by the closed form: the
  # two agree to ~1e-12 where the integrand is well resolved. Any algebraic
  # error is orders of magnitude larger than this, not a few ulp.
  grid <- expand.grid(q = c(0.001, 0.01, 0.05, 0.2), sigma = c(0.8, 1.0, 1.5),
                      alpha = c(2, 3, 5, 8))
  for (i in seq_len(nrow(grid))) {
    g <- grid[i, ]
    closed <- rdp_sgm(g$q, g$sigma, g$alpha)
    quad <- rdp_sgm_quadrature(g$q, g$sigma, g$alpha)
    expect_lt(abs(closed - quad) / abs(quad), 1e-5,
              label = sprintf("rel. error at q=%g sigma=%g alpha=%g",
                              g$q, g$sigma, g$alpha))
  }
})


test_that("boundary sampling rates are exact", {
  # Nothing sampled: no privacy loss.
  expect_equal(rdp_sgm(0, 1.0, 8), 0)
  # Everything sampled: no amplification, so the plain Gaussian bound.
  for (alpha in c(2, 8, 32)) {
    expect_equal(rdp_sgm(1, 1.1, alpha), alpha / (2 * 1.1^2))
  }
})


test_that("subsampling never increases the RDP", {
  for (q in c(1e-4, 1e-3, 0.005, 0.01, 0.05, 0.1, 0.3)) {
    for (sigma in c(0.6, 0.8, 1.0, 1.5, 3.0)) {
      for (alpha in c(2, 4, 8, 16, 32, 64)) {
        expect_lte(rdp_sgm(q, sigma, alpha), alpha / (2 * sigma^2) + 1e-12)
      }
    }
  }
})


test_that("the bound is continuous in the sampling rate", {
  # 0.2.0 switched formula at q = 0.01 and jumped by four orders of magnitude.
  sigma <- 1.1; alpha <- 32
  qs <- c(0.00999, 0.009999, 0.01, 0.010001, 0.01001)
  vals <- vapply(qs, function(q) rdp_sgm(q, sigma, alpha), numeric(1))
  expect_lt(max(abs(diff(vals))) / vals[1], 0.01)

  legacy <- vapply(qs, function(q) legacy_rdp(q, sigma, alpha), numeric(1))
  expect_gt(max(abs(diff(legacy))) / legacy[1], 100)
})


test_that("epsilon is monotone in the parameters", {
  mk <- function(q, sigma, steps) {
    a <- dp_accountant_poisson$new(sampling_rate = q, noise_multiplier = sigma,
                                   target_delta = 1e-5)
    a$step(steps)
    a$get_epsilon()
  }
  expect_lt(mk(0.01, 1.0, 100), mk(0.01, 1.0, 1000))   # more steps
  expect_lt(mk(0.005, 1.0, 500), mk(0.02, 1.0, 500))   # higher sampling rate
  expect_gt(mk(0.01, 0.8, 500), mk(0.01, 2.0, 500))    # less noise
  expect_equal(mk(0.01, 1.0, 0), 0)                    # nothing spent
})


test_that("0.2.0 severely understated epsilon at small sampling rates", {
  # The regime DP-SGD actually runs in: q = batch / n well below 0.01.
  q <- 0.005; sigma <- 1.0; alpha <- 16
  expect_gt(rdp_sgm(q, sigma, alpha) / legacy_rdp(q, sigma, alpha), 100)
})


test_that("noise calibration and step budgeting agree with the accountant", {
  target_eps <- 2.0; target_delta <- 1e-5; q <- 0.01; steps <- 1000

  sigma <- calibrate_noise_multiplier(target_eps, target_delta, q, steps)
  a <- dp_accountant_poisson$new(sampling_rate = q, noise_multiplier = sigma,
                                 target_delta = target_delta)
  a$step(steps)
  # Calibration must land inside the budget, never just outside it.
  expect_lte(a$get_epsilon(), target_eps)

  max_steps <- compute_max_steps(target_eps, target_delta, q, sigma)
  b <- dp_accountant_poisson$new(sampling_rate = q, noise_multiplier = sigma,
                                 target_delta = target_delta)
  b$step(max_steps)
  expect_lte(b$get_epsilon(), target_eps)

  d <- dp_accountant_poisson$new(sampling_rate = q, noise_multiplier = sigma,
                                 target_delta = target_delta)
  d$step(max_steps + 1L)
  expect_gt(d$get_epsilon(), target_eps)
})


test_that("rdp_sgm validates its arguments", {
  expect_error(rdp_sgm(0.01, 0, 4), "sigma must be")
  expect_error(rdp_sgm(-0.1, 1, 4), "q must be")
  expect_error(rdp_sgm(0.01, 1, 1), "greater than 1")
})


# --- fractional orders ------------------------------------------------------
#
# Restricting the order grid to integers inflates epsilon by up to 11% when the
# optimal order is small, because consecutive integers are then far apart in
# relative terms. The series below covers non-integer orders.

test_that("the fractional series reproduces the exact closed form at integer orders", {
  # This is the discriminating check: at integer orders the binomial sum is
  # exact, so any error in the series shows up directly. (Quadrature is too
  # coarse to settle it - for q = 0.001 it is itself off by ~1e-6.)
  for (q in c(0.001, 0.01, 0.05)) {
    for (sigma in c(0.8, 1.2)) {
      for (alpha in c(2L, 3L, 4L, 6L)) {
        expect_lt(
          abs(log_moment_sgm_frac(q, sigma, alpha) -
                log_moment_sgm_int(q, sigma, alpha)) /
            abs(log_moment_sgm_int(q, sigma, alpha)),
          1e-9,
          label = sprintf("series vs closed form at q=%g sigma=%g alpha=%d",
                          q, sigma, alpha)
        )
      }
    }
  }
})


test_that("the fractional series matches numerical integration", {
  for (q in c(0.01, 0.05, 0.2)) {
    for (sigma in c(0.5, 0.8, 1.2)) {
      for (alpha in c(1.25, 1.5, 2.5, 3.75, 6.5)) {
        series <- log_moment_sgm_frac(q, sigma, alpha)
        quad <- (alpha - 1) * rdp_sgm_quadrature(q, sigma, alpha)
        expect_lt(abs(series - quad) / abs(quad), 1e-5,
                  label = sprintf("q=%g sigma=%g alpha=%g", q, sigma, alpha))
      }
    }
  }
})


test_that("fractional orders are accepted and behave sensibly", {
  # Renyi divergence is non-decreasing in the order.
  vals <- vapply(c(1.2, 1.5, 2, 2.5, 3, 4.5, 8), function(a) rdp_sgm(0.01, 1.0, a),
                 numeric(1))
  expect_true(all(diff(vals) > 0))
  # Subsampling never increases the RDP, at fractional orders too.
  for (alpha in c(1.3, 2.7, 5.5, 9.1)) {
    expect_lte(rdp_sgm(0.02, 0.9, alpha), alpha / (2 * 0.9^2) + 1e-12)
  }
})


test_that("the default order grid is within 1% of the optimum", {
  eps_cks <- function(r, a, d) r - (log(d) + log(a)) / (a - 1) + log((a - 1) / a)
  eps_grid <- function(q, s, steps, d, orders) {
    min(vapply(orders, function(a) eps_cks(steps * rdp_sgm(q, s, a), a, d), numeric(1)))
  }
  fine <- unique(sort(c(seq(1.02, 12, by = 0.02), 13:256)))

  # The three settings where an integer-only grid was worst: the optimum sits
  # at alpha 2-3, where the integer grid is coarsest.
  for (cfg in list(c(0.02, 0.6, 500, 1e-5), c(0.002, 0.5, 100, 1e-4),
                   c(0.01, 0.6, 100, 1e-5))) {
    q <- cfg[1]; s <- cfg[2]; steps <- cfg[3]; d <- cfg[4]
    a <- dp_accountant_poisson$new(sampling_rate = q, noise_multiplier = s,
                                   target_delta = d)
    a$step(steps)
    expect_lt(a$get_epsilon() / eps_grid(q, s, steps, d, fine) - 1, 0.01,
              label = sprintf("default-grid gap at q=%g sigma=%g steps=%g", q, s, steps))
    # ... and strictly better than integer-only would have been.
    expect_lt(a$get_epsilon(), eps_grid(q, s, steps, d, 2:256))
  }
})
