# Correspondence between this accountant and the Lean 4 formalization.
#
# Each helper below is a direct, deliberately naive transcription of a Lean
# definition -- written to read like the Lean source, not to be numerically
# careful. The tests check that the shipped implementation, which IS numerically
# careful, computes the same thing. This is what makes the proofs load-bearing
# for the R package rather than a parallel artefact.

# PrivatePGB.sgmMoment:
#   sum_{k in range (a+1)} choose(a,k) * (1-q)^(a-k) * q^k * exp(k(k-1)/(2v))
lean_sgmMoment <- function(q, v, a) {
  k <- 0:a
  sum(choose(a, k) * (1 - q)^(a - k) * q^k * exp(k * (k - 1) / (2 * v)))
}

# PrivatePGB.sgmRDP: log (sgmMoment q v a) / (a - 1)
lean_sgmRDP <- function(q, v, a) log(lean_sgmMoment(q, v, a)) / (a - 1)

# The epsilon of PrivatePGB.approxDP_transcript_cks, at one order:
#   T * tau - (log delta + log alpha) / (alpha - 1) + log ((alpha - 1) / alpha)
lean_cks_eps <- function(total_rdp, alpha, delta) {
  total_rdp - (log(delta) + log(alpha)) / (alpha - 1) + log((alpha - 1) / alpha)
}


test_that("log_moment_sgm_int is Lean's sgmMoment, with v = sigma^2", {
  grid <- expand.grid(q = c(1e-4, 0.001, 0.01, 0.1, 0.3, 0.5),
                      sigma = c(0.7, 1.0, 1.5, 4.0),
                      alpha = 2:12)
  for (i in seq_len(nrow(grid))) {
    q <- grid$q[i]; sigma <- grid$sigma[i]; a <- grid$alpha[i]
    direct <- lean_sgmMoment(q, sigma^2, a)
    # Only compare where the naive sum is itself trustworthy.
    if (!is.finite(direct) || direct <= 0) next
    # Relative difference of the moments, formed in log space so that the
    # comparison stays meaningful when the moment is astronomically large.
    # Comparing the logs relatively would not: log A passes through zero as
    # q -> 0, and there any absolute error looks infinitely relative.
    rel <- abs(expm1(log_moment_sgm_int(q, sigma, a) - log(direct)))
    expect_lt(rel, 1e-12)
  }
})


test_that("rdp_sgm at integer orders is Lean's sgmRDP", {
  grid <- expand.grid(q = c(1e-4, 0.001, 0.01, 0.1, 0.3),
                      sigma = c(0.8, 1.0, 2.0, 5.0),
                      alpha = 2:10)
  for (i in seq_len(nrow(grid))) {
    q <- grid$q[i]; sigma <- grid$sigma[i]; a <- grid$alpha[i]
    direct <- lean_sgmRDP(q, sigma^2, a)
    if (!is.finite(direct) || direct <= 0) next
    # Mixed absolute/relative: the RDP is ~1e-9 at the small-q end, where a
    # purely relative bound would be testing the naive sum's own round-off.
    expect_lt(abs(rdp_sgm(q, sigma, a) - direct), 1e-12 * max(1, abs(direct)))
  }
})


test_that("the q = 1 and q = 0 edges match sgmMoment_one and sgmMoment_zero", {
  # sgmMoment_one: sgmMoment 1 v a = exp(a(a-1)/(2v)), so the RDP is a/(2v).
  for (sigma in c(0.5, 1.0, 3.0)) for (a in c(2, 5, 17, 2.5)) {
    expect_equal(rdp_sgm(1, sigma, a), a / (2 * sigma^2))
  }
  # sgmMoment_zero: sgmMoment 0 v a = 1, so the RDP is 0.
  for (sigma in c(0.5, 1.0, 3.0)) for (a in c(2, 5, 17, 2.5)) {
    expect_equal(rdp_sgm(0, sigma, a), 0)
  }
})


test_that("get_epsilon minimises Lean's approxDP_transcript_cks expression", {
  for (steps in c(1, 50, 1000)) {
    acc <- dp_accountant_poisson$new(sampling_rate = 0.01,
                                     noise_multiplier = 1.1,
                                     target_delta = 1e-5)
    acc$step(steps)

    orders <- acc$orders
    direct <- vapply(orders, function(a) {
      lean_cks_eps(steps * rdp_sgm(0.01, 1.1, a), a, 1e-5)
    }, numeric(1))
    direct <- direct[is.finite(direct)]

    expect_equal(acc$get_epsilon(), max(0, min(direct)))
  }
})


test_that("the CKS conversion is never worse (cksEps_le_classicalEps)", {
  for (tau in c(1e-4, 0.01, 0.5, 3)) {
    for (alpha in c(1.5, 2, 8, 64, 256)) {
      for (delta in c(1e-3, 1e-5, 1e-10)) {
        classical <- tau + log(1 / delta) / (alpha - 1)
        expect_lte(lean_cks_eps(tau, alpha, delta), classical)
      }
    }
  }
})


test_that("fractional orders are bracketed by the certified integer orders", {
  # Monotonicity of the Renyi divergence in its order. This is the run-time
  # check inside rdp_sgm(); here it is exercised across the default grid, which
  # is where an under-converged series would otherwise corrupt the minimum.
  acc <- dp_accountant_poisson$new(sampling_rate = 0.01, noise_multiplier = 1.1)
  for (sigma in c(0.8, 1.1, 2.0)) {
    vals <- vapply(acc$orders, function(a) rdp_sgm(0.01, sigma, a), numeric(1))
    expect_true(all(is.finite(vals)))
    expect_false(is.unsorted(vals))
  }
})


test_that("a fractional value outside the certified bracket is refused", {
  q <- 0.01; sigma <- 1.1; alpha <- 4.5
  lo <- log_moment_sgm_int(q, sigma, 4) / 3
  hi <- log_moment_sgm_int(q, sigma, 5) / 4

  # The honest value passes.
  expect_silent(check_rdp_bracket(rdp_sgm(q, sigma, alpha), q, sigma, alpha))

  # An understated one is caught: this is the failure mode that matters,
  # since get_epsilon() minimises over orders and would adopt it.
  expect_error(check_rdp_bracket(0.5 * lo, q, sigma, alpha), "understated")
  expect_error(check_rdp_bracket(2 * hi, q, sigma, alpha), "above the certified")
})


test_that("certified_only keeps integer orders and never reports less", {
  full <- dp_accountant_poisson$new(sampling_rate = 0.01, noise_multiplier = 1.1)
  cert <- dp_accountant_poisson$new(sampling_rate = 0.01, noise_multiplier = 1.1,
                                    certified_only = TRUE)
  expect_true(all(abs(cert$orders - round(cert$orders)) < 1e-8))
  expect_lt(length(cert$orders), length(full$orders))

  for (steps in c(1, 100, 5000)) {
    full$steps <- steps
    cert$steps <- steps
    expect_gte(cert$get_epsilon(), full$get_epsilon())
  }
})
