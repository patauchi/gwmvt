## Robustness tests for GWPCA methods in the gwmvt package
## - Generates synthetic spatial data (coords + 5 variables)
## - Injects 0.1%, 1%, 5%, 10% outliers
## - Fits GWPCA with standard and all robust methods
## - Compares PC1/PC2 scores (robust vs standard) with 1:1 line
## - Computes simple empirical robustness metrics: breakdown proxy and influence proxy
##
## Notes
## - This is an example script intended for exploration. Some parts can be
##   computationally intensive; use the controls in the Config section.

set.seed(123)

## ---- Config ---------------------------------------------------------------
N <- 200              # number of observations (>= 1000 so 0.1% = 1 point)
P <- 5                 # variables
BW <- 0.2              # bandwidth in [0, 1] since we simulate coords in [0,1]^2
USE_COR <- FALSE       # use correlation matrix instead of covariance
K_KEEP <- 2            # components to analyze in scatter comparisons

# Outlier proportions to test
cont_levels <- c(`0.1%` = 0.001, `1%` = 0.01, `5%` = 0.05, `10%` = 0.10)

# Robust methods exposed by gwpca()
robust_methods <- c(
  "adaptive_huber", "adaptive_mcd", "spatial_trim", "mve",
  "s_estimator", "mm_estimator", "lof", "bacon", "spatial_depth", "robpca"
)

# Influence-function estimation controls (can be slow)
ENABLE_IF <- TRUE      # set to FALSE to skip IF estimation
N_IF_POINTS <- 25      # number of leave-one-out points per method

## ---- Helpers --------------------------------------------------------------
angle_between <- function(a, b, eps = 1e-12) {
  a <- as.numeric(a); b <- as.numeric(b)
  a <- a / max(sqrt(sum(a^2)), eps)
  b <- b / max(sqrt(sum(b^2)), eps)
  acos(pmin(pmax(sum(a * b), -1), 1)) * 180 / pi
}

standardize_cols <- function(X) {
  sx <- apply(X, 2, sd)
  sx[sx == 0] <- 1
  scale(X, center = TRUE, scale = sx)
}

generate_spatial_data <- function(n = N, p = P) {
  coords <- cbind(runif(n), runif(n))
  x <- coords[, 1]
  y <- coords[, 2]

  # Two latent spatial factors with smooth variation
  f1 <- sin(2 * pi * x) + 0.3 * cos(2 * pi * y)
  f2 <- cos(2 * pi * y) - 0.3 * sin(2 * pi * x)

  # Loading matrix (fixed) for 5 variables
  L <- matrix(c(
    0.8,  0.1,
    0.6, -0.2,
    0.5,  0.6,
   -0.2,  0.7,
   -0.5, -0.3
  ), nrow = p, byrow = TRUE)

  F <- cbind(f1, f2)
  signal <- F %*% t(L)             # n x p
  noise <- matrix(rnorm(n * p, sd = 0.3), n, p)
  X <- signal + noise

  list(data = X, coords = coords)
}

contaminate_data <- function(X, prop, amp = 10) {
  n <- nrow(X); p <- ncol(X)
  m <- max(1L, floor(n * prop))
  idx <- sample.int(n, m)
  Xc <- X
  sds <- apply(X, 2, sd)
  sds[sds == 0] <- 1

  # Create multivariate outliers: shift by amp * sd in random directions
  shifts <- matrix(rnorm(m * p), m, p)
  shifts <- t(t(shifts) * sds)      # scale by variable sd
  shifts <- amp * shifts
  Xc[idx, ] <- Xc[idx, ] + shifts
  attr(Xc, "outlier_idx") <- idx
  Xc
}

fit_gwpca <- function(X, coords, method = "standard", k = K_KEEP,
                       bw = BW, use_cor = USE_COR, verbose = FALSE) {
  gwmvt::gwpca(
    data = X,
    coords = coords,
    bandwidth = bw,
    method = method,
    use_correlation = use_cor,
    k = k,
    detect_outliers = FALSE,  # keep control; we contaminate explicitly
    verbose = verbose
  )
}

plot_scores_compare <- function(std_fit, rob_fit, comp = 1, main_extra = "") {
  oldpar <- par(no.readonly = TRUE); on.exit(par(oldpar))
  par(mar = c(4,4,3,1))
  x <- std_fit$scores[, comp]
  y <- rob_fit$scores[, comp]
  plot(x, y, pch = 16, col = rgb(0,0,0,0.4),
       xlab = paste0("Standard PC", comp),
       ylab = paste0("Robust PC", comp),
       main = paste0("Scores: PC", comp, " (", main_extra, ")"))
  abline(0, 1, col = "red", lwd = 2, lty = 2)
}

compare_method_metrics <- function(std_fit, rob_fit, coords) {
  # Correlation between scores for PC1/PC2
  k_use <- min(dim(std_fit$scores)[2], dim(rob_fit$scores)[2], K_KEEP)
  score_cor <- vapply(seq_len(k_use), function(j) {
    suppressWarnings(cor(std_fit$scores[, j], rob_fit$scores[, j], use = "pairwise"))
  }, numeric(1))

  # Mean absolute difference of variance explained
  med <- mean(abs(std_fit$var_explained[, 1:k_use, drop = FALSE] -
                  rob_fit$var_explained[, 1:k_use, drop = FALSE]), na.rm = TRUE)

  # Loading angle at a central location (nearest to median coords)
  med_xy <- apply(coords, 2, median)
  d <- sqrt((coords[,1] - med_xy[1])^2 + (coords[,2] - med_xy[2])^2)
  i0 <- which.min(d)
  ang <- vapply(seq_len(k_use), function(j) {
    angle_between(std_fit$loadings[i0, , j], rob_fit$loadings[i0, , j])
  }, numeric(1))

  list(score_cor = score_cor, var_expl_mad = med, angle_deg = ang)
}

empirical_breakdown_proxy <- function(clean_fit, cont_fit, angle_thr = 45, var_thr = 0.5) {
  # Compare contaminated-fit vs clean-fit (same method)
  k_use <- min(dim(clean_fit$scores)[2], dim(cont_fit$scores)[2], K_KEEP)
  # Angle at center
  coords <- clean_fit$coords
  med_xy <- apply(coords, 2, median)
  d <- sqrt((coords[,1] - med_xy[1])^2 + (coords[,2] - med_xy[2])^2)
  i0 <- which.min(d)
  ang <- vapply(seq_len(k_use), function(j) {
    angle_between(clean_fit$loadings[i0, , j], cont_fit$loadings[i0, , j])
  }, numeric(1))
  # Mean relative change in variance explained on PC1..k
  vr_clean <- colMeans(clean_fit$var_explained[, 1:k_use, drop = FALSE], na.rm = TRUE)
  vr_cont  <- colMeans(cont_fit$var_explained[, 1:k_use, drop = FALSE],  na.rm = TRUE)
  rel_diff <- mean(abs(vr_cont - vr_clean) / pmax(vr_clean, 1e-9))
  failed <- (any(ang > angle_thr) || rel_diff > var_thr)
  list(failed = failed, angle_deg = ang, rel_var_diff = rel_diff)
}

influence_proxy <- function(X, coords, method, bw = BW, k = K_KEEP,
                            n_points = N_IF_POINTS) {
  # Approximates IF by leave-one-out change in mean variance explained (PC1)
  fit_all <- fit_gwpca(X, coords, method = method, k = k, bw = bw)
  n <- nrow(X)
  idx <- sample.int(n, min(n_points, n))
  base <- mean(fit_all$var_explained[, 1], na.rm = TRUE)

  diffs <- numeric(length(idx))
  for (i in seq_along(idx)) {
    j <- idx[i]
    X_loo <- X[-j, , drop = FALSE]
    coords_loo <- coords[-j, , drop = FALSE]
    fit_loo <- fit_gwpca(X_loo, coords_loo, method = method, k = k, bw = bw)
    diffs[i] <- abs(mean(fit_loo$var_explained[, 1], na.rm = TRUE) - base)
  }
  list(mean = mean(diffs), median = median(diffs), max = max(diffs), diffs = diffs)
}

## ---- Data -----------------------------------------------------------------
cat("Generating synthetic spatial dataset...\n")
gen <- generate_spatial_data(N, P)
X0 <- gen$data
coords <- gen$coords

if (USE_COR) {
  X0 <- standardize_cols(X0)
}

## Baseline fits on clean data (standard and robust methods)
cat("Fitting baseline (clean) models...\n")
base_standard <- fit_gwpca(X0, coords, method = "standard")
base_robust <- lapply(robust_methods, function(m)
  fit_gwpca(X0, coords, method = m))
names(base_robust) <- robust_methods

## ---- Contamination loops ---------------------------------------------------
results <- list()

for (lab in names(cont_levels)) {
  lab <- 1
  prop <- cont_levels[[lab]]
  cat(sprintf("\n=== Contamination: %s (%.3f) ===\n", lab, prop))
  Xc <- contaminate_data(X0, prop)
  out_idx <- attr(Xc, "outlier_idx")

  # Standard and robust fits on contaminated data
  fit_std_c <- fit_gwpca(Xc, coords, method = "standard")
  fit_rob_c <- lapply(robust_methods, function(m)
    fit_gwpca(Xc, coords, method = m))
  names(fit_rob_c) <- robust_methods

  # Comparisons robust vs standard (same contaminated data)
  comp_metrics <- lapply(robust_methods, function(m) {
    compare_method_metrics(fit_std_c, fit_rob_c[[m]], coords)
  })
  names(comp_metrics) <- robust_methods

  # Empirical breakdown proxy: compare same method clean vs contaminated
  breakdown <- lapply(robust_methods, function(m) {
    empirical_breakdown_proxy(base_robust[[m]], fit_rob_c[[m]])
  })
  names(breakdown) <- robust_methods

  results[[lab]] <- list(
    prop = prop,
    outlier_idx = out_idx,
    fit_standard = fit_std_c,
    fit_robust = fit_rob_c,
    compare_metrics = comp_metrics,
    breakdown_proxy = breakdown
  )

  ## Plots: PC1/PC2 robust vs standard
  opar <- par(no.readonly = TRUE); on.exit(par(opar), add = TRUE)
  for (m in robust_methods) {
    # 2-panel plot for PC1 and PC2
    grDevices::dev.new(noRStudioGD = TRUE)
    par(mfrow = c(1, 2))
    plot_scores_compare(fit_std_c, fit_rob_c[[m]], comp = 1,
                        main_extra = sprintf("%s, %s", m, lab))
    plot_scores_compare(fit_std_c, fit_rob_c[[m]], comp = 2,
                        main_extra = sprintf("%s, %s", m, lab))
  }
}

## ---- Influence proxies (optional) -----------------------------------------
if (ENABLE_IF) {
  cat("\nEstimating influence proxies (leave-one-out deltas)...\n")
  inf_res <- lapply(robust_methods, function(m) {
    influence_proxy(X0, coords, method = m)
  })
  names(inf_res) <- robust_methods
} else {
  inf_res <- NULL
}

## ---- Reporting -------------------------------------------------------------
cat("\nSummary metrics (by contamination level and method):\n")
for (lab in names(results)) {
  cat(sprintf("\n-- %s --\n", lab))
  comp <- results[[lab]]$compare_metrics
  for (m in names(comp)) {
    s <- comp[[m]]
    cat(sprintf("%15s | cor(PC1,PC2)=(%.3f, %.3f), angle_deg=(%.1f, %.1f), MAD(var)~=%.3f\n",
                m, s$score_cor[1], s$score_cor[2], s$angle_deg[1], s$angle_deg[2], s$var_expl_mad))
  }
}

cat("\nBreakdown proxy flags (angle>45° OR rel var change>0.5):\n")
for (lab in names(results)) {
  cat(sprintf("\n-- %s --\n", lab))
  br <- results[[lab]]$breakdown_proxy
  for (m in names(br)) {
    cat(sprintf("%15s | failed=%s, rel_var_diff=%.3f, angle1=%.1f°\n",
                m, ifelse(br[[m]]$failed, "YES", "no"), br[[m]]$rel_var_diff, br[[m]]$angle_deg[1]))
  }
}

if (!is.null(inf_res)) {
  cat("\nInfluence proxy (leave-one-out mean delta of mean VarExpl[PC1]):\n")
  for (m in names(inf_res)) {
    cat(sprintf("%15s | mean=%.4f, median=%.4f, max=%.4f\n",
                m, inf_res[[m]]$mean, inf_res[[m]]$median, inf_res[[m]]$max))
  }
}

cat("\nDone.\n")

