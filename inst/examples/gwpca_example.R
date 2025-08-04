# Example usage of GWPCA functions from gwmvt package
# This script demonstrates various features of the geographically weighted PCA

# Load the package
library(gwmvt)

# Set random seed for reproducibility
set.seed(123)

# Example 1: Basic GWPCA with simulated data
# ==========================================

# Generate spatial coordinates
n <- 500
coords <- matrix(runif(n * 2, 0, 10), ncol = 2)
colnames(coords) <- c("X", "Y")

# Generate spatially varying data
# Create 5 variables with different spatial patterns
p <- 5
data <- matrix(0, n, p)

# Variable 1: North-South gradient
data[, 1] <- coords[, 2] + rnorm(n, 0, 0.5)

# Variable 2: East-West gradient
data[, 2] <- coords[, 1] + rnorm(n, 0, 0.5)

# Variable 3: Distance from center
center <- c(5, 5)
data[, 3] <- sqrt((coords[, 1] - center[1])^2 + (coords[, 2] - center[2])^2) +
             rnorm(n, 0, 0.5)

# Variable 4: Quadrant-based pattern
data[, 4] <- ifelse(coords[, 1] > 5 & coords[, 2] > 5, 1,
             ifelse(coords[, 1] < 5 & coords[, 2] < 5, -1, 0)) +
             rnorm(n, 0, 0.3)

# Variable 5: Random noise
data[, 5] <- rnorm(n, 0, 1)

# Add some outliers
outlier_idx <- sample(1:n, 20)
data[outlier_idx, ] <- data[outlier_idx, ] + matrix(rnorm(20 * p, 0, 3), 20, p)

# Example 1.1: Standard GWPCA
# ---------------------------
cat("Running standard GWPCA...\n")
result_standard <- gwpca(data, coords,
                        bandwidth = 2.0,
                        method = "standard",
                        k = 3,
                        verbose = TRUE)

print(result_standard)
summary(result_standard)

# Example 1.2: Robust GWPCA with adaptive Huber
# ---------------------------------------------
cat("\nRunning robust GWPCA with adaptive Huber method...\n")
result_robust <- gwpca(data, coords,
                      bandwidth = 2.0,
                      method = "adaptive_huber",
                      k = 3,
                      detect_outliers = TRUE,
                      outlier_threshold = 2.5,
                      verbose = TRUE)

print(result_robust)

# Compare number of outliers detected
cat("\nOutliers detected:", sum(result_robust$spatial_outliers), "\n")
cat("True outliers:", length(outlier_idx), "\n")
cat("Correctly identified:",
    sum(result_robust$spatial_outliers[outlier_idx]),
    "out of", length(outlier_idx), "\n")

# Example 1.3: Using correlation matrix instead of covariance
# ----------------------------------------------------------
cat("\nRunning GWPCA with correlation matrix...\n")
result_corr <- gwpca(data, coords,
                    bandwidth = 2.0,
                    method = "adaptive_huber",
                    use_correlation = TRUE,
                    k = 3,
                    verbose = FALSE)

# Example 2: Bandwidth Selection
# ==============================
cat("\nTesting different bandwidths...\n")

# Define bandwidth candidates
bandwidths <- seq(0.5, 5.0, by = 0.5)

# Cross-validation for bandwidth selection
cv_result <- gwpca_bandwidth_cv(data, coords,
                               bandwidths = bandwidths,
                               method = "adaptive_huber",
                               criterion = "CV",
                               k = 3,
                               verbose = TRUE)

cat("\nOptimal bandwidth:", cv_result$optimal, "\n")

# Example 3: Visualization
# ========================
if (interactive()) {
  # Set up plotting area
  par(mfrow = c(2, 2))

  # Plot 1: Variance explained by PC1
  plot(result_robust, type = "variance", component = 1,
       main = "Variance Explained by PC1")

  # Plot 2: Component scores for PC1
  plot(result_robust, type = "scores", component = 1,
       main = "PC1 Scores")

  # Plot 3: Eigenvalue distribution
  plot(result_robust, type = "eigenvalues",
       main = "Eigenvalue Distribution")

  # Plot 4: Outlier locations
  plot(coords[, 1], coords[, 2],
       col = ifelse(result_robust$spatial_outliers == 1, "red", "black"),
       pch = ifelse(result_robust$spatial_outliers == 1, 16, 1),
       main = "Spatial Outliers",
       xlab = "X", ylab = "Y")
  legend("topright", c("Normal", "Outlier"),
         col = c("black", "red"), pch = c(1, 16))

  par(mfrow = c(1, 1))
}

# Example 4: Local Analysis
# =========================
cat("\nAnalyzing specific locations...\n")

# Select a few points of interest
poi_coords <- matrix(c(2.5, 2.5,  # Lower left
                      7.5, 7.5,  # Upper right
                      5.0, 5.0), # Center
                    ncol = 2, byrow = TRUE)

# Get local variance explained
local_var <- get_local_variance(result_robust, cumulative = TRUE)

# Find nearest points to POIs
poi_indices <- numeric(nrow(poi_coords))
for (i in 1:nrow(poi_coords)) {
  distances <- sqrt((coords[, 1] - poi_coords[i, 1])^2 +
                   (coords[, 2] - poi_coords[i, 2])^2)
  poi_indices[i] <- which.min(distances)
}

# Display local results
cat("\nLocal variance explained (cumulative):\n")
for (i in 1:length(poi_indices)) {
  idx <- poi_indices[i]
  cat(sprintf("Location (%.1f, %.1f): PC1=%.3f, PC2=%.3f, PC3=%.3f\n",
              coords[idx, 1], coords[idx, 2],
              local_var[idx, 1], local_var[idx, 2], local_var[idx, 3]))
}

# Get local loadings for center point
center_loadings <- get_local_loadings(result_robust,
                                     location = c(5.0, 5.0),
                                     n_components = 3)
cat("\nLoadings at center location:\n")
print(round(center_loadings, 3))

# Example 5: Prediction on New Data
# =================================
cat("\nPredicting scores for new locations...\n")

# Generate new data
n_new <- 50
new_coords <- matrix(runif(n_new * 2, 0, 10), ncol = 2)
new_data <- matrix(0, n_new, p)

# Similar patterns as training data
new_data[, 1] <- new_coords[, 2] + rnorm(n_new, 0, 0.5)
new_data[, 2] <- new_coords[, 1] + rnorm(n_new, 0, 0.5)
new_data[, 3] <- sqrt((new_coords[, 1] - 5)^2 + (new_coords[, 2] - 5)^2) +
                 rnorm(n_new, 0, 0.5)
new_data[, 4] <- ifelse(new_coords[, 1] > 5 & new_coords[, 2] > 5, 1,
                 ifelse(new_coords[, 1] < 5 & new_coords[, 2] < 5, -1, 0)) +
                 rnorm(n_new, 0, 0.3)
new_data[, 5] <- rnorm(n_new, 0, 1)

# Predict scores
new_scores <- predict(result_robust, new_data, new_coords)
cat("Predicted scores for first 5 new observations:\n")
print(round(new_scores[1:5, ], 3))

# Example 6: Different Robust Methods
# ===================================
cat("\nComparing different robust methods...\n")

methods <- c("adaptive_huber", "adaptive_mcd", "spatial_trim")
method_results <- list()

for (method in methods) {
  cat(sprintf("Running %s...\n", method))
  method_results[[method]] <- gwpca(data, coords,
                                   bandwidth = 2.0,
                                   method = method,
                                   k = 3,
                                   detect_outliers = TRUE,
                                   verbose = FALSE)
}

# Compare results
cat("\nMean variance explained by PC1:\n")
for (method in methods) {
  mean_var <- mean(method_results[[method]]$var_explained[, 1], na.rm = TRUE)
  cat(sprintf("%s: %.3f\n", method, mean_var))
}

# Example 7: Parallel Processing Performance
# ==========================================
if (get_max_threads() > 1) {
  cat("\nTesting parallel processing performance...\n")

  # Sequential
  time_seq <- system.time({
    result_seq <- gwpca(data, coords,
                       bandwidth = 2.0,
                       method = "adaptive_huber",
                       parallel = FALSE,
                       verbose = FALSE)
  })

  # Parallel
  time_par <- system.time({
    result_par <- gwpca(data, coords,
                       bandwidth = 2.0,
                       method = "adaptive_huber",
                       parallel = TRUE,
                       verbose = FALSE)
  })

  cat(sprintf("Sequential time: %.2f seconds\n", time_seq[3]))
  cat(sprintf("Parallel time: %.2f seconds\n", time_par[3]))
  cat(sprintf("Speedup: %.2fx\n", time_seq[3] / time_par[3]))
}

# Example 8: Adaptive Bandwidth
# =============================
cat("\nUsing adaptive bandwidth based on k-nearest neighbors...\n")

# Calculate adaptive bandwidths
k_neighbors <- 50
adaptive_bw <- adaptive_bandwidth_nn(coords, k_neighbors)

cat(sprintf("Adaptive bandwidth range: [%.2f, %.2f]\n",
            min(adaptive_bw), max(adaptive_bw)))

# Note: Full adaptive GWPCA would require modifying the main function
# This is shown here for demonstration of the adaptive bandwidth calculation

# Example 9: Spatial Autocorrelation of Components
# ===============================================
cat("\nCalculating spatial autocorrelation of principal components...\n")

# Calculate Moran's I for each component
for (i in 1:3) {
  moran_i <- morans_i(result_robust$scores[, i], coords, bandwidth = 2.0)
  cat(sprintf("Moran's I for PC%d: %.3f\n", i, moran_i))
}

# Local Moran's I for PC1
local_moran <- local_morans_i(result_robust$scores[, 1], coords, bandwidth = 2.0)

if (interactive()) {
  # Plot local Moran's I
  plot(coords[, 1], coords[, 2],
       col = heat.colors(100)[cut(local_moran, 100)],
       pch = 16,
       main = "Local Moran's I for PC1",
       xlab = "X", ylab = "Y")
}

cat("\nExample completed successfully!\n")
