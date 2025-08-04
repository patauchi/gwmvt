#' Geographically Weighted Principal Component Analysis
#'
#' Performs Geographically Weighted PCA with various robust methods
#'
#' @param data Numeric matrix (n x p) of data values
#' @param coords Numeric matrix (n x 2) of coordinates (e.g., longitude, latitude)
#' @param bandwidth Positive numeric value for spatial bandwidth
#' @param method Character string specifying the robust method. Options include:
#'   \itemize{
#'     \item "standard" - Standard GWPCA
#'     \item "adaptive_huber" - Adaptive Huber M-estimator (default)
#'     \item "adaptive_mcd" - Adaptive Minimum Covariance Determinant
#'     \item "spatial_trim" - Spatial trimming method
#'   }
#' @param use_correlation Logical, whether to use correlation matrix instead of covariance
#' @param k Integer, number of components to retain (default: all)
#' @param detect_outliers Logical, whether to detect spatial outliers first
#' @param outlier_threshold Numeric, threshold for outlier detection (default: 2.5)
#' @param trim_prop Numeric, trimming proportion for spatial_trim method (default: 0.1)
#' @param h_fraction Numeric, fraction parameter for adaptive methods (default: 0.75)
#' @param lof_k Integer, number of neighbors for LOF outlier detection (default: 10)
#' @param bacon_alpha Numeric, significance level for BACON method (default: 0.05)
#' @param depth_type Character, type of depth measure ("mahalanobis", "projection", "spatial")
#' @param robpca_k_max Integer, maximum number of components for ROBPCA (default: 10)
#' @param parallel Logical, whether to use parallel processing
#' @param n_threads Integer, number of threads to use (0 = auto)
#' @param verbose Logical, whether to show progress
#'
#' @return A list of class 'gwpca' containing:
#'   \item{eigenvalues}{Matrix (n x p) of eigenvalues at each location}
#'   \item{loadings}{3D array (n x p x k) of component loadings}
#'   \item{scores}{Matrix (n x k) of component scores}
#'   \item{var_explained}{Matrix (n x k) of variance explained proportions}
#'   \item{coords}{Input coordinate matrix}
#'   \item{bandwidth}{Used bandwidth value}
#'   \item{method}{Method used}
#'   \item{use_correlation}{Whether correlation was used}
#'   \item{n_components}{Number of components retained}
#'   \item{spatial_outliers}{Vector of outlier indicators (if detect_outliers = TRUE)}
#'
#' @examples
#' \dontrun{
#' # Generate example data
#' n <- 500
#' p <- 5
#' coords <- matrix(runif(n * 2), ncol = 2)
#' data <- matrix(rnorm(n * p), ncol = p)
#'
#' # Standard GWPCA
#' result <- gwpca(data, coords, bandwidth = 0.3)
#'
#' # Robust GWPCA with outlier detection
#' result_robust <- gwpca(data, coords, bandwidth = 0.3,
#'                       method = "adaptive_huber",
#'                       detect_outliers = TRUE)
#'
#' # Print summary
#' print(result_robust)
#' summary(result_robust)
#' }
#'
#' @export
gwpca <- function(data,
                  coords,
                  bandwidth,
                  method = c("adaptive_huber", "adaptive_mcd", "spatial_trim", "standard"),
                  use_correlation = FALSE,
                  k = NULL,
                  detect_outliers = TRUE,
                  outlier_threshold = 2.5,
                  trim_prop = 0.1,
                  h_fraction = 0.75,
                  lof_k = 10,
                  bacon_alpha = 0.05,
                  depth_type = "mahalanobis",
                  robpca_k_max = 10,
                  parallel = TRUE,
                  n_threads = 0,
                  verbose = FALSE) {

  # Input validation
  if (!is.matrix(data) && !is.data.frame(data)) {
    stop("data must be a matrix or data frame")
  }

  if (!is.matrix(coords) && !is.data.frame(coords)) {
    stop("coords must be a matrix or data frame")
  }

  # Convert to matrices
  data <- as.matrix(data)
  coords <- as.matrix(coords)

  # Check dimensions
  if (nrow(data) != nrow(coords)) {
    stop("Number of rows in data must match number of rows in coords")
  }

  if (ncol(coords) != 2) {
    stop("coords must have exactly 2 columns")
  }

  if (bandwidth <= 0) {
    stop("bandwidth must be positive")
  }

  # Match method
  method <- match.arg(method)

  # Set k if not specified
  if (is.null(k)) {
    k <- ncol(data)
  } else {
    k <- min(k, ncol(data))
  }

  # Call C++ function
  result <- gwpca_cpp(
    data = data,
    coords = coords,
    bandwidth = bandwidth,
    method = method,
    use_correlation = use_correlation,
    k = as.integer(k),
    detect_outliers = detect_outliers,
    outlier_threshold = outlier_threshold,
    trim_prop = trim_prop,
    h_fraction = h_fraction,
    lof_k = as.integer(lof_k),
    bacon_alpha = bacon_alpha,
    depth_type = depth_type,
    robpca_k_max = as.integer(robpca_k_max),
    parallel = parallel,
    n_threads = as.integer(n_threads),
    verbose = verbose
  )

  # Add class
  class(result) <- "gwpca"

  return(result)
}

#' Print method for gwpca objects
#'
#' @param x A gwpca object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.gwpca <- function(x, ...) {
  cat("Geographically Weighted PCA Results\n")
  cat("===================================\n")
  cat("Method:", x$method, "\n")
  cat("Number of locations:", nrow(x$coords), "\n")
  cat("Number of variables:", ncol(x$eigenvalues), "\n")
  cat("Number of components retained:", x$n_components, "\n")
  cat("Bandwidth:", x$bandwidth, "\n")
  cat("Correlation-based:", x$use_correlation, "\n")

  if (!is.null(x$spatial_outliers)) {
    n_outliers <- sum(x$spatial_outliers)
    cat("Spatial outliers detected:", n_outliers,
        sprintf("(%.1f%%)", 100 * n_outliers / length(x$spatial_outliers)), "\n")
  }

  # Summary of variance explained
  n_components_to_show <- min(x$n_components, 10)
  var_exp_summary <- matrix(NA, n_components_to_show, 3)
  colnames(var_exp_summary) <- c("Min", "Mean", "Max")
  rownames(var_exp_summary) <- paste0("PC", 1:n_components_to_show)

  for (i in 1:n_components_to_show) {
    var_exp_summary[i, ] <- c(
      min(x$var_explained[, i], na.rm = TRUE),
      mean(x$var_explained[, i], na.rm = TRUE),
      max(x$var_explained[, i], na.rm = TRUE)
    )
  }

  cat("\nVariance explained (proportion):\n")
  print(round(var_exp_summary, 3))

  if (x$n_components > n_components_to_show) {
    cat(sprintf("\n(Showing first %d of %d components)\n",
                n_components_to_show, x$n_components))
  }

  invisible(x)
}

#' Summary method for gwpca objects
#'
#' @param object A gwpca object
#' @param ... Additional arguments (ignored)
#'
#' @return A summary.gwpca object
#' @export
summary.gwpca <- function(object, ...) {
  result <- list(
    method = object$method,
    n_locations = nrow(object$coords),
    n_variables = dim(object$loadings)[2],
    n_components = object$n_components,
    bandwidth = object$bandwidth,
    use_correlation = object$use_correlation,
    var_explained_mean = colMeans(object$var_explained, na.rm = TRUE),
    var_explained_sd = apply(object$var_explained, 2, sd, na.rm = TRUE),
    eigenvalue_range = apply(object$eigenvalues, 2, range, na.rm = TRUE)
  )

  if (!is.null(object$spatial_outliers)) {
    result$n_outliers <- sum(object$spatial_outliers)
    result$outlier_percentage <- 100 * result$n_outliers / length(object$spatial_outliers)
  }

  class(result) <- "summary.gwpca"
  return(result)
}

#' Print method for summary.gwpca objects
#'
#' @param x A summary.gwpca object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.summary.gwpca <- function(x, ...) {
  cat("Summary of Geographically Weighted PCA\n")
  cat("=====================================\n")
  cat("Method:", x$method, "\n")
  cat("Number of locations:", x$n_locations, "\n")
  cat("Number of variables:", x$n_variables, "\n")
  cat("Number of components:", x$n_components, "\n")
  cat("Bandwidth:", x$bandwidth, "\n")
  cat("Correlation-based:", x$use_correlation, "\n")

  if (!is.null(x$n_outliers)) {
    cat("Spatial outliers:", x$n_outliers,
        sprintf("(%.1f%%)", x$outlier_percentage), "\n")
  }

  cat("\nMean variance explained by component:\n")
  var_exp_df <- data.frame(
    Component = paste0("PC", 1:length(x$var_explained_mean)),
    Mean = round(x$var_explained_mean, 4),
    SD = round(x$var_explained_sd, 4)
  )
  print(var_exp_df, row.names = FALSE)

  cat("\nEigenvalue ranges:\n")
  eigenval_df <- data.frame(
    Variable = paste0("Var", 1:ncol(x$eigenvalue_range)),
    Min = round(x$eigenvalue_range[1, ], 4),
    Max = round(x$eigenvalue_range[2, ], 4)
  )
  print(eigenval_df, row.names = FALSE)

  invisible(x)
}

#' Plot method for gwpca objects
#'
#' @param x A gwpca object
#' @param type Character string specifying plot type:
#'   "variance" (default), "loadings", "scores", or "eigenvalues"
#' @param component Integer, which component to plot (default: 1)
#' @param ... Additional arguments passed to plot functions
#'
#' @export
plot.gwpca <- function(x, type = c("variance", "loadings", "scores", "eigenvalues"),
                       component = 1, ...) {
  type <- match.arg(type)

  if (component > x$n_components) {
    stop("component exceeds number of retained components")
  }

  switch(type,
    variance = {
      plot(x$coords[, 1], x$coords[, 2],
           col = heat.colors(100)[cut(x$var_explained[, component], 100)],
           pch = 16,
           xlab = "X Coordinate",
           ylab = "Y Coordinate",
           main = paste("Variance Explained - PC", component),
           ...)

      # Add legend
      var_range <- range(x$var_explained[, component], na.rm = TRUE)
      legend("topright",
             legend = round(seq(var_range[1], var_range[2], length.out = 5), 3),
             col = heat.colors(5),
             pch = 16,
             title = "Var. Explained")
    },

    loadings = {
      # Plot first two loadings for selected component
      plot(x$loadings[, 1, component], x$loadings[, 2, component],
           col = rgb(x$coords[, 1], x$coords[, 2], 0.5),
           pch = 16,
           xlab = "Loading 1",
           ylab = "Loading 2",
           main = paste("Loadings - PC", component),
           ...)
    },

    scores = {
      plot(x$coords[, 1], x$coords[, 2],
           col = heat.colors(100)[cut(x$scores[, component], 100)],
           pch = 16,
           xlab = "X Coordinate",
           ylab = "Y Coordinate",
           main = paste("Component Scores - PC", component),
           ...)
    },

    eigenvalues = {
      boxplot(x$eigenvalues,
              xlab = "Variable",
              ylab = "Eigenvalue",
              main = "Distribution of Eigenvalues",
              ...)
    }
  )

  invisible(x)
}

#' Predict method for gwpca objects
#'
#' @param object A gwpca object
#' @param newdata Matrix of new data points
#' @param newcoords Matrix of new coordinates
#' @param ... Additional arguments (ignored)
#'
#' @return Matrix of predicted scores
#' @export
predict.gwpca <- function(object, newdata, newcoords, ...) {
  # Input validation
  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
  if (!is.matrix(newcoords)) newcoords <- as.matrix(newcoords)

  if (ncol(newdata) != dim(object$loadings)[2]) {
    stop("Number of variables in newdata must match original data")
  }

  if (nrow(newdata) != nrow(newcoords)) {
    stop("Number of rows in newdata must match newcoords")
  }

  n_new <- nrow(newdata)
  n_comp <- object$n_components
  scores_new <- matrix(NA, n_new, n_comp)

  # For each new point
  for (i in 1:n_new) {
    # Find nearest neighbor in original data
    distances <- sqrt((object$coords[, 1] - newcoords[i, 1])^2 +
                     (object$coords[, 2] - newcoords[i, 2])^2)
    nearest <- which.min(distances)

    # Use loadings from nearest point
    local_loadings <- object$loadings[nearest, , 1:n_comp]

    # Center data if correlation was used
    if (object$use_correlation) {
      # Estimate local center and scale from nearest points
      weights <- exp(-distances^2 / (2 * object$bandwidth^2))
      weights <- weights / sum(weights)

      # Weighted statistics
      local_center <- colSums(object$eigenvalues * weights) / sum(weights)
      newdata_centered <- newdata[i, ] - local_center
    } else {
      newdata_centered <- newdata[i, ]
    }

    # Project onto components
    scores_new[i, ] <- newdata_centered %*% local_loadings
  }

  return(scores_new)
}

#' Optimal bandwidth selection for GWPCA
#'
#' @param data Numeric matrix of data
#' @param coords Numeric matrix of coordinates
#' @param bandwidths Vector of bandwidth values to test
#' @param method Robust method to use
#' @param criterion Selection criterion ("CV" or "AIC")
#' @param k Number of components
#' @param parallel Use parallel processing
#' @param verbose Show progress
#'
#' @return List containing optimal bandwidth and criterion values
#' @export
gwpca_bandwidth_cv <- function(data,
                              coords,
                              bandwidths = NULL,
                              method = "adaptive_huber",
                              criterion = c("CV", "AIC"),
                              k = NULL,
                              parallel = TRUE,
                              verbose = FALSE) {

  # Input validation
  data <- as.matrix(data)
  coords <- as.matrix(coords)
  criterion <- match.arg(criterion)

  # Generate default bandwidths if not provided
  if (is.null(bandwidths)) {
    coord_range <- max(dist(coords))
    bandwidths <- seq(coord_range * 0.05, coord_range * 0.5, length.out = 20)
  }

  if (is.null(k)) k <- ncol(data)

  if (verbose) {
    cat("Testing", length(bandwidths), "bandwidth values using", criterion, "\n")
  }

  # Call C++ function
  optimal_bw <- gwpca_bandwidth_selection(
    data = data,
    coords = coords,
    bandwidths = bandwidths,
    method = method,
    criterion = criterion,
    k = as.integer(k),
    parallel = parallel
  )

  return(list(
    optimal = optimal_bw,
    bandwidths = bandwidths,
    criterion = criterion
  ))
}

#' Extract variance explained at each location
#'
#' @param object A gwpca object
#' @param cumulative Logical, whether to return cumulative variance
#'
#' @return Matrix of variance explained
#' @export
get_local_variance <- function(object, cumulative = FALSE) {
  if (!inherits(object, "gwpca")) {
    stop("object must be of class 'gwpca'")
  }

  if (cumulative) {
    return(t(apply(object$var_explained, 1, cumsum)))
  } else {
    return(object$var_explained)
  }
}

#' Extract loadings for a specific location
#'
#' @param object A gwpca object
#' @param location Integer index or coordinate vector
#' @param n_components Number of components to extract
#'
#' @return Matrix of loadings
#' @export
get_local_loadings <- function(object, location, n_components = NULL) {
  if (!inherits(object, "gwpca")) {
    stop("object must be of class 'gwpca'")
  }

  # Handle location specification
  if (length(location) == 1) {
    # Index provided
    idx <- location
  } else if (length(location) == 2) {
    # Coordinates provided - find nearest
    distances <- sqrt((object$coords[, 1] - location[1])^2 +
                     (object$coords[, 2] - location[2])^2)
    idx <- which.min(distances)
  } else {
    stop("location must be an index or coordinate pair")
  }

  if (is.null(n_components)) {
    n_components <- object$n_components
  }

  loadings <- object$loadings[idx, , 1:n_components]
  return(loadings)
}
