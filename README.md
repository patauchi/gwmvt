# gwmvt: Geographically Weighted Multivariate Analysis Toolkit

[![R Package](https://img.shields.io/badge/R%20Package-gwmvt-blue)](https://github.com/yourusername/gwmvt)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

The `gwmvt` package provides high-performance implementations of geographically weighted multivariate analysis methods. All core computations are implemented in C++ with support for parallel processing, making it suitable for analyzing large spatial datasets.

## Features

- **High Performance**: Core algorithms implemented in C++ with OpenMP and RcppParallel support
- **Robust Methods**: Multiple robust estimation methods to handle spatial outliers
- **Parallel Processing**: Automatic parallelization for large datasets
- **Memory Efficient**: Optimized memory usage for large-scale spatial analysis
- **Flexible**: Works with any spatial data in matrix format

## Currently Implemented Methods

### Geographically Weighted PCA (GWPCA)
- Standard GWPCA
- Robust GWPCA with multiple methods:
  - Adaptive Huber M-estimator
  - Adaptive Minimum Covariance Determinant (MCD)
  - Spatial trimming
  - Minimum Volume Ellipsoid (MVE)
  - S-estimators
  - MM-estimators
  - Local Outlier Factor (LOF)
  - BACON algorithm
  - Spatial depth-based methods
  - ROBPCA

## Installation

### Development Version

```r
# Install development tools if needed
install.packages(c("devtools", "Rcpp", "RcppArmadillo", "RcppParallel"))

# Install from GitHub (when available)
devtools::install_github("yourusername/gwmvt")

# Or install from local directory
devtools::install("path/to/gwmvt")
```

### System Requirements

- R (>= 4.0.0)
- C++14 compiler
- GNU make
- OpenMP support (optional but recommended for parallel processing)

## Quick Start

```r
library(gwmvt)

# Generate example data
n <- 500
coords <- matrix(runif(n * 2, 0, 10), ncol = 2)
data <- matrix(rnorm(n * 5), ncol = 5)

# Basic GWPCA
result <- gwpca(data, coords, bandwidth = 2.0)
print(result)

# Robust GWPCA with outlier detection
result_robust <- gwpca(data, coords, 
                      bandwidth = 2.0,
                      method = "adaptive_huber",
                      detect_outliers = TRUE,
                      verbose = TRUE)

# Summary and visualization
summary(result_robust)
plot(result_robust, type = "variance")
```

## Core Functions

### `gwpca()`
Main function for Geographically Weighted Principal Component Analysis.

```r
gwpca(data, coords, bandwidth, 
      method = c("adaptive_huber", "adaptive_mcd", "spatial_trim", "standard"),
      use_correlation = FALSE,
      k = NULL,
      detect_outliers = TRUE,
      outlier_threshold = 2.5,
      parallel = TRUE,
      n_threads = 0,
      verbose = FALSE)
```

#### Parameters:
- `data`: Numeric matrix (n × p) of data values
- `coords`: Numeric matrix (n × 2) of coordinates
- `bandwidth`: Positive numeric value for spatial bandwidth
- `method`: Robust method to use
- `use_correlation`: Use correlation matrix instead of covariance
- `k`: Number of components to retain
- `detect_outliers`: Detect spatial outliers before analysis
- `parallel`: Use parallel processing
- `verbose`: Show progress information

### `gwpca_bandwidth_cv()`
Optimal bandwidth selection using cross-validation or AIC.

```r
gwpca_bandwidth_cv(data, coords, 
                   bandwidths = NULL,
                   method = "adaptive_huber",
                   criterion = c("CV", "AIC"),
                   k = NULL,
                   parallel = TRUE)
```

## Utility Functions

- `detect_spatial_outliers_cpp()`: Detect spatial outliers using Local Moran's I
- `adaptive_bandwidth_nn()`: Calculate adaptive bandwidth using k-nearest neighbors
- `adaptive_bandwidth_quantile()`: Calculate adaptive bandwidth using distance quantiles
- `morans_i()`: Calculate global Moran's I statistic
- `local_morans_i()`: Calculate local Moran's I statistics
- `create_spatial_weights()`: Create spatial weight matrix
- `get_local_variance()`: Extract variance explained at each location
- `get_local_loadings()`: Extract loadings for specific locations

## Performance Optimization

### Parallel Processing
The package automatically detects and uses available CPU cores:

```r
# Check OpenMP support
gwmvt::has_openmp_support()

# Get/set number of threads
gwmvt::get_max_threads()
gwmvt::set_num_threads(4)

# Or use R option
options(gwmvt.threads = 4)
```

### Memory Management
For very large datasets, use batch processing:

```r
# Create batches for processing
batches <- create_batches(n = 10000, max_memory = 2048)
```

## Advanced Usage

### Custom Robust Methods
```r
# Use different robust methods
methods <- c("adaptive_huber", "adaptive_mcd", "spatial_trim", 
             "mve", "s_estimator", "mm_estimator")

results <- lapply(methods, function(m) {
  gwpca(data, coords, bandwidth = 2.0, method = m)
})
```

### Spatial Weight Matrices
```r
# Create spatial weight matrix
W <- create_spatial_weights(coords, 
                           bandwidth = 2.0,
                           kernel = "gaussian",
                           adaptive = TRUE,
                           k = 30)
```

## Visualization

The package provides S3 methods for visualization:

```r
# Different plot types
plot(result, type = "variance", component = 1)  # Variance explained
plot(result, type = "scores", component = 1)    # Component scores
plot(result, type = "loadings", component = 1)  # Loadings biplot
plot(result, type = "eigenvalues")              # Eigenvalue distribution
```

## Roadmap

Future implementations will include:
- Geographically Weighted Canonical Correlation Analysis (GWCCA)
- Geographically Weighted Partial Least Squares (GWPLS)
- Geographically Weighted Redundancy Analysis (GWRDA)
- Geographically Weighted Factor Analysis (GWFA)
- Geographically Weighted Correlation (GWCORR)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```
@Manual{gwmvt,
  title = {gwmvt: Geographically Weighted Multivariate Analysis Toolkit},
  author = {Your Name},
  year = {2024},
  note = {R package version 0.1.0},
  url = {https://github.com/yourusername/gwmvt}
}
```

## License

This package is licensed under GPL (>= 3).

## Acknowledgments

This package builds upon the excellent work of:
- Rcpp, RcppArmadillo, and RcppParallel development teams
- Original GWPCA methodology developers
- The R spatial analysis community