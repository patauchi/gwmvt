# gwmvt: Geographically Weighted Multivariate Analysis

[![R Package](https://img.shields.io/badge/R%20Package-gwmvt-blue)](https://github.com/patauchi/gwmvt)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

`gwmvt` delivers high-performance geographically weighted multivariate analysis with an emphasis on robust estimation. Core routines are written in C++ (Armadillo + Rcpp) and leverage OpenMP / RcppParallel when available, making the package well-suited to large spatial datasets.

## Features

- **GWPCA core**: Standard geographically weighted PCA and a suite of robust estimators.
- **Robust choices**: Huber, MCD, MVE, S-, MM-, LOF-, BACON-, spatial-depth, spatial-trim, and ROBPCA variants.
- **Parallel aware**: Uses RcppParallel when present; respects `options(gwmvt.threads)`.
- **Startup diagnostics**: Detects OpenMP availability and reports thread defaults politely.

## Installation

```r
# Install build deps
install.packages(c("devtools", "Rcpp", "RcppArmadillo", "RcppParallel"))

# Development version
devtools::install_github("patauchi/gwmvt")
```

System requirements:

- R >= 4.0.0
- C++17-capable compiler
- GNU make
- OpenMP recommended (Automatically detected; package falls back gracefully when absent.)

#### Enabling OpenMP

OpenMP dramatically accelerates the heavy GWPCA workloads. Installation differs by platform:

- **macOS** (homebrew + clang-omp):

  ```sh
  brew install llvm libomp
  echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
  echo 'export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"' >> ~/.zshrc
  echo 'export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"' >> ~/.zshrc
  ```

  After restarting the shell, configure your R `~/.R/Makevars` (or `Makevars` in `~/.R`):

  ```make
  CC=/opt/homebrew/opt/llvm/bin/clang
  CXX=/opt/homebrew/opt/llvm/bin/clang++
  CXX11=/opt/homebrew/opt/llvm/bin/clang++
  CXX14=/opt/homebrew/opt/llvm/bin/clang++
  CXX17=/opt/homebrew/opt/llvm/bin/clang++
  CFLAGS=-O3 -march=native
  CXXFLAGS=-O3 -march=native
  LDFLAGS+=-L/opt/homebrew/opt/llvm/lib
  CPPFLAGS+=-I/opt/homebrew/opt/llvm/include
  SHLIB_OPENMP_CFLAGS=-fopenmp
  SHLIB_OPENMP_CXXFLAGS=-fopenmp
  SHLIB_OPENMP_FCFLAGS=-fopenmp
  SHLIB_OPENMP_FFLAGS=-fopenmp
  ```

  Intel macs should replace `/opt/homebrew` with `/usr/local`.

- **Windows** (Rtools + gcc):

  1. Install the latest [Rtools](https://cran.r-project.org/bin/windows/Rtools/).
  2. Ensure `C:\rtools44\usr\bin` and the corresponding `mingw` paths (e.g. `C:\rtools44\x86_64-w64-mingw32.static.posix\bin`) are on `PATH`.
  3. Open `~\Documents\.R\Makevars.win` (create if missing) and add:

     ```make
     CXX11FLAGS=-O3 -march=native -mtune=native
     CXX14FLAGS=$(CXX11FLAGS)
     CXX17FLAGS=$(CXX11FLAGS)
     CXXFLAGS=$(CXX11FLAGS)
     CFLAGS=$(CXX11FLAGS)
     SHLIB_OPENMP_CFLAGS=-fopenmp
     SHLIB_OPENMP_CXXFLAGS=-fopenmp
     SHLIB_OPENMP_FFLAGS=-fopenmp
     SHLIB_OPENMP_FCFLAGS=-fopenmp
     ```

  4. Restart R (or RStudio) so the toolchain and flags are picked up.

On both platforms, verify `gwmvt::has_openmp_support()` returns `TRUE` after rebuilding any packages that rely on OpenMP.

## Quick Start

```r
library(gwmvt)

set.seed(123)                 # Recommended for stochastic robust methods
n <- 500; p <- 5
coords <- matrix(runif(n * 2, 0, 10), ncol = 2)
data   <- matrix(rnorm(n * p), ncol = p)

# Standard GWPCA
result <- gwpca(data, coords, bandwidth = 0.3)
print(result)

# Robust GWPCA with outlier detection
result_robust <- gwpca(
  data,
  coords,
  bandwidth = 0.3,
  method = "mm_estimator",
  detect_outliers = TRUE,
  parallel = TRUE
)

summary(result_robust)
plot(result_robust, type = "variance")
```

## Core Functions

### `gwpca()`

Main entry point for geographically weighted PCA.

```r
gwpca(
  data,
  coords,
  bandwidth,
  method = c(
    "standard", "adaptive_huber", "adaptive_mcd", "spatial_trim",
    "mve", "s_estimator", "mm_estimator", "lof", "bacon",
    "spatial_depth", "robpca"
  ),
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
  verbose = FALSE
)
```

`gwpca()` returns an object of class `gwpca` with S3 methods for `print()`, `summary()`, `plot()`, and `predict()`. Use `get_local_variance()` and `get_local_loadings()` to extract location-specific results.

### `gwpca_bandwidth_cv()`

Convenience helper for bandwidth selection via cross-validation or AIC.

```r
cv <- gwpca_bandwidth_cv(
  data,
  coords,
  bandwidths = NULL,
  method = "adaptive_huber",
  criterion = c("CV", "AIC"),
  k = NULL,
  parallel = TRUE,
  verbose = FALSE
)

cv$optimal   # chosen bandwidth
cv$scores    # score per candidate bandwidth
```

Internally `gwpca_bandwidth_cv()` evaluates candidate bandwidths with robust guards and returns a tidy list containing the optimal bandwidth, the candidate grid, criterion, and per-bandwidth score vector.

## Practical Tips

- **Threading**: The package consults `options(gwmvt.threads)` at load time. Leave it unset for automatic detection or override globally via `options(gwmvt.threads = <n>)`.
- **Randomised estimators**: Methods such as `"robpca"` and `"adaptive_mcd"` rely on random subsets. Call `set.seed()` from R before invoking `gwpca()` to ensure reproducibility and to silence RNG warnings.
- **Outlier diagnostics**: `summary.gwpca()` reports robust variance summaries and outlier counts to help gauge estimator behaviour across space.

## Visualization

```r
plot(result, type = "variance", component = 1)
plot(result, type = "scores",   component = 1)
plot(result, type = "loadings", component = 1)
plot(result, type = "eigenvalues")
```

## Roadmap

Planned additions include geographically weighted CCA, PLS, redundancy analysis, factor analysis, and correlation diagnostics. Contributions and issue reports are welcome via the GitHub tracker.

## Contributing

Please open an issue or pull request describing proposed changes. For significant contributions, include reproducible examples and confirm that `devtools::check(document = FALSE)` runs cleanly (aside from the GNU make NOTE).

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```
@Manual{gwmvt,
  title = {gwmvt: Geographically Weighted Multivariate Analysis Toolkit},
  author = {P.Joser Atauchi},
  year = {2024},
  note = {R package version 0.1.0},
  url = {https://github.com/patauchi/gwmvt}
}
```

## License

This package is licensed under GPL (>= 3).
