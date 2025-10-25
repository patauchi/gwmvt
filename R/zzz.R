.onLoad <- function(libname, pkgname) {
  # Set RcppParallel options
  if (requireNamespace("RcppParallel", quietly = TRUE)) {
    # Set default number of threads
    default_threads <- RcppParallel::defaultNumThreads()
    if (is.null(getOption("gwmvt.threads"))) {
      options(gwmvt.threads = default_threads)
    }

    # Set thread stack size if needed
    if (.Platform$OS.type == "unix") {
      RcppParallel::setThreadOptions(numThreads = "auto")
    }
  } else if (is.null(getOption("gwmvt.threads"))) {
    options(gwmvt.threads = 1L)
  }
}

.onAttach <- function(libname, pkgname) {
  pkg_version <- utils::packageVersion(pkgname)
  packageStartupMessage(
    sprintf(
      paste(
        "gwmvt: Geographically Weighted Multivariate Analysis Toolkit",
        "Version %s",
        "Type 'citation(\"%s\")' for citing this package in publications.",
        "Use 'options(gwmvt.threads = n)' to control parallel processing.",
        sep = "\n"
      ),
      pkg_version,
      pkgname
    )
  )

  openmp_support <- try(has_openmp_support(), silent = TRUE)
  if (!isTRUE(openmp_support)) {
    packageStartupMessage(
      "Note: OpenMP support not detected. Some parallel features may be limited."
    )
  }
}

.onUnload <- function(libpath) {
  # Clean up
  library.dynam.unload("gwmvt", libpath)
}

# Package environment for storing internal data
gwmvt_env <- new.env(parent = emptyenv())

# Export environment for internal use
.gwmvt_env <- gwmvt_env
