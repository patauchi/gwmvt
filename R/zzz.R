.onLoad <- function(libname, pkgname) {
  # Register S3 methods
  registerS3method("print", "gwpca", print.gwpca)
  registerS3method("summary", "gwpca", summary.gwpca)
  registerS3method("plot", "gwpca", plot.gwpca)
  registerS3method("predict", "gwpca", predict.gwpca)

  # Set RcppParallel options
  if (requireNamespace("RcppParallel", quietly = TRUE)) {
    # Set default number of threads
    default_threads <- RcppParallel::defaultNumThreads()
    options(gwmvt.threads = default_threads)

    # Set thread stack size if needed
    if (.Platform$OS.type == "unix") {
      RcppParallel::setThreadOptions(numThreads = "auto")
    }
  }

  # Package startup message
  packageStartupMessage(
    "gwmvt: Geographically Weighted Multivariate Analysis Toolkit\n",
    "Version ", utils::packageVersion(pkgname), "\n",
    "Type 'citation(\"gwmvt\")' for citing this package in publications.\n",
    "Use 'options(gwmvt.threads = n)' to control parallel processing."
  )
}

.onAttach <- function(libname, pkgname) {
  # Check for OpenMP support
  # Temporarily disabled due to initialization issues
  # if (!check_openmp_support()) {
  #   packageStartupMessage(
  #     "Note: OpenMP support not detected. ",
  #     "Some parallel features may be limited."
  #   )
  # }
  packageStartupMessage(
    "Note: OpenMP support not detected. ",
    "Some parallel features may be limited."
  )
}

.onUnload <- function(libpath) {
  # Clean up
  library.dynam.unload("gwmvt", libpath)
}

# Helper function to check OpenMP support
check_openmp_support <- function() {
  # This will be implemented in C++
  # Temporarily return FALSE to avoid initialization issues
  return(FALSE)
  # tryCatch({
  #   .Call("_gwmvt_has_openmp_support", PACKAGE = "gwmvt")
  # }, error = function(e) {
  #   FALSE
  # })
}

# Package environment for storing internal data
gwmvt_env <- new.env(parent = emptyenv())

# Export environment for internal use
.gwmvt_env <- gwmvt_env
