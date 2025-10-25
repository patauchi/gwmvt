# Fixing RNGScope Issue in gwmvt Package

## Problem Description

When trying to run functions from the gwmvt package, you may encounter the following error:

```
Error in gwpca_cpp(...) : function 'enterRNGScope' not provided by package 'Rcpp'
```

This error occurs due to a compatibility issue between the compiled C++ code and the Rcpp package's RNG (Random Number Generator) handling mechanism.

## Root Cause

The issue stems from the automatically generated `RcppExports.cpp` file, which includes `Rcpp::RNGScope` declarations for all exported functions, even when those functions don't use random number generation. In certain environments or with certain versions of Rcpp, this can cause runtime linking errors.

## Solution

The solution is to remove the RNGScope declarations from the generated RcppExports.cpp file. Here's the step-by-step process:

### 1. Clean Previous Builds

```bash
cd /path/to/gwmvt
rm -f src/*.o src/*.so
```

### 2. Regenerate RcppExports

```bash
R -e "Rcpp::compileAttributes('.')"
```

### 3. Remove RNGScope Lines

Remove all `Rcpp::RNGScope` lines from `src/RcppExports.cpp`:

```bash
sed -i '' 's/Rcpp::RNGScope rcpp_rngScope_gen;//g' src/RcppExports.cpp
```

Note: On Linux, use `sed -i` without the empty quotes:
```bash
sed -i 's/Rcpp::RNGScope rcpp_rngScope_gen;//g' src/RcppExports.cpp
```

### 4. Reinstall the Package

```bash
R CMD INSTALL --preclean --no-lock .
```

## Automated Script

You can create a script to automate this process:

```bash
#!/bin/bash
# fix_gwmvt.sh

# Clean previous builds
rm -f src/*.o src/*.so

# Regenerate RcppExports
R -e "Rcpp::compileAttributes('.')"

# Remove RNGScope lines
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/Rcpp::RNGScope rcpp_rngScope_gen;//g' src/RcppExports.cpp
else
    # Linux
    sed -i 's/Rcpp::RNGScope rcpp_rngScope_gen;//g' src/RcppExports.cpp
fi

# Reinstall package
R CMD INSTALL --preclean --no-lock .
```

## Important Notes

1. **Do NOT regenerate RcppExports.cpp** after applying this fix unless you're prepared to remove the RNGScope lines again.

2. **This is a workaround**, not a permanent solution. The issue may be related to:
   - Version mismatches between Rcpp at compile time vs runtime
   - Specific R installation configurations
   - macOS-specific linking issues

3. **Alternative approaches** that were tested but didn't work:
   - Using `[[Rcpp::no_rng]]` attribute (not recognized in older Rcpp versions)
   - Manually linking to Rcpp libraries
   - Using pure C API instead of Rcpp (works but defeats the purpose)

## Verification

After applying the fix, test that the package works:

```r
library(gwmvt)
set.seed(123)
n <- 100
coords <- matrix(runif(n * 2, 0, 10), ncol = 2)
data <- matrix(rnorm(n * 5), ncol = 5)
result <- gwpca(data, coords, bandwidth = 2.0)
print("Success!")
```

If you see "Success!" printed without errors, the fix has been applied successfully.

## Long-term Solution

For a more permanent solution, consider:

1. Updating to the latest versions of R, Rcpp, and RcppArmadillo
2. Reporting the issue to the Rcpp maintainers with a minimal reproducible example
3. Investigating whether the package can be restructured to avoid the issue

## References

- Similar issues have been reported in various Rcpp-based packages
- The RNGScope is used to ensure proper random number generation state management in R
- The issue appears to be environment-specific and doesn't affect all systems