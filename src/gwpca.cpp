#include <RcppArmadillo.h>
#include <memory>
#include <string>
#include "methods/pca/robust/all_robust_methods.h"
#include "methods/pca/gwpca.cpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Calculate distances from one point to all others
vec calculate_distances(const mat& coords, int focal_idx) {
    int n = coords.n_rows;
    vec distances(n);
    
    for (int i = 0; i < n; ++i) {
        double dx = coords(i, 0) - coords(focal_idx, 0);
        double dy = coords(i, 1) - coords(focal_idx, 1);
        distances(i) = std::sqrt(dx * dx + dy * dy);
    }
    
    return distances;
}

// Gaussian kernel weights
vec gaussian_weights(const vec& distances, double bandwidth) {
    return exp(-(distances % distances) / (2.0 * bandwidth * bandwidth));
}

// Weighted mean
vec weighted_mean(const mat& data, const vec& weights) {
    return data.t() * weights / sum(weights);
}

// Weighted covariance
mat weighted_covariance(const mat& data, const vec& weights, const vec& center) {
    int n = data.n_rows;
    int p = data.n_cols;
    mat centered(n, p);
    
    // Center the data
    for (int i = 0; i < n; ++i) {
        centered.row(i) = data.row(i) - center.t();
    }
    
    // Calculate weighted covariance
    vec sqrt_weights = sqrt(weights);
    mat weighted_data = centered.each_col() % sqrt_weights;
    
    mat cov = weighted_data.t() * weighted_data / sum(weights);
    
    return cov;
}

// Main GWPCA function
// [[Rcpp::export]]
List gwpca_cpp(const arma::mat& data,
               const arma::mat& coords,
               double bandwidth,
               std::string method = "standard",
               bool use_correlation = false,
               int k = 0,
               bool detect_outliers = true,
               double outlier_threshold = 2.5,
               double trim_prop = 0.1,
               double h_fraction = 0.75,
               int lof_k = 10,
               double bacon_alpha = 0.05,
               std::string depth_type = "mahalanobis",
               int robpca_k_max = 10,
               bool parallel = true,
               int n_threads = 0,
               bool verbose = false,
               std::string kernel = "gaussian",
               bool adaptive_bandwidth = false,
               int adaptive_k = 30) {
    
    int p = data.n_cols;
    int n_components = (k <= 0) ? p : std::min(k, p);
    
    // Configure core settings
    gwmvt::GWConfig config;
    config.bandwidth = bandwidth;
    // Kernel selection
    try {
        if (kernel == "gaussian") config.kernel_type = gwmvt::KernelType::GAUSSIAN;
        else if (kernel == "bisquare") config.kernel_type = gwmvt::KernelType::BISQUARE;
        else if (kernel == "exponential") config.kernel_type = gwmvt::KernelType::EXPONENTIAL;
        else if (kernel == "tricube") config.kernel_type = gwmvt::KernelType::TRICUBE;
        else if (kernel == "boxcar") config.kernel_type = gwmvt::KernelType::BOXCAR;
        else if (kernel == "adaptive") {
            // Alias for gaussian with adaptive bandwidth
            config.kernel_type = gwmvt::KernelType::GAUSSIAN;
            adaptive_bandwidth = true;
        } else {
            Rcpp::stop("Unknown kernel: %s", kernel);
        }
    } catch (const Rcpp::internal::InterruptedException&) {
        throw;
    } catch (...) {
        Rcpp::stop("Failed to set kernel type");
    }
    config.adaptive_bandwidth = adaptive_bandwidth;
    config.adaptive_k = adaptive_k;
    config.parallel_strategy = parallel ? gwmvt::ParallelStrategy::AUTO
                                        : gwmvt::ParallelStrategy::SEQUENTIAL;
    config.n_threads = n_threads;
    config.verbose = verbose;
    config.show_progress = verbose;
    
    // Configure GWPCA-specific options
    auto pca_config = std::make_shared<gwmvt::GWPCAConfig>();
    pca_config->n_components = n_components;
    pca_config->use_correlation = use_correlation;
    pca_config->detect_outliers = detect_outliers;
    pca_config->outlier_threshold = outlier_threshold;
    pca_config->trim_proportion = trim_prop;
    pca_config->h_fraction = h_fraction;
    pca_config->lof_k = lof_k;
    pca_config->bacon_alpha = bacon_alpha;
    pca_config->depth_type = depth_type;
    pca_config->robpca_k_max = robpca_k_max;
    
    try {
        pca_config->robust_method = gwmvt::parse_robust_method(method);
    } catch (const std::invalid_argument& ex) {
        stop(ex.what());
    }

    config.method_config = pca_config;
    
    // Fit model
    gwmvt::GWPCA gwpca_obj(data, coords, config);
    auto result_ptr = gwpca_obj.fit();
    auto* pca_result = dynamic_cast<gwmvt::GWPCAResult*>(result_ptr.get());
    
    if (!pca_result) {
        stop("Internal error: failed to retrieve GWPCA result.");
    }
    
    // Subset eigenvalues to requested components
    arma::mat eigenvalues_subset = (n_components < p)
        ? pca_result->eigenvalues.cols(0, n_components - 1)
        : pca_result->eigenvalues;
    
    // Prepare output
    List result = List::create(
        Named("eigenvalues") = eigenvalues_subset,
        // Standard deviations of PCs (sqrt of eigenvalues) for convenience/comparison
        Named("sdev") = arma::sqrt(eigenvalues_subset),
        Named("loadings") = pca_result->loadings,
        Named("scores") = pca_result->scores,
        Named("var_explained") = pca_result->var_explained,
        Named("centers") = pca_result->centers,
        Named("coords") = pca_result->coords,
        Named("bandwidth") = bandwidth,
        Named("method") = method,
        Named("use_correlation") = use_correlation,
        Named("n_components") = n_components,
        Named("spatial_outliers") = pca_result->spatial_outliers,
        Named("kernel") = kernel,
        Named("adaptive_bandwidth") = adaptive_bandwidth,
        Named("adaptive_k") = adaptive_k
    );
    
    // Attach diagnostics
    const gwmvt::DiagnosticInfo& diag = gwpca_obj.get_diagnostics();
    result.attr("diagnostics") = List::create(
        Named("numerical_issues") = diag.numerical_issues,
        Named("convergence_issues") = diag.convergence_issues,
        Named("warnings") = diag.warnings,
        Named("errors") = diag.errors
    );
    
    return result;
}

// Spatial outlier detection
// [[Rcpp::export]]
arma::uvec detect_spatial_outliers_cpp(const arma::mat& data,
                                       const arma::mat& coords,
                                       double bandwidth,
                                       double threshold = 2.5) {
    int n = data.n_rows;
    vec outlier_scores(n, fill::zeros);
    
    // Calculate local Mahalanobis distances
    for (int i = 0; i < n; ++i) {
        if ((i & 255) == 0) Rcpp::checkUserInterrupt();
        vec distances = calculate_distances(coords, i);
        vec weights = gaussian_weights(distances, bandwidth);
        weights = weights / sum(weights);
        
        vec center = weighted_mean(data, weights);
        mat cov = weighted_covariance(data, weights, center);
        
        // Calculate Mahalanobis distance
        vec diff = data.row(i).t() - center;
        outlier_scores(i) = sqrt(as_scalar(diff.t() * inv_sympd(cov) * diff));
    }
    
    // Identify outliers
    uvec outliers = find(outlier_scores > threshold);
    
    return outliers;
}

// Adaptive bandwidth using k-nearest neighbors
// [[Rcpp::export]]
arma::vec adaptive_bandwidth_nn(const arma::mat& coords, int k) {
    int n = coords.n_rows;
    vec bandwidths(n);
    
    for (int i = 0; i < n; ++i) {
        if ((i & 255) == 0) Rcpp::checkUserInterrupt();
        vec distances = calculate_distances(coords, i);
        vec sorted_distances = sort(distances);
        bandwidths(i) = sorted_distances(k);
    }
    
    return bandwidths;
}

// Moran's I statistic
// [[Rcpp::export]]
double morans_i(const arma::vec& values, const arma::mat& coords, double bandwidth) {
    int n = values.n_elem;
    double mean_val = mean(values);
    vec centered = values - mean_val;
    
    double numerator = 0.0;
    double denominator = sum(centered % centered);
    double weight_sum = 0.0;
    
    for (int i = 0; i < n; ++i) {
        vec distances = calculate_distances(coords, i);
        vec weights = gaussian_weights(distances, bandwidth);
        
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                numerator += weights(j) * centered(i) * centered(j);
                weight_sum += weights(j);
            }
        }
    }
    
    return (n / weight_sum) * (numerator / denominator);
}

// Local Moran's I
// [[Rcpp::export]]
arma::vec local_morans_i(const arma::vec& values, const arma::mat& coords, double bandwidth) {
    int n = values.n_elem;
    vec local_i(n);
    double mean_val = mean(values);
    vec centered = values - mean_val;
    double var_val = var(values);
    
    for (int i = 0; i < n; ++i) {
        vec distances = calculate_distances(coords, i);
        vec weights = gaussian_weights(distances, bandwidth);
        
        double local_sum = 0.0;
        double weight_sum = 0.0;
        
        for (int j = 0; j < n; ++j) {
            if (i != j && weights(j) > 1e-10) {
                local_sum += weights(j) * centered(j);
                weight_sum += weights(j);
            }
        }
        
        local_i(i) = (centered(i) / var_val) * local_sum;
    }
    
    return local_i;
}

// Check OpenMP support
// [[Rcpp::export]]
bool has_openmp_support() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

// Get maximum number of threads
// [[Rcpp::export]]
int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Set number of threads
// [[Rcpp::export]]
void set_num_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}

// Simple test function to isolate RNGScope issue
