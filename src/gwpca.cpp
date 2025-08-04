#include <RcppArmadillo.h>
#include <memory>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Simple structure to hold PCA results at one location
struct LocalPCAResult {
    vec eigenvalues;
    mat eigenvectors;
    vec center;
    
    LocalPCAResult(int p) : eigenvalues(p), eigenvectors(p, p), center(p) {}
};

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

// Perform local PCA at one location
LocalPCAResult local_pca(const mat& data, const vec& weights, bool use_correlation) {
    int p = data.n_cols;
    LocalPCAResult result(p);
    
    // Calculate weighted center
    result.center = weighted_mean(data, weights);
    
    // Calculate weighted covariance
    mat cov = weighted_covariance(data, weights, result.center);
    
    // Convert to correlation if requested
    if (use_correlation) {
        vec sds = sqrt(cov.diag());
        cov = cov / (sds * sds.t());
    }
    
    // Eigen decomposition
    eig_sym(result.eigenvalues, result.eigenvectors, cov);
    
    // Sort in descending order
    uvec indices = sort_index(result.eigenvalues, "descend");
    result.eigenvalues = result.eigenvalues(indices);
    result.eigenvectors = result.eigenvectors.cols(indices);
    
    return result;
}

// Main GWPCA function
// [[Rcpp::export]]
List gwpca_cpp(const arma::mat& data,
               const arma::mat& coords,
               double bandwidth,
               std::string method = "adaptive_huber",
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
               bool verbose = false) {
    
    int n = data.n_rows;
    int p = data.n_cols;
    
    // Determine number of components to retain
    if (k <= 0) k = p;
    k = std::min(k, p);
    
    // Set up parallel processing
#ifdef _OPENMP
    if (parallel && n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
#endif
    
    // Initialize output matrices
    mat eigenvalues(n, k, fill::zeros);
    cube loadings(n, p, k, fill::zeros);
    mat scores(n, k, fill::zeros);
    mat var_explained(n, k, fill::zeros);
    mat centers(n, p, fill::zeros);
    
    // Process each location
#pragma omp parallel for if(parallel)
    for (int i = 0; i < n; ++i) {
        // Calculate distances and weights
        vec distances = calculate_distances(coords, i);
        vec weights = gaussian_weights(distances, bandwidth);
        
        // Normalize weights
        weights = weights / sum(weights);
        
        // Perform local PCA
        LocalPCAResult local_result = local_pca(data, weights, use_correlation);
        
        // Store results
        centers.row(i) = local_result.center.t();
        
        // Extract k components
        for (int j = 0; j < k; ++j) {
            eigenvalues(i, j) = local_result.eigenvalues(j);
            loadings.tube(i, j) = local_result.eigenvectors.col(j);
            
            // Calculate scores
            vec centered_data = data.row(i).t() - local_result.center;
            scores(i, j) = dot(centered_data, local_result.eigenvectors.col(j));
            
            // Calculate variance explained
            double total_var = sum(local_result.eigenvalues);
            if (total_var > 0) {
                var_explained(i, j) = local_result.eigenvalues(j) / total_var;
            }
        }
    }
    
    // Create output list
    List result = List::create(
        Named("eigenvalues") = eigenvalues,
        Named("loadings") = loadings,
        Named("scores") = scores,
        Named("var_explained") = var_explained,
        Named("centers") = centers,
        Named("coords") = coords,
        Named("bandwidth") = bandwidth,
        Named("method") = method,
        Named("use_correlation") = use_correlation,
        Named("n_components") = k
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
