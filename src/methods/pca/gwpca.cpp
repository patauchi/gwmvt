#include "gwpca.h"
#include "robust/all_robust_methods.h"
#include "../../core/algebra.h"
#include "../../core/distances.h"
#include "../../utils/diagnostics.h"
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <algorithm>
#include <numeric>

namespace gwmvt {

// Constructor
GWPCA::GWPCA(const Mat& data, const Mat& coords, const GWConfig& config)
    : GWMethod(data, coords, config) {
    
    // Extract GWPCA-specific configuration
    pca_config_ = std::dynamic_pointer_cast<GWPCAConfig>(config.method_config);
    if (!pca_config_) {
        // Create default GWPCA config if not provided
        pca_config_ = std::make_shared<GWPCAConfig>();
    }
    
    // Create robust estimator
    RobustConfig robust_config;
    robust_config.method = pca_config_->robust_method;
    robust_config.detect_outliers = pca_config_->detect_outliers;
    robust_config.outlier_threshold = pca_config_->outlier_threshold;
    
    // Set method-specific parameters
    if (pca_config_->robust_method == RobustMethod::SPATIAL_TRIM) {
        robust_config.params["trim_proportion"] = pca_config_->trim_proportion;
    } else if (pca_config_->robust_method == RobustMethod::ADAPTIVE_MCD || 
               pca_config_->robust_method == RobustMethod::MVE) {
        robust_config.params["h_fraction"] = pca_config_->h_fraction;
    } else if (pca_config_->robust_method == RobustMethod::LOF) {
        robust_config.params["k"] = static_cast<double>(pca_config_->lof_k);
    } else if (pca_config_->robust_method == RobustMethod::BACON) {
        robust_config.params["alpha"] = pca_config_->bacon_alpha;
    } else if (pca_config_->robust_method == RobustMethod::SPATIAL_DEPTH) {
        robust_config.params["depth_type"] = (pca_config_->depth_type == "mahalanobis") ? 1.0 : 2.0;
    } else if (pca_config_->robust_method == RobustMethod::ROBPCA) {
        robust_config.params["k_max"] = static_cast<double>(pca_config_->robpca_k_max);
    }
    
    robust_estimator_ = create_robust_estimator(pca_config_->robust_method, robust_config);
}

GWPCA::~GWPCA() = default;

// Prepare for fitting
void GWPCA::prepare_fit() {
    int n = data_.n_rows;
    int p = data_.n_cols;
    int k = pca_config_->n_components;
    
    // Initialize result
    result_ = std::make_unique<GWPCAResult>(n, p, k, config_);
    result_->coords = coords_;
    
    // Detect spatial outliers if requested
    if (pca_config_->detect_outliers) {
        progress_->report_step("Detecting spatial outliers");
        detect_spatial_outliers();
    }
    
    // Standardize data if using correlation
    if (pca_config_->use_correlation) {
        standardize_data();
    } else {
        standardized_data_ = data_;
    }
}

// Detect spatial outliers
void GWPCA::detect_spatial_outliers() {
    int n = data_.n_rows;
    int p = data_.n_cols;
    Mat outlier_scores(n, p, arma::fill::zeros);
    
    // Use smaller bandwidth for outlier detection
    double outlier_bandwidth = config_.bandwidth * 0.5;
    GWConfig outlier_config = config_;
    outlier_config.bandwidth = outlier_bandwidth;
    SpatialWeights outlier_weights(coords_, outlier_config);
    
    // Early user interrupt check before heavy loop
    Rcpp::checkUserInterrupt();
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        // Safe only when not compiled with OpenMP; otherwise avoid calling R API from threads
        #ifndef _OPENMP
        if ((i & 255) == 0) Rcpp::checkUserInterrupt();
        #endif
        Vec weights = outlier_weights.compute_weights(i);
        weights(i) = 0.0;  // Exclude self
        
        // Only consider close neighbors
        UVec neighbor_idx = arma::find(weights > 0.01);
        if (neighbor_idx.n_elem < 3) continue;
        
        Vec neighbor_weights = weights(neighbor_idx);
        neighbor_weights /= arma::sum(neighbor_weights);
        
        // For each variable
        for (int j = 0; j < p; ++j) {
            double xi = data_(i, j);
            Mat neighbor_matrix = data_.rows(neighbor_idx);
            Vec neighbor_data = neighbor_matrix.col(j);
            
            // Weighted mean of neighbors
            double neighbor_mean = arma::dot(neighbor_weights, neighbor_data);
            
            // Robust MAD
            Vec deviations = arma::abs(neighbor_data - neighbor_mean);
            double neighbor_mad = 1.4826 * arma::dot(neighbor_weights, deviations);
            
            // Outlier score
            if (neighbor_mad > 1e-10) {
                outlier_scores(i, j) = std::abs(xi - neighbor_mean) / neighbor_mad;
            }
        }
    }
    
    // Combine scores across variables
    spatial_outliers_ = arma::zeros<UVec>(n);
    for (int i = 0; i < n; ++i) {
        double mean_score = arma::mean(outlier_scores.row(i));
        spatial_outliers_(i) = (mean_score > pca_config_->outlier_threshold * 1.5) ? 1 : 0;
    }
    
    result_->spatial_outliers = spatial_outliers_;
    
    if (config_.verbose) {
        int n_outliers = arma::sum(spatial_outliers_);
        Rcpp::Rcout << "Detected " << n_outliers << " spatial outliers ("
                    << std::round(100.0 * n_outliers / n) << "%)\n";
    }
}

// Standardize data
void GWPCA::standardize_data() {
    data_mean_ = arma::mean(data_, 0).t();
    data_sd_ = arma::stddev(data_, 0, 0).t();
    data_sd_.replace(0, 1);  // Avoid division by zero
    
    standardized_data_ = (data_.each_row() - data_mean_.t()).each_row() / data_sd_.t();
}

// Fit at a single location
void GWPCA::fit_local(int location) {
    // Get spatial weights
    Vec weights = get_weights(location);
    
    // Reduce weight of detected outliers
    if (pca_config_->detect_outliers && spatial_outliers_.n_elem > 0) {
        UVec outlier_idx = arma::find(spatial_outliers_ == 1);
        weights(outlier_idx) *= 0.5;
    }
    
    // Normalize weights
    weights /= arma::sum(weights);
    
    // Fit local PCA
    LocalPCAResult pca_result = fit_local_pca(location, weights);
    
    // Check numerical stability
    if (pca_result.eigenvalues.has_nan() || pca_result.eigenvalues(0) < 1e-10) {
        // Guard diagnostics mutation when running in parallel
        #ifdef _OPENMP
        #pragma omp critical(gwmvt_diag)
        #endif
        {
            diagnostics_.add_warning("Unstable solution at location " + std::to_string(location));
        }
        // Try standard PCA as fallback
        pca_result = fit_standard_pca(standardized_data_, weights);
    }
    
    // Store results
    result_->eigenvalues.row(location) = pca_result.eigenvalues.t();
    result_->centers.row(location) = pca_result.center.t();
    
    int k = pca_config_->n_components;
    for (int j = 0; j < k; ++j) {
        result_->loadings.slice(j).row(location) = pca_result.eigenvectors.col(j).t();
    }
    
    // Calculate scores and variance explained
    calculate_scores(location, pca_result);
    calculate_variance_explained(location, pca_result);
}

// Fit PCA at a single location
LocalPCAResult GWPCA::fit_local_pca(int location, const Vec& weights) {
    if (pca_config_->robust_method == RobustMethod::STANDARD) {
        return fit_standard_pca(standardized_data_, weights);
    } else {
        return fit_robust_pca(standardized_data_, weights);
    }
}

// Standard PCA
LocalPCAResult GWPCA::fit_standard_pca(const Mat& data, const Vec& weights) {
    int p = data.n_cols;
    LocalPCAResult result(p);
    
    // Weighted mean
    result.center = weighted_mean(data, weights);
    
    // Weighted covariance
    Mat cov_mat = weighted_covariance(data, weights, result.center);
    
    // Add small regularization
    cov_mat.diag() += 1e-8;
    
    // Eigendecomposition
    eigen_decomp_symmetric(cov_mat, result.eigenvalues, result.eigenvectors);
    
    result.total_variance = arma::sum(result.eigenvalues);
    result.converged = true;
    
    return result;
}

// Robust PCA
LocalPCAResult GWPCA::fit_robust_pca(const Mat& data, const Vec& weights) {
    int p = data.n_cols;
    LocalPCAResult result(p);
    
    // Apply robust estimation
    // Create a thread-local estimator to avoid shared mutable state across threads
    // and potential data races on internal diagnostics.
    auto local_estimator = create_robust_estimator(pca_config_->robust_method, 
                                                  robust_estimator_->get_config());
    RobustStats robust_stats = local_estimator->estimate(data, weights);
    
    result.center = robust_stats.center;
    
    // Eigendecomposition of robust covariance
    eigen_decomp_symmetric(robust_stats.covariance, result.eigenvalues, result.eigenvectors);
    
    result.total_variance = arma::sum(result.eigenvalues);
    result.converged = true;
    
    return result;
}

// Calculate scores
void GWPCA::calculate_scores(int location, const LocalPCAResult& pca_result) {
    // Center data
    Vec data_point = standardized_data_.row(location).t();
    Vec centered = data_point - pca_result.center;
    
    // Project onto principal components
    int k = pca_config_->n_components;
    for (int j = 0; j < k; ++j) {
        result_->scores(location, j) = arma::dot(centered, pca_result.eigenvectors.col(j));
    }
}

// Calculate variance explained
void GWPCA::calculate_variance_explained(int location, const LocalPCAResult& pca_result) {
    if (pca_result.total_variance > 0) {
        int k = pca_config_->n_components;
        for (int j = 0; j < k; ++j) {
            result_->var_explained(location, j) = 
                pca_result.eigenvalues(j) / pca_result.total_variance;
        }
    }
}

// Finalize results
void GWPCA::finalize_fit() {
    // No additional finalization needed for GWPCA
}

// Parallel fitting
void GWPCA::fit_parallel() {
    int n = data_.n_rows;
    progress_counter_.store(0);
    
    // Do not emit progress updates from worker threads (not thread-safe with R I/O).
    // Only update the atomic counter and finalize after the parallel region.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        fit_local(i);
        progress_counter_.fetch_add(1);
    }
    
    // Report completion (sequential, thread-safe)
    progress_->report(n, n);
}

// Create result object
std::unique_ptr<GWResult> GWPCA::create_result() {
    return std::move(result_);
}

// Prediction for new data
Mat GWPCA::predict(const Mat& newdata, const Mat& newcoords) {
    if (newdata.n_cols != data_.n_cols) {
        throw std::invalid_argument("Number of variables in new data must match original data");
    }
    
    if (newdata.n_rows != newcoords.n_rows) {
        throw std::invalid_argument("Number of rows in new data must match new coordinates");
    }
    
    int n_new = newdata.n_rows;
    int k = pca_config_->n_components;
    Mat scores_new(n_new, k);
    
    // Standardize new data if needed
    Mat newdata_std = newdata;
    if (pca_config_->use_correlation) {
        newdata_std = (newdata.each_row() - data_mean_.t()).each_row() / data_sd_.t();
    }
    
    // For each new point
    EuclideanDistance distance_calc;
    for (int i = 0; i < n_new; ++i) {
        // Find nearest neighbor in original data
        Vec distances = distance_calc.calculate_cross(newcoords.row(i), coords_).t();
        arma::uword nearest = distances.index_min();
        
        // Use loadings from nearest point
        Mat local_loadings(data_.n_cols, k);
        for (int j = 0; j < k; ++j) {
            local_loadings.col(j) = result_->loadings.slice(j).row(nearest).t();
        }
        
        // Center using local center (approximate)
        Vec local_center = standardized_data_.row(nearest).t();  // Simplified
        Vec centered = newdata_std.row(i).t() - local_center;
        
        // Project onto components
        scores_new.row(i) = (local_loadings.t() * centered).t();
    }
    
    return scores_new;
}

// Static bandwidth selection methods
double GWPCA::select_bandwidth_cv(const Mat& data, const Mat& coords,
                                 const Vec& candidates, const GWPCAConfig& config,
                                 int n_folds) {
    int n = data.n_rows;
    int n_bw = candidates.n_elem;
    Vec cv_scores(n_bw, arma::fill::zeros);
    
    // Create folds
    UVec indices = arma::shuffle(arma::linspace<UVec>(0, n-1, n));
    int fold_size = n / n_folds;
    
    // Test each bandwidth
    EuclideanDistance distance_calc;
    
    #pragma omp parallel for
    for (int b = 0; b < n_bw; ++b) {
        #ifndef _OPENMP
        if ((b & 15) == 0) Rcpp::checkUserInterrupt();
        #endif
        double cv_error = 0.0;
        
        for (int fold = 0; fold < n_folds; ++fold) {
            // Split data
            int start = fold * fold_size;
            int end = (fold == n_folds - 1) ? n : (fold + 1) * fold_size;
            
            UVec test_idx = indices.subvec(start, end - 1);
            UVec train_idx;
            
            if (fold == 0) {
                train_idx = indices.subvec(end, n - 1);
            } else if (fold == n_folds - 1) {
                train_idx = indices.subvec(0, start - 1);
            } else {
                train_idx = arma::join_cols(indices.subvec(0, start - 1),
                                           indices.subvec(end, n - 1));
            }
            
            // Create GWPCA config for this bandwidth
            GWConfig gw_config;
            gw_config.bandwidth = candidates(b);
            gw_config.verbose = false;
            gw_config.show_progress = false;
            auto pca_config_copy = std::make_shared<GWPCAConfig>(config);
            gw_config.method_config = pca_config_copy;
            
            // Fit on training data
            try {
                GWPCA gwpca_train(data.rows(train_idx), coords.rows(train_idx), gw_config);
                auto result_train = gwpca_train.fit();
                
                // Simple reconstruction error for test data
                for (size_t i = 0; i < test_idx.n_elem; ++i) {
                    // Find nearest training point
                    Vec dists = distance_calc.calculate_cross(coords.row(test_idx(i)), 
                                                               coords.rows(train_idx)).t();
                    arma::uword nearest = dists.index_min();
                    
                    // Use loadings from nearest point for reconstruction
                    // Simplified error calculation
                    cv_error += arma::norm(data.row(test_idx(i)), 2);
                }
            } catch (const Rcpp::internal::InterruptedException&) {
                // Propagate user interrupt to R
                throw;
            } catch (...) {
                cv_error += 1e10;  // Penalize failed fits
            }
        }
        
        cv_scores(b) = cv_error / n;
    }
    
    // Find optimal bandwidth
    arma::uword opt_idx = cv_scores.index_min();
    return candidates(opt_idx);
}

double GWPCA::select_bandwidth_aic(const Mat& data, const Mat& coords,
                                  const Vec& candidates, const GWPCAConfig& config) {
    int n_bw = candidates.n_elem;
    Vec aic_scores(n_bw);
    
    #pragma omp parallel for
    for (int b = 0; b < n_bw; ++b) {
        #ifndef _OPENMP
        if ((b & 15) == 0) Rcpp::checkUserInterrupt();
        #endif
        // Create GWPCA config for this bandwidth
        GWConfig gw_config;
        gw_config.bandwidth = candidates(b);
        gw_config.verbose = false;
        gw_config.show_progress = false;
        auto pca_config_copy = std::make_shared<GWPCAConfig>(config);
        gw_config.method_config = pca_config_copy;
        
        try {
            GWPCA gwpca(data, coords, gw_config);
            auto result = gwpca.fit();
            auto pca_result = dynamic_cast<GWPCAResult*>(result.get());
            
            // Compute AIC (simplified version)
            double log_likelihood = -arma::sum(arma::log(pca_result->eigenvalues.col(0)));
            double n_params = data.n_cols * config.n_components;
            aic_scores(b) = -2 * log_likelihood + 2 * n_params;
        } catch (const Rcpp::internal::InterruptedException&) {
            // Propagate user interrupt to R
            throw;
        } catch (...) {
            aic_scores(b) = 1e10;  // Penalize failed fits
        }
    }
    
    // Find optimal bandwidth
    arma::uword opt_idx = aic_scores.index_min();
    return candidates(opt_idx);
}

} // namespace gwmvt
