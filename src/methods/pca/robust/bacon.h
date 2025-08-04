#ifndef GWMVT_METHODS_PCA_ROBUST_BACON_H
#define GWMVT_METHODS_PCA_ROBUST_BACON_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <algorithm>
#include <set>

namespace gwmvt {

// BACON (Blocked Adaptive Computationally Efficient Outlier Nominators) estimator
class BACONEstimator : public RobustEstimator {
private:
    double alpha_;           // Significance level for outlier detection
    double init_fraction_;   // Initial fraction of data to use
    int max_iterations_;     // Maximum BACON iterations
    bool use_correction_;    // Use finite sample correction
    
public:
    // Constructor
    explicit BACONEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), alpha_(0.05), init_fraction_(0.3), 
          max_iterations_(20), use_correction_(true) {
        
        // Check for custom alpha
        auto it = config.params.find("alpha");
        if (it != config.params.end()) {
            alpha_ = std::max(0.001, std::min(0.5, it->second));
        }
        
        // Check for initial fraction
        it = config.params.find("init_fraction");
        if (it != config.params.end()) {
            init_fraction_ = std::max(0.1, std::min(0.5, it->second));
        }
        
        // Check for max iterations
        it = config.params.find("max_iterations");
        if (it != config.params.end()) {
            max_iterations_ = std::max(5, static_cast<int>(it->second));
        }
        
        // Check for correction flag
        it = config.params.find("use_correction");
        if (it != config.params.end()) {
            use_correction_ = (it->second > 0.5);
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Initialize with basic subset
        int init_size = std::max(p + 1, static_cast<int>(std::ceil(init_fraction_ * arma::sum(weights > 0.01))));
        
        // Select initial subset based on highest weights
        UVec weight_order = arma::sort_index(weights, "descend");
        std::set<int> current_subset;
        
        for (int i = 0; i < init_size && i < n; ++i) {
            if (weights(weight_order(i)) > 0.01) {
                current_subset.insert(weight_order(i));
            }
        }
        
        // Ensure minimum size
        if (current_subset.size() < static_cast<size_t>(p + 1)) {
            diagnostics_.add_error("BACON: Not enough observations for initial subset");
            // Fallback to standard weighted estimation
            return compute_standard_estimate(data, weights);
        }
        
        // Chi-square cutoff for outlier detection
        double chi2_cutoff = R::qchisq(1.0 - alpha_, p, true, false);
        
        // BACON iterations
        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Compute statistics for current subset
            RobustStats subset_stats = compute_subset_statistics(data, weights, current_subset);
            
            // Compute Mahalanobis distances for all observations
            Vec distances = compute_mahalanobis_distances(data, subset_stats);
            
            // Apply finite sample correction if requested
            if (use_correction_) {
                double correction = compute_finite_sample_correction(current_subset.size(), p, n);
                chi2_cutoff *= correction;
            }
            
            // Update subset: include all points with distance <= cutoff
            std::set<int> new_subset;
            for (int i = 0; i < n; ++i) {
                if (weights(i) > 0.01 && distances(i) <= chi2_cutoff) {
                    new_subset.insert(i);
                }
            }
            
            // Check convergence
            if (new_subset == current_subset) {
                break;
            }
            
            // Ensure subset doesn't shrink below minimum
            if (new_subset.size() < static_cast<size_t>(p + 1)) {
                // Keep points with smallest distances
                UVec dist_order = arma::sort_index(distances);
                new_subset.clear();
                for (int i = 0; i < p + 1; ++i) {
                    if (weights(dist_order(i)) > 0.01) {
                        new_subset.insert(dist_order(i));
                    }
                }
            }
            
            current_subset = new_subset;
            
            // Reset chi-square cutoff for next iteration
            chi2_cutoff = R::qchisq(1.0 - alpha_, p, true, false);
        }
        
        // Final statistics from clean subset
        RobustStats final_stats = compute_subset_statistics(data, weights, current_subset);
        
        // Identify outliers
        Vec final_distances = compute_mahalanobis_distances(data, final_stats);
        final_stats.outliers = arma::zeros<UVec>(n);
        
        for (int i = 0; i < n; ++i) {
            if (final_distances(i) > chi2_cutoff) {
                final_stats.outliers(i) = 1;
            }
        }
        
        // Update weights to reflect subset membership
        final_stats.weights = arma::zeros<Vec>(n);
        for (int idx : current_subset) {
            final_stats.weights(idx) = weights(idx);
        }
        final_stats.weights /= arma::sum(final_stats.weights);
        
        return final_stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return "bacon"; 
    }
    
    double get_breakdown_point() const override { 
        // BACON breakdown depends on initial subset size
        return std::min(0.5, 1.0 - init_fraction_);
    }
    
    double get_efficiency() const override { 
        // BACON achieves high efficiency by identifying clean subset
        return 0.95;
    }
    
private:
    // Compute statistics for a subset
    RobustStats compute_subset_statistics(const Mat& data, const Vec& weights,
                                         const std::set<int>& subset) {
        int p = data.n_cols;
        RobustStats stats(p);
        
        // Extract subset data and weights
        UVec subset_vec(subset.size());
        int idx = 0;
        for (int i : subset) {
            subset_vec(idx++) = i;
        }
        
        Mat subset_data = data.rows(subset_vec);
        Vec subset_weights = weights(subset_vec);
        subset_weights /= arma::sum(subset_weights);
        
        // Compute weighted statistics
        stats.center = weighted_mean(subset_data, subset_weights);
        stats.covariance = weighted_covariance(subset_data, subset_weights, stats.center);
        
        // Regularize covariance
        regularize_covariance(stats.covariance);
        stats.scale = arma::sqrt(stats.covariance.diag());
        
        return stats;
    }
    
    // Compute Mahalanobis distances
    Vec compute_mahalanobis_distances(const Mat& data, const RobustStats& stats) {
        int n = data.n_rows;
        Vec distances(n);
        
        Mat data_centered = data.each_row() - stats.center.t();
        Mat cov_inv = safe_inv(stats.covariance);
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            distances(i) = arma::as_scalar(x.t() * cov_inv * x);
        }
        
        return distances;
    }
    
    // Compute finite sample correction factor
    double compute_finite_sample_correction(int subset_size, int p, int n) {
        // Hardin and Rocke (2005) correction
        double m = subset_size;
        double h = (m + p + 1.0) / 2.0;
        
        // Correction factor
        double c_alpha = R::qchisq(1.0 - alpha_, p, true, false) / p;
        double c_h = -R::qnorm(h / n, 0.0, 1.0, true, false);
        
        double correction = 1.0 + (c_h * std::sqrt(2.0)) / std::sqrt(m) +
                           (2.0 * c_h * c_h) / m;
        
        // Additional small sample correction
        if (m < 10 * p) {
            correction *= (1.0 + 0.5 / (m - p));
        }
        
        return correction;
    }
    
    // Compute standard estimate as fallback
    RobustStats compute_standard_estimate(const Mat& data, const Vec& weights) {
        int p = data.n_cols;
        RobustStats stats(p);
        
        stats.center = weighted_mean(data, weights);
        stats.covariance = weighted_covariance(data, weights, stats.center);
        regularize_covariance(stats.covariance);
        stats.scale = arma::sqrt(stats.covariance.diag());
        stats.weights = weights;
        stats.outliers = arma::zeros<UVec>(data.n_rows);
        
        return stats;
    }
    
    // Iterative version with convergence tracking
    RobustStats estimate_iterative(const Mat& data, const Vec& weights,
                                  ConvergenceInfo& conv_info) override {
        // BACON is inherently iterative, so we use the main estimate
        RobustStats result = estimate(data, weights);
        
        conv_info.converged = true;
        conv_info.iterations = 1;  // Actual iterations happen inside estimate()
        conv_info.message = "BACON completed";
        
        return result;
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::BACON, BACONEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_BACON_H