#ifndef GWMVT_METHODS_PCA_ROBUST_SPATIAL_TRIM_H
#define GWMVT_METHODS_PCA_ROBUST_SPATIAL_TRIM_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <algorithm>

namespace gwmvt {

// Spatial trimming robust estimator
class SpatialTrimEstimator : public RobustEstimator {
private:
    double trim_proportion_;  // Proportion to trim
    bool adaptive_trim_;      // Use adaptive trimming based on local density
    
public:
    // Constructor
    explicit SpatialTrimEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), trim_proportion_(0.1), adaptive_trim_(true) {
        
        // Check for custom trim proportion
        auto it = config.params.find("trim_proportion");
        if (it != config.params.end()) {
            trim_proportion_ = std::max(0.0, std::min(0.5, it->second));
        }
        
        // Check for adaptive flag
        it = config.params.find("adaptive_trim");
        if (it != config.params.end()) {
            adaptive_trim_ = (it->second > 0.5);
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Get initial robust estimate
        RobustStats initial_stats = compute_initial_estimate(data, weights);
        
        // Calculate Mahalanobis distances
        Mat cov_inv = safe_inv(initial_stats.covariance);
        Vec mahal_dist = mahalanobis_distance(data, initial_stats.center, cov_inv);
        
        // Determine trimming threshold
        double threshold;
        if (adaptive_trim_) {
            // Adaptive threshold based on local density of distances
            Vec sorted_dist = arma::sort(mahal_dist);
            
            // Find elbow point in distance distribution
            threshold = find_elbow_threshold(sorted_dist, trim_proportion_);
        } else {
            // Fixed quantile threshold
            threshold = weighted_quantile(mahal_dist, weights, 1.0 - trim_proportion_);
        }
        
        // Create trimmed weights
        Vec trim_weights = weights;
        int n_trimmed = 0;
        
        for (int i = 0; i < n; ++i) {
            if (mahal_dist(i) > threshold) {
                trim_weights(i) *= 0.1;  // Reduce weight of outliers
                n_trimmed++;
            }
        }
        
        // Renormalize weights
        trim_weights /= arma::sum(trim_weights);
        
        // Recompute statistics with trimmed weights
        RobustStats final_stats(p);
        final_stats.center = weighted_mean(data, trim_weights);
        final_stats.covariance = weighted_covariance(data, trim_weights, final_stats.center);
        
        // Regularize covariance
        regularize_covariance(final_stats.covariance);
        
        // Compute scale
        final_stats.scale = arma::sqrt(final_stats.covariance.diag());
        
        // Store weights and identify outliers
        final_stats.weights = trim_weights;
        final_stats.outliers = arma::zeros<UVec>(n);
        for (int i = 0; i < n; ++i) {
            if (mahal_dist(i) > threshold) {
                final_stats.outliers(i) = 1;
            }
        }
        
        // Add diagnostic info
        if (config_.verbose) {
            double trim_actual = static_cast<double>(n_trimmed) / n;
            if (std::abs(trim_actual - trim_proportion_) > 0.1) {
                diagnostics_.add_warning("Actual trim proportion (" + 
                                       std::to_string(trim_actual) + 
                                       ") differs from target (" + 
                                       std::to_string(trim_proportion_) + ")");
            }
        }
        
        return final_stats;
    }
    
    // Iterative estimation with convergence tracking
    RobustStats estimate_iterative(const Mat& data, const Vec& weights,
                                  ConvergenceInfo& conv_info) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Initialize with standard estimate
        RobustStats stats(p);
        stats.center = weighted_mean(data, weights);
        Vec old_center = stats.center;
        
        // Iterative trimming
        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // Current covariance estimate
            stats.covariance = weighted_covariance(data, weights, stats.center);
            regularize_covariance(stats.covariance);
            
            // Calculate Mahalanobis distances
            Mat cov_inv = safe_inv(stats.covariance);
            Vec mahal_dist = mahalanobis_distance(data, stats.center, cov_inv);
            
            // Determine threshold
            double threshold = weighted_quantile(mahal_dist, weights, 1.0 - trim_proportion_);
            
            // Update weights
            Vec new_weights = weights;
            for (int i = 0; i < n; ++i) {
                if (mahal_dist(i) > threshold) {
                    new_weights(i) *= 0.1;
                }
            }
            new_weights /= arma::sum(new_weights);
            
            // Update center
            stats.center = weighted_mean(data, new_weights);
            
            // Check convergence
            if (check_convergence(old_center, stats.center, iter + 1, conv_info)) {
                break;
            }
            
            old_center = stats.center;
        }
        
        // Final estimate
        return estimate(data, weights);
    }
    
    // Method properties
    std::string get_name() const override { 
        return "spatial_trim"; 
    }
    
    double get_breakdown_point() const override { 
        return trim_proportion_; 
    }
    
    double get_efficiency() const override { 
        // Efficiency depends on trim proportion
        return 1.0 - trim_proportion_;
    }
    
private:
    // Compute initial robust estimate
    RobustStats compute_initial_estimate(const Mat& data, const Vec& weights) {
        int p = data.n_cols;
        RobustStats stats(p);
        
        // Use coordinate-wise median and MAD for initial estimate
        for (int j = 0; j < p; ++j) {
            stats.center(j) = weighted_median(data.col(j), weights);
            stats.scale(j) = weighted_mad(data.col(j), weights, stats.center(j));
        }
        
        // Initial covariance as diagonal matrix
        stats.covariance = arma::diagmat(stats.scale % stats.scale);
        
        // Try to improve with one iteration
        Vec trim_weights = weights;
        Vec distances(data.n_rows);
        
        // Simple distance measure
        for (int i = 0; i < data.n_rows; ++i) {
            Vec centered = (data.row(i).t() - stats.center) / stats.scale;
            distances(i) = arma::norm(centered, 2);
        }
        
        // Trim based on distances
        double dist_threshold = weighted_quantile(distances, weights, 0.9);
        for (int i = 0; i < data.n_rows; ++i) {
            if (distances(i) > dist_threshold) {
                trim_weights(i) *= 0.5;
            }
        }
        trim_weights /= arma::sum(trim_weights);
        
        // Recompute with trimmed weights
        stats.center = weighted_mean(data, trim_weights);
        stats.covariance = weighted_covariance(data, trim_weights, stats.center);
        regularize_covariance(stats.covariance);
        
        return stats;
    }
    
    // Find elbow point in sorted distances for adaptive threshold
    double find_elbow_threshold(const Vec& sorted_dist, double target_prop) {
        int n = sorted_dist.n_elem;
        int target_idx = static_cast<int>((1.0 - target_prop) * n);
        
        // Compute curvature at each point
        Vec curvature(n - 2);
        for (int i = 1; i < n - 1; ++i) {
            // Second derivative approximation
            curvature(i - 1) = sorted_dist(i + 1) - 2 * sorted_dist(i) + sorted_dist(i - 1);
        }
        
        // Find maximum curvature point near target
        int search_start = std::max(1, target_idx - n / 10);
        int search_end = std::min(n - 2, target_idx + n / 10);
        
        double max_curv = 0.0;
        int elbow_idx = target_idx;
        
        for (int i = search_start; i < search_end; ++i) {
            if (curvature(i - 1) > max_curv) {
                max_curv = curvature(i - 1);
                elbow_idx = i;
            }
        }
        
        return sorted_dist(elbow_idx);
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::SPATIAL_TRIM, SpatialTrimEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_SPATIAL_TRIM_H