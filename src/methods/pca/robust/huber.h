#ifndef GWMVT_METHODS_PCA_ROBUST_HUBER_H
#define GWMVT_METHODS_PCA_ROBUST_HUBER_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <cmath>

namespace gwmvt {

// Adaptive Huber M-estimator
class HuberEstimator : public RobustEstimator {
private:
    double k_constant_;  // Huber constant
    bool adaptive_;      // Use adaptive threshold
    
public:
    // Constructor
    explicit HuberEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), k_constant_(1.345), adaptive_(true) {
        
        // Check for custom k constant
        auto it = config.params.find("k_constant");
        if (it != config.params.end()) {
            k_constant_ = it->second;
        }
        
        // Check for adaptive flag
        it = config.params.find("adaptive");
        if (it != config.params.end()) {
            adaptive_ = (it->second > 0.5);
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        RobustStats stats(p);
        
        // Compute robust center and scale for each variable
        for (int j = 0; j < p; ++j) {
            compute_univariate_huber(data.col(j), weights, 
                                   stats.center(j), stats.scale(j));
        }
        
        // Center data
        Mat data_centered = data.each_row() - stats.center.t();
        
        // Compute Huber weights matrix
        Mat huber_weights(n, p, arma::fill::ones);
        
        for (int j = 0; j < p; ++j) {
            if (stats.scale(j) > 1e-10) {
                double k = k_constant_;
                
                if (adaptive_) {
                    // Adaptive threshold based on local distribution
                    Vec local_values = arma::abs(data_centered.col(j)) / stats.scale(j);
                    UVec valid_idx = arma::find(weights > 0.01);
                    
                    if (valid_idx.n_elem > 5) {
                        Vec valid_values = local_values(valid_idx);
                        k = std::max(2.0, arma::as_scalar(arma::quantile(valid_values, 0.95)));
                    } else {
                        k = 2.0;
                    }
                }
                
                // Apply Huber weights
                for (int i = 0; i < n; ++i) {
                    double u = std::abs(data_centered(i, j)) / stats.scale(j);
                    if (u > k) {
                        huber_weights(i, j) = k / u;
                    }
                }
            }
        }
        
        // Combine weights
        Vec combined_weights(n);
        for (int i = 0; i < n; ++i) {
            // Combined weight: spatial * product of Huber weights
            combined_weights(i) = weights(i) * arma::prod(huber_weights.row(i));
            // Avoid collapse: maintain minimum weight
            combined_weights(i) = std::max(combined_weights(i), weights(i) * 0.1);
        }
        
        // Normalize weights
        combined_weights /= arma::sum(combined_weights);
        
        // Compute weighted covariance
        stats.covariance = weighted_covariance(data, combined_weights, stats.center);
        
        // Regularize for numerical stability
        regularize_covariance(stats.covariance);
        
        // Store final weights
        stats.weights = combined_weights;
        
        // Identify outliers
        stats.outliers = arma::zeros<UVec>(n);
        for (int i = 0; i < n; ++i) {
            if (arma::mean(huber_weights.row(i)) < 0.5) {
                stats.outliers(i) = 1;
            }
        }
        
        return stats;
    }
    
    // Iterative estimation
    RobustStats estimate_iterative(const Mat& data, const Vec& weights,
                                  ConvergenceInfo& conv_info) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        RobustStats stats(p);
        stats.center = arma::mean(data.each_row() % weights.t(), 0).t();
        
        Vec old_center = stats.center;
        
        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // M-step: compute weights
            Mat data_centered = data.each_row() - stats.center.t();
            Vec scale(p);
            
            for (int j = 0; j < p; ++j) {
                scale(j) = weighted_mad(data.col(j), weights, stats.center(j));
            }
            
            // Compute Huber weights
            Mat huber_weights(n, p, arma::fill::ones);
            
            for (int j = 0; j < p; ++j) {
                if (scale(j) > 1e-10) {
                    for (int i = 0; i < n; ++i) {
                        double u = std::abs(data_centered(i, j)) / scale(j);
                        if (u > k_constant_) {
                            huber_weights(i, j) = k_constant_ / u;
                        }
                    }
                }
            }
            
            // Update center
            Vec combined_weights(n);
            for (int i = 0; i < n; ++i) {
                combined_weights(i) = weights(i) * arma::prod(huber_weights.row(i));
            }
            combined_weights /= arma::sum(combined_weights);
            
            stats.center = data.t() * combined_weights;
            
            // Check convergence
            if (check_convergence(old_center, stats.center, iter + 1, conv_info)) {
                break;
            }
            
            old_center = stats.center;
        }
        
        // Final covariance computation
        stats = estimate(data, weights);
        conv_info.converged = true;
        
        return stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return adaptive_ ? "adaptive_huber" : "huber"; 
    }
    
    double get_breakdown_point() const override { 
        return 0.5 * (1.0 - 1.0 / std::sqrt(1.0 + k_constant_ * k_constant_)); 
    }
    
    double get_efficiency() const override { 
        return 0.95; // Approximate for k=1.345
    }
    
private:
    // Compute univariate Huber estimates
    void compute_univariate_huber(const Vec& x, const Vec& weights,
                                 double& center, double& scale) {
        // Initial robust estimates
        center = weighted_median(x, weights);
        scale = weighted_mad(x, weights, center);
        
        if (scale < 1e-10) {
            // Constant variable
            return;
        }
        
        // Iterative refinement
        int max_iter = 20;
        double tol = 1e-6;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            double old_center = center;
            
            // Compute Huber weights
            Vec huber_w(x.n_elem);
            for (size_t i = 0; i < x.n_elem; ++i) {
                double u = std::abs(x(i) - center) / scale;
                huber_w(i) = (u <= k_constant_) ? 1.0 : k_constant_ / u;
            }
            
            // Update center
            Vec combined_w = weights % huber_w;
            combined_w /= arma::sum(combined_w);
            center = arma::dot(x, combined_w);
            
            // Check convergence
            if (std::abs(center - old_center) < tol * scale) {
                break;
            }
        }
        
        // Update scale
        scale = weighted_mad(x, weights, center);
    }
    
    // Huber psi function
    double psi_huber(double u) const {
        if (std::abs(u) <= k_constant_) {
            return u;
        } else {
            return k_constant_ * (u > 0 ? 1.0 : -1.0);
        }
    }
    
    // Huber weight function
    double weight_huber(double u) const {
        if (std::abs(u) <= k_constant_) {
            return 1.0;
        } else {
            return k_constant_ / std::abs(u);
        }
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::ADAPTIVE_HUBER, HuberEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_HUBER_H