#ifndef GWMVT_METHODS_PCA_ROBUST_S_ESTIMATOR_H
#define GWMVT_METHODS_PCA_ROBUST_S_ESTIMATOR_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <cmath>
#include <algorithm>

namespace gwmvt {

// S-estimator for robust covariance estimation
class SEstimator : public RobustEstimator {
private:
    double b_;              // Breakdown point parameter
    double c_;              // Tuning constant for rho function
    double k0_;             // Expected value of rho under normal distribution
    int max_refinements_;   // Maximum refinement iterations
    
public:
    // Constructor
    explicit SEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), b_(0.5), c_(1.547), k0_(0.1995), max_refinements_(2) {
        
        // Check for custom breakdown point
        auto it = config.params.find("breakdown_point");
        if (it != config.params.end()) {
+            b_ = std::max(0.1, std::min(0.5, it->second));
            // Adjust tuning constant based on breakdown point
            update_tuning_constant();
        }
        
        // Check for custom tuning constant
        it = config.params.find("c_tuning");
        if (it != config.params.end()) {
            c_ = it->second;
        }
        
        // Check for max refinements
        it = config.params.find("max_refinements");
        if (it != config.params.end()) {
            max_refinements_ = static_cast<int>(it->second);
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Get initial estimate using fast method
        RobustStats initial_stats = compute_initial_estimate(data, weights);
        
        // Refine using S-estimation
        RobustStats s_stats = refine_s_estimate(data, weights, initial_stats);
        
        // Final reweighting for efficiency
        s_stats = final_reweighting(data, weights, s_stats);
        
        return s_stats;
    }
    
    // Iterative estimation
    RobustStats estimate_iterative(const Mat& data, const Vec& weights,
                                  ConvergenceInfo& conv_info) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Initialize with coordinate-wise median
        RobustStats stats(p);
        for (int j = 0; j < p; ++j) {
            stats.center(j) = weighted_median(data.col(j), weights);
        }
        
        Vec old_center = stats.center;
        double old_scale = 1.0;
        
        // Main iteration
        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // Center data
            Mat data_centered = data.each_row() - stats.center.t();
            
            // Compute scale estimate
            double scale = compute_s_scale(data_centered, weights, stats);
            
            // Update location and scatter
            stats = update_estimates(data, weights, stats, scale);
            
            // Check convergence
            double center_change = arma::norm(stats.center - old_center, 2);
            double scale_change = std::abs(scale - old_scale) / (old_scale + 1e-10);
            
            conv_info.error = std::max(center_change, scale_change);
            conv_info.iterations = iter + 1;
            
            if (center_change < config_.tolerance && scale_change < config_.tolerance) {
                conv_info.converged = true;
                conv_info.message = "S-estimator converged";
                break;
            }
            
            old_center = stats.center;
            old_scale = scale;
        }
        
        if (!conv_info.converged) {
            conv_info.message = "S-estimator reached maximum iterations";
            diagnostics_.add_warning("S-estimator did not converge within " + 
                                   std::to_string(config_.max_iterations) + " iterations");
        }
        
        return stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return "s_estimator"; 
    }
    
    double get_breakdown_point() const override { 
        return b_; 
    }
    
    double get_efficiency() const override { 
        // Efficiency depends on breakdown point
        if (b_ >= 0.5) return 0.28;  // 50% breakdown
        if (b_ >= 0.25) return 0.66;  // 25% breakdown
        return 0.95;  // Low breakdown, high efficiency
    }
    
private:
    // Tukey's biweight rho function
    double rho_biweight(double u) const {
        double u_abs = std::abs(u);
        if (u_abs <= c_) {
            double u_c = u / c_;
            double u_c2 = u_c * u_c;
            return (c_ * c_ / 6.0) * (1.0 - std::pow(1.0 - u_c2, 3.0));
        } else {
            return c_ * c_ / 6.0;
        }
    }
    
    // Derivative of rho (psi function)
    double psi_biweight(double u) const {
        double u_abs = std::abs(u);
        if (u_abs <= c_) {
            double u_c = u / c_;
            double u_c2 = u_c * u_c;
            return u * std::pow(1.0 - u_c2, 2.0);
        } else {
            return 0.0;
        }
    }
    
    // Weight function (psi(u)/u)
    double weight_biweight(double u) const {
        if (std::abs(u) < 1e-10) return 1.0;
        return psi_biweight(u) / u;
    }
    
    // Update tuning constant based on breakdown point
    void update_tuning_constant() {
        // Tuning constants for Tukey's biweight
        if (b_ >= 0.5) {
            c_ = 1.547;  // 50% breakdown
            k0_ = 0.1995;
        } else if (b_ >= 0.25) {
            c_ = 2.937;  // 25% breakdown
            k0_ = 0.4310;
        } else {
            c_ = 4.685;  // ~10% breakdown
            k0_ = 0.7317;
        }
    }
    
    // Compute initial estimate
    RobustStats compute_initial_estimate(const Mat& data, const Vec& weights) {
        int p = data.n_cols;
        RobustStats stats(p);
        
        // Use coordinate-wise median and MAD
        for (int j = 0; j < p; ++j) {
            stats.center(j) = weighted_median(data.col(j), weights);
            stats.scale(j) = weighted_mad(data.col(j), weights, stats.center(j));
        }
        
        // Initial covariance as scaled identity
        double median_scale = arma::median(stats.scale);
        stats.covariance = median_scale * median_scale * arma::eye(p, p);
        
        return stats;
    }
    
    // Compute S-scale
    double compute_s_scale(const Mat& data_centered, const Vec& weights,
                          const RobustStats& stats) {
        int n = data_centered.n_rows;
        
        // Compute Mahalanobis distances
        Mat cov_inv = safe_inv(stats.covariance);
        Vec distances(n);
        
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
        }
        
        // Find scale s such that average of rho equals k0
        double scale = 1.0;
        double scale_old = 0.0;
        
        // Scale iteration
        for (int iter = 0; iter < 50; ++iter) {
            double sum_rho = 0.0;
            double sum_weights = 0.0;
            
            for (int i = 0; i < n; ++i) {
                sum_rho += weights(i) * rho_biweight(distances(i) / scale);
                sum_weights += weights(i);
            }
            
            double avg_rho = sum_rho / sum_weights;
            
            // Update scale
            scale = scale * std::sqrt(avg_rho / k0_);
            
            // Check convergence
            if (std::abs(scale - scale_old) / scale < 1e-6) {
                break;
            }
            scale_old = scale;
        }
        
        return scale;
    }
    
    // Update location and scatter estimates
    RobustStats update_estimates(const Mat& data, const Vec& weights,
                                const RobustStats& current_stats, double scale) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        RobustStats new_stats(p);
        
        // Compute weights based on current estimates
        Mat data_centered = data.each_row() - current_stats.center.t();
        Mat cov_inv = safe_inv(current_stats.covariance);
        
        Vec w_vec(n);
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            double d = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
            w_vec(i) = weight_biweight(d / scale) * weights(i);
        }
        
        // Normalize weights
        w_vec /= arma::sum(w_vec);
        
        // Update center
        new_stats.center = weighted_mean(data, w_vec);
        
        // Update covariance
        data_centered = data.each_row() - new_stats.center.t();
        new_stats.covariance = arma::zeros<Mat>(p, p);
        
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            new_stats.covariance += w_vec(i) * (x * x.t());
        }
        
        // Scale covariance to maintain consistency
        new_stats.covariance *= scale * scale / k0_;
        
        // Regularize
        regularize_covariance(new_stats.covariance);
        new_stats.scale = arma::sqrt(new_stats.covariance.diag());
        
        return new_stats;
    }
    
    // Refine S-estimate
    RobustStats refine_s_estimate(const Mat& data, const Vec& weights,
                                 const RobustStats& initial_stats) {
        RobustStats current_stats = initial_stats;
        
        for (int ref = 0; ref < max_refinements_; ++ref) {
+            // Center data
            Mat data_centered = data.each_row() - current_stats.center.t();
            
            // Compute S-scale
            double scale = compute_s_scale(data_centered, weights, current_stats);
            
            // Update estimates
            current_stats = update_estimates(data, weights, current_stats, scale);
            
            // Check validity
            if (!validate_covariance(current_stats.covariance)) {
                diagnostics_.add_warning("S-estimator: Invalid covariance in refinement " + 
                                       std::to_string(ref));
                break;
            }
        }
        
        return current_stats;
    }
    
    // Final reweighting for efficiency
    RobustStats final_reweighting(const Mat& data, const Vec& weights,
                                 const RobustStats& s_stats) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Compute final distances
        Mat data_centered = data.each_row() - s_stats.center.t();
        Mat cov_inv = safe_inv(s_stats.covariance);
        
        Vec distances(n);
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
        }
        
        // Use hard rejection for efficiency
        double chi2_cutoff = std::sqrt(R::qchisq(0.975, p, true, false));
        
        Vec final_weights = weights;
        for (int i = 0; i < n; ++i) {
            if (distances(i) > chi2_cutoff) {
                final_weights(i) *= 0.01;
            }
        }
        final_weights /= arma::sum(final_weights);
        
        // Compute final estimates
        RobustStats final_stats(p);
        final_stats.center = weighted_mean(data, final_weights);
        final_stats.covariance = weighted_covariance(data, final_weights, final_stats.center);
        
        // Apply consistency factor
        double consistency_factor = 1.0 / (1.0 - 2.0 * R::pnorm(-chi2_cutoff, 0.0, 1.0, true, false));
        final_stats.covariance *= consistency_factor;
        
        // Regularize
        regularize_covariance(final_stats.covariance);
        final_stats.scale = arma::sqrt(final_stats.covariance.diag());
        final_stats.weights = final_weights;
        
        // Identify outliers
        final_stats.outliers = arma::zeros<UVec>(n);
        for (int i = 0; i < n; ++i) {
            if (distances(i) > chi2_cutoff) {
                final_stats.outliers(i) = 1;
            }
        }
        
        return final_stats;
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::S_ESTIMATOR, SEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_S_ESTIMATOR_H