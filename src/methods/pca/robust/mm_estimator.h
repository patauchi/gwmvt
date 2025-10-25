#ifndef GWMVT_METHODS_PCA_ROBUST_MM_ESTIMATOR_H
#define GWMVT_METHODS_PCA_ROBUST_MM_ESTIMATOR_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <cmath>
#include <algorithm>

namespace gwmvt {

// MM-estimator combining high breakdown point with high efficiency
class MMEstimator : public RobustEstimator {
private:
    double efficiency_;       // Target efficiency (default 0.95)
    double breakdown_;        // Breakdown point for initial S-estimator
    double c_efficiency_;     // Tuning constant for efficiency
    double c_breakdown_;      // Tuning constant for breakdown
    double k0_eff_;          // Expected value for efficiency
    double k0_bdp_;          // Expected value for breakdown
    
public:
    // Constructor
    explicit MMEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), efficiency_(0.95), breakdown_(0.5),
          c_efficiency_(4.685), c_breakdown_(1.547), 
          k0_eff_(0.7317), k0_bdp_(0.1995) {
        
        // Check for custom efficiency
        auto it = config.params.find("efficiency");
        if (it != config.params.end()) {
            efficiency_ = std::max(0.8, std::min(0.99, it->second));
            update_efficiency_constant();
        }
        
        // Check for custom breakdown point
        it = config.params.find("breakdown_point");
        if (it != config.params.end()) {
            breakdown_ = std::max(0.25, std::min(0.5, it->second));
            update_breakdown_constant();
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Step 1: Get initial S-estimator with high breakdown
        RobustStats s_stats = compute_s_estimator(data, weights);
        
        // Step 2: M-step for efficiency
        RobustStats mm_stats = m_step_refinement(data, weights, s_stats);
        
        // Step 3: Final reweighting
        mm_stats = final_reweighting(data, weights, mm_stats);
        
        return mm_stats;
    }
    
    // Iterative estimation
    RobustStats estimate_iterative(const Mat& data, const Vec& weights,
                                  ConvergenceInfo& conv_info) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Get S-estimator
        RobustStats s_stats = compute_s_estimator(data, weights);
        
        // M-step iterations
        RobustStats mm_stats = s_stats;
        Vec old_center = mm_stats.center;
        
        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // Update using M-estimation
            mm_stats = m_step_update(data, weights, mm_stats);
            
            // Check convergence
            if (check_convergence(old_center, mm_stats.center, iter + 1, conv_info)) {
                break;
            }
            
            old_center = mm_stats.center;
        }
        
        // Final reweighting
        mm_stats = final_reweighting(data, weights, mm_stats);
        
        return mm_stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return "mm_estimator"; 
    }
    
    double get_breakdown_point() const override { 
        return breakdown_; 
    }
    
    double get_efficiency() const override { 
        return efficiency_; 
    }
    
private:
    // Tukey's biweight rho function
    double rho_biweight(double u, double c) const {
        double u_abs = std::abs(u);
        if (u_abs <= c) {
            double u_c = u / c;
            double u_c2 = u_c * u_c;
            return (c * c / 6.0) * (1.0 - std::pow(1.0 - u_c2, 3.0));
        } else {
            return c * c / 6.0;
        }
    }
    
    // Derivative of rho (psi function)
    double psi_biweight(double u, double c) const {
        double u_abs = std::abs(u);
        if (u_abs <= c) {
            double u_c = u / c;
            double u_c2 = u_c * u_c;
            return u * std::pow(1.0 - u_c2, 2.0);
        } else {
            return 0.0;
        }
    }
    
    // Weight function
    double weight_biweight(double u, double c) const {
        if (std::abs(u) < 1e-10) return 1.0;
        return psi_biweight(u, c) / u;
    }
    
    // Update tuning constants for efficiency
    void update_efficiency_constant() {
        if (efficiency_ >= 0.95) {
            c_efficiency_ = 4.685;
            k0_eff_ = 0.7317;
        } else if (efficiency_ >= 0.90) {
            c_efficiency_ = 3.882;
            k0_eff_ = 0.6278;
        } else if (efficiency_ >= 0.85) {
            c_efficiency_ = 3.444;
            k0_eff_ = 0.5332;
        } else {
            c_efficiency_ = 3.0;
            k0_eff_ = 0.4496;
        }
    }
    
    // Update tuning constants for breakdown
    void update_breakdown_constant() {
        if (breakdown_ >= 0.5) {
            c_breakdown_ = 1.547;
            k0_bdp_ = 0.1995;
        } else if (breakdown_ >= 0.4) {
            c_breakdown_ = 1.988;
            k0_bdp_ = 0.2823;
        } else if (breakdown_ >= 0.3) {
            c_breakdown_ = 2.382;
            k0_bdp_ = 0.3593;
        } else {
            c_breakdown_ = 2.937;
            k0_bdp_ = 0.4310;
        }
    }
    
    // Compute initial S-estimator
    RobustStats compute_s_estimator(const Mat& data, const Vec& weights) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Initialize with coordinate-wise robust estimates
        RobustStats s_stats(p);
        for (int j = 0; j < p; ++j) {
            s_stats.center(j) = weighted_median(data.col(j), weights);
            s_stats.scale(j) = weighted_mad(data.col(j), weights, s_stats.center(j));
            if (!std::isfinite(s_stats.scale(j)) || s_stats.scale(j) < 1e-6) {
                s_stats.scale(j) = 1e-6;
            }
        }
        
        // Initial scatter as diagonal
        s_stats.covariance = arma::diagmat(s_stats.scale % s_stats.scale);
        s_stats.covariance = arma::symmatu(s_stats.covariance);
        
        // S-estimation iterations
        double scale = 1.0;
        
        for (int iter = 0; iter < 20; ++iter) {
            // Center data
            Mat data_centered = data.each_row() - s_stats.center.t();
            
            // Compute Mahalanobis distances
            Mat cov_inv = safe_inv(s_stats.covariance);
            Vec distances(n);
            
            for (int i = 0; i < n; ++i) {
                Vec x = data_centered.row(i).t();
                distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
            }
            
            // Update scale
            scale = compute_s_scale(distances, weights);
            
            // Update estimates
            Vec w_vec(n);
            for (int i = 0; i < n; ++i) {
                w_vec(i) = weight_biweight(distances(i) / scale, c_breakdown_) * weights(i);
            }
            double w_sum = arma::sum(w_vec);
            if (w_sum <= 1e-12 || !std::isfinite(w_sum)) {
                w_vec.fill(1.0 / static_cast<double>(n));
            } else {
                w_vec /= w_sum;
            }
            
            // Update center and covariance
            Vec new_center = weighted_mean(data, w_vec);
            
            // Check convergence
            if (arma::norm(new_center - s_stats.center, 2) < 1e-6) {
                break;
            }
            
            s_stats.center = new_center;
            s_stats.covariance = weighted_covariance(data, w_vec, s_stats.center);
            s_stats.covariance = arma::symmatu(s_stats.covariance);
            
            // Apply S-estimator consistency factor
            s_stats.covariance *= scale * scale / k0_bdp_;
            
            // Regularize
            regularize_covariance(s_stats.covariance);
        }
        
        s_stats.scale = arma::sqrt(s_stats.covariance.diag());
        return s_stats;
    }
    
    // Compute S-scale
    double compute_s_scale(const Vec& distances, const Vec& weights) {
        double scale = arma::median(distances);
        if (!std::isfinite(scale) || scale < 1e-6) {
            scale = 1e-6;
        }
        
        // Scale iteration
        for (int iter = 0; iter < 30; ++iter) {
            double sum_rho = 0.0;
            double sum_weights = 0.0;
            
            for (size_t i = 0; i < distances.n_elem; ++i) {
                sum_rho += weights(i) * rho_biweight(distances(i) / scale, c_breakdown_);
                sum_weights += weights(i);
            }
            
            double avg_rho = sum_rho / sum_weights;
            double ratio = avg_rho / k0_bdp_;
            ratio = std::max(ratio, 1e-8);
            double scale_new = scale * std::sqrt(ratio);
            
            if (std::abs(scale_new - scale) / scale < 1e-6) {
                break;
            }
            scale = scale_new;
        }
        
        if (!std::isfinite(scale) || scale < 1e-6) {
            scale = 1e-6;
        }
        
        return scale;
    }
    
    // M-step refinement for efficiency
    RobustStats m_step_refinement(const Mat& data, const Vec& weights,
                                 const RobustStats& s_stats) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Use S-estimator scale
        Mat data_centered = data.each_row() - s_stats.center.t();
        Mat cov_inv = safe_inv(s_stats.covariance);
        
        Vec distances(n);
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
        }

        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(distances(i))) {
                distances(i) = 0.0;
            }
        }

        double s_scale = compute_s_scale(distances, weights);
        
        // M-step with efficiency tuning constant
        RobustStats m_stats = s_stats;
        
        for (int iter = 0; iter < 10; ++iter) {
            // Compute M-weights
            Vec m_weights(n);
            for (int i = 0; i < n; ++i) {
                m_weights(i) = weight_biweight(distances(i) / s_scale, c_efficiency_) * weights(i);
            }
            double m_sum = arma::sum(m_weights);
            if (m_sum <= 1e-12 || !std::isfinite(m_sum)) {
                m_weights.fill(1.0 / static_cast<double>(n));
            } else {
                m_weights /= m_sum;
            }
            
            // Update center
            Vec new_center = weighted_mean(data, m_weights);
            
            // Check convergence
            if (arma::norm(new_center - m_stats.center, 2) < 1e-6) {
                break;
            }
            
            m_stats.center = new_center;
            
            // Update covariance
            data_centered = data.each_row() - m_stats.center.t();
            m_stats.covariance = arma::zeros<Mat>(p, p);
            
            for (int i = 0; i < n; ++i) {
                Vec x = data_centered.row(i).t();
                double d = distances(i) / s_scale;
                double w = psi_biweight(d, c_efficiency_) / d;
                if (std::abs(d) < 1e-10) w = 1.0;
                m_stats.covariance += m_weights(i) * w * (x * x.t());
            }
            m_stats.covariance = arma::symmatu(m_stats.covariance);
            
            // Update distances for next iteration
            cov_inv = safe_inv(m_stats.covariance);
            for (int i = 0; i < n; ++i) {
                Vec x = data_centered.row(i).t();
                distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
                if (!std::isfinite(distances(i))) {
                    distances(i) = 0.0;
                }
            }
        }
        
        // Apply consistency correction
        double consistency_factor = k0_eff_ / (R::pchisq(c_efficiency_ * c_efficiency_, p, true, false));
        m_stats.covariance /= consistency_factor;
        
        // Regularize
        regularize_covariance(m_stats.covariance);
        m_stats.scale = arma::sqrt(m_stats.covariance.diag());
        
        return m_stats;
    }
    
    // M-step update (for iterative version)
    RobustStats m_step_update(const Mat& data, const Vec& weights,
                             const RobustStats& current_stats) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Center data
        Mat data_centered = data.each_row() - current_stats.center.t();
        Mat cov_inv = safe_inv(current_stats.covariance);
        
        // Compute weights based on efficiency constant
        Vec m_weights(n);
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            double d = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
            m_weights(i) = weight_biweight(d, c_efficiency_) * weights(i);
        }
        double m_sum = arma::sum(m_weights);
        if (m_sum <= 1e-12 || !std::isfinite(m_sum)) {
            m_weights.fill(1.0 / static_cast<double>(n));
        } else {
            m_weights /= m_sum;
        }
        
        // Update estimates
        RobustStats new_stats(p);
        new_stats.center = weighted_mean(data, m_weights);
        new_stats.covariance = weighted_covariance(data, m_weights, new_stats.center);
        new_stats.covariance = arma::symmatu(new_stats.covariance);
        
        // Regularize
        regularize_covariance(new_stats.covariance);
        new_stats.scale = arma::sqrt(new_stats.covariance.diag());
        
        return new_stats;
    }
    
    // Final reweighting
    RobustStats final_reweighting(const Mat& data, const Vec& weights,
                                 const RobustStats& mm_stats) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Compute final distances
        Mat data_centered = data.each_row() - mm_stats.center.t();
        Mat cov_inv = safe_inv(mm_stats.covariance);
        
        Vec distances(n);
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
            if (!std::isfinite(distances(i))) {
                distances(i) = 0.0;
            }
        }
        
        // Hard rejection using chi-square cutoff
        double chi2_cutoff = std::sqrt(R::qchisq(0.975, p, true, false));
        
        Vec final_weights = weights;
        for (int i = 0; i < n; ++i) {
            if (distances(i) > chi2_cutoff) {
                final_weights(i) *= 0.01;
            }
        }
        double final_sum = arma::sum(final_weights);
        if (final_sum <= 1e-12 || !std::isfinite(final_sum)) {
            final_weights.fill(1.0 / static_cast<double>(n));
        } else {
            final_weights /= final_sum;
        }
        
        // Final estimates
        RobustStats final_stats(p);
        final_stats.center = weighted_mean(data, final_weights);
        final_stats.covariance = weighted_covariance(data, final_weights, final_stats.center);
        final_stats.covariance = arma::symmatu(final_stats.covariance);
        
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
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::MM_ESTIMATOR, MMEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_MM_ESTIMATOR_H
