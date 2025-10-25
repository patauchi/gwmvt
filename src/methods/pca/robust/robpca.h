#ifndef GWMVT_METHODS_PCA_ROBUST_ROBPCA_H
#define GWMVT_METHODS_PCA_ROBUST_ROBPCA_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace gwmvt {

// ROBPCA (Robust PCA via projection pursuit) estimator
class ROBPCAEstimator : public RobustEstimator {
private:
    int k_max_;              // Maximum number of components
    double alpha_;           // Robustness parameter (0.5 = 50% breakdown)
    int n_directions_;       // Number of directions for outlyingness
    bool use_mad_;           // Use MAD instead of Qn scale estimator
    double convergence_tol_; // Convergence tolerance for PP
    
public:
    // Constructor
    explicit ROBPCAEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), k_max_(10), alpha_(0.5), 
          n_directions_(250), use_mad_(true), convergence_tol_(1e-4) {
        
        // Check for k_max
        auto it = config.params.find("k_max");
        if (it != config.params.end()) {
            k_max_ = std::max(1, static_cast<int>(it->second));
        }
        
        // Check for alpha
        it = config.params.find("alpha");
        if (it != config.params.end()) {
            alpha_ = std::max(0.5, std::min(0.9, it->second));
        }
        
        // Check for number of directions
        it = config.params.find("n_directions");
        if (it != config.params.end()) {
            n_directions_ = std::max(100, static_cast<int>(it->second));
        }
        
        // Check for MAD flag
        it = config.params.find("use_mad");
        if (it != config.params.end()) {
            use_mad_ = (it->second > 0.5);
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Step 1: Dimension reduction if p > 10
        Mat data_reduced;
        Mat transform_matrix;
        Vec center_init;
        
        if (p > 10) {
            auto reduction_result = reduce_dimension(data, weights);
            data_reduced = std::get<0>(reduction_result);
            transform_matrix = std::get<1>(reduction_result);
            center_init = std::get<2>(reduction_result);
        } else {
            data_reduced = data;
            transform_matrix = arma::eye<Mat>(p, p);
            center_init = arma::zeros<Vec>(p);
        }
        
        // Step 2: Detect outliers in reduced space
        UVec outlier_flags = detect_outliers_pp(data_reduced, weights);
        
        // Step 3: Compute robust PCA on clean data
        Vec robust_weights = weights;
        for (int i = 0; i < n; ++i) {
            if (outlier_flags(i) == 1) {
                robust_weights(i) *= 0.1;
            }
        }
        robust_weights /= arma::sum(robust_weights);
        
        // Final robust estimates
        RobustStats stats(p);
        
        if (p > 10) {
            // Transform back to original space
            Mat data_clean = data.rows(arma::find(outlier_flags == 0));
            Vec weights_clean = robust_weights(arma::find(outlier_flags == 0));
            if (weights_clean.n_elem > 0) {
                double clean_sum = arma::sum(weights_clean);
                if (clean_sum > 0.0) {
                    weights_clean /= clean_sum;
                } else {
                    weights_clean.fill(1.0 / weights_clean.n_elem);
                }
            }
            
            stats.center = weighted_mean(data, robust_weights);
            stats.covariance = weighted_covariance(data, robust_weights, stats.center);
        } else {
            stats.center = weighted_mean(data, robust_weights);
            stats.covariance = weighted_covariance(data, robust_weights, stats.center);
        }
        
        // Regularize covariance
        regularize_covariance(stats.covariance);
        stats.scale = arma::sqrt(stats.covariance.diag());
        stats.weights = robust_weights;
        stats.outliers = outlier_flags;
        
        return stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return "robpca"; 
    }
    
    double get_breakdown_point() const override { 
        return alpha_;
    }
    
    double get_efficiency() const override { 
        // ROBPCA efficiency depends on dimension and contamination
        return 0.6;  // Conservative estimate
    }
    
private:
    // Dimension reduction for high-dimensional data
    std::tuple<Mat, Mat, Vec> reduce_dimension(const Mat& data, const Vec& weights) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        if (n == 0 || p == 0) {
            return std::make_tuple(Mat(n, 0, arma::fill::zeros),
                                   Mat(p, 0, arma::fill::zeros),
                                   Vec(p, arma::fill::zeros));
        }
        
        // Use robust SVD on subset
        double weight_median = arma::median(weights);
        UVec high_weight_idx = arma::find(weights > weight_median);
        
        if (high_weight_idx.n_elem < static_cast<arma::uword>(p + 1)) {
            UVec sorted_idx = arma::sort_index(weights, "descend");
            arma::uword take = std::min(
                static_cast<arma::uword>(p + 1),
                sorted_idx.n_elem
            );
            high_weight_idx = sorted_idx.head(take);
        }
        
        Mat data_subset = data.rows(high_weight_idx);
        Vec weights_subset = weights(high_weight_idx);
        double weight_sum = arma::sum(weights_subset);
        if (weight_sum <= 0.0) {
            weights_subset.fill(1.0 / weights_subset.n_elem);
        } else {
            weights_subset /= weight_sum;
        }
        
        // Center using spatial median (more robust than mean)
        Vec center = compute_spatial_median(data_subset, weights_subset);
        Mat data_centered_subset = data_subset.each_row() - center.t();
        
        // Weighted SVD
        for (arma::uword i = 0; i < data_subset.n_rows; ++i) {
            double w = std::sqrt(std::max(weights_subset(i), 0.0));
            data_centered_subset.row(i) *= w;
        }
        
        Mat U, V;
        Vec s;
        bool svd_success = arma::svd_econ(U, s, V, data_centered_subset, "right");
        
        Mat data_centered_full = data.each_row() - center.t();
        
        if (!svd_success || s.n_elem == 0) {
            Mat identity = arma::eye<Mat>(p, p);
            return std::make_tuple(data_centered_full, identity, center);
        }
        
        Vec s2 = s % s;
        double total_var = arma::sum(s2);
        
        int k_keep = 1;
        if (total_var > 0) {
            Vec cum_var = arma::cumsum(s2) / total_var;
            for (int i = 0; i < static_cast<int>(cum_var.n_elem); ++i) {
                if (cum_var(i) >= 0.99) {
                    k_keep = i + 1;
                    break;
                }
            }
        }
        
        k_keep = std::max(1, std::min({k_keep,
                                       k_max_,
                                       static_cast<int>(V.n_cols)}));
        
        Mat transform = V.cols(0, k_keep - 1);
        Mat data_reduced = data_centered_full * transform;
        
        return std::make_tuple(data_reduced, transform, center);
    }
    
    // Detect outliers using projection pursuit
    UVec detect_outliers_pp(const Mat& data, const Vec& weights) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        if (n == 0 || p == 0) {
            return UVec(n, arma::fill::zeros);
        }
        
        // Generate random directions
        arma::arma_rng::set_seed_random();
        double theoretical_max = (p < 30)
            ? std::pow(2.0, static_cast<double>(p)) - 1.0
            : std::pow(2.0, 30.0) - 1.0;
        int max_directions = static_cast<int>(std::max(1.0, theoretical_max));
        int actual_directions = std::min(n_directions_, max_directions);
        actual_directions = std::max(1, actual_directions);
        
        Mat directions(p, actual_directions, arma::fill::randn);
        for (int j = 0; j < actual_directions; ++j) {
            directions.col(j) /= arma::norm(directions.col(j), 2);
        }
        
        // Compute outlyingness for each observation
        Vec max_outlyingness(n, arma::fill::zeros);
        
        for (int j = 0; j < actual_directions; ++j) {
            Vec direction = directions.col(j);
            Vec projected = data * direction;
            
            // Robust location and scale
            UVec valid_idx = arma::find(weights > 0.01);
            if (valid_idx.is_empty()) {
                valid_idx = arma::regspace<UVec>(0, n - 1);
            }
            Vec valid_proj = projected(valid_idx);
            Vec valid_weights = weights(valid_idx);
            double valid_sum = arma::sum(valid_weights);
            if (valid_sum <= 0.0) {
                valid_weights.fill(1.0 / valid_weights.n_elem);
            } else {
                valid_weights /= valid_sum;
            }
            
            double location = weighted_median(valid_proj, valid_weights);
            double scale;
            
            if (use_mad_) {
                scale = weighted_mad(valid_proj, valid_weights, location);
            } else {
                scale = compute_qn_scale(valid_proj, valid_weights);
            }
            
            if (scale < 1e-10) continue;
            
            // Update outlyingness
            for (int i = 0; i < n; ++i) {
                double outlying = std::abs(projected(i) - location) / scale;
                max_outlyingness(i) = std::max(max_outlyingness(i), outlying);
            }
        }
        
        // Determine outliers based on outlyingness
        double cutoff = R::qnorm(0.975, 0.0, 1.0, true, false);
        if (p > 1) {
            // Adjust for multiple testing
            cutoff = R::qnorm(std::pow(0.975, 1.0/p), 0.0, 1.0, true, false);
        }
        
        UVec outlier_flags(n, arma::fill::zeros);
        for (int i = 0; i < n; ++i) {
            if (max_outlyingness(i) > cutoff) {
                outlier_flags(i) = 1;
            }
        }
        
        return outlier_flags;
    }
    
    // Compute spatial median (L1 median)
    Vec compute_spatial_median(const Mat& data, const Vec& weights) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Initialize with coordinate-wise median
        Vec median(p);
        for (int j = 0; j < p; ++j) {
            median(j) = weighted_median(data.col(j), weights);
        }
        
        // Weiszfeld's algorithm for spatial median
        int max_iter = 100;
        double tol = 1e-6;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            Vec new_median(p, arma::fill::zeros);
            double weight_sum = 0.0;
            
            for (int i = 0; i < n; ++i) {
                Vec diff = data.row(i).t() - median;
                double dist = arma::norm(diff, 2);
                
                if (dist > 1e-10) {
                    double w = weights(i) / dist;
                    new_median += w * data.row(i).t();
                    weight_sum += w;
                } else {
                    // Point coincides with current median
                    new_median += weights(i) * median;
                    weight_sum += weights(i);
                }
            }
            
            new_median /= weight_sum;
            
            // Check convergence
            if (arma::norm(new_median - median, 2) < tol) {
                break;
            }
            
            median = new_median;
        }
        
        return median;
    }
    
    // Qn scale estimator (more efficient than MAD for normal data)
    double compute_qn_scale(const Vec& x, const Vec& weights) {
        int n = x.n_elem;
        
        // Compute pairwise differences
        std::vector<double> diffs;
        std::vector<double> diff_weights;
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                diffs.push_back(std::abs(x(i) - x(j)));
                diff_weights.push_back(weights(i) * weights(j));
            }
        }
        
        // Convert to arma vectors
        Vec diff_vec = arma::conv_to<Vec>::from(diffs);
        Vec weight_vec = arma::conv_to<Vec>::from(diff_weights);
        weight_vec /= arma::sum(weight_vec);
        
        // First quartile of pairwise differences
        double q1 = weighted_quantile(diff_vec, weight_vec, 0.25);
        
        // Consistency factor for normal distribution
        double cn = 2.2191;  // For first quartile
        
        return cn * q1;
    }
    
    // Iterative refinement
    RobustStats estimate_iterative(const Mat& data, const Vec& weights,
                                  ConvergenceInfo& conv_info) override {
        // ROBPCA is not naturally iterative, but we can refine the estimate
        RobustStats stats = estimate(data, weights);
        
        // Refine by reweighting based on robust distances
        Mat data_centered = data.each_row() - stats.center.t();
        Mat cov_inv = safe_inv(stats.covariance);
        
        Vec distances(data.n_rows);
        for (int i = 0; i < data.n_rows; ++i) {
            Vec x = data_centered.row(i).t();
            distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
        }
        
        // Reweight based on distances
        double chi2_cutoff = std::sqrt(R::qchisq(0.975, data.n_cols, true, false));
        Vec refined_weights = weights;
        
        for (int i = 0; i < data.n_rows; ++i) {
            if (distances(i) > chi2_cutoff) {
                refined_weights(i) *= 0.01;
            }
        }
        refined_weights /= arma::sum(refined_weights);
        
        // Final estimate with refined weights
        stats.center = weighted_mean(data, refined_weights);
        stats.covariance = weighted_covariance(data, refined_weights, stats.center);
        regularize_covariance(stats.covariance);
        stats.scale = arma::sqrt(stats.covariance.diag());
        stats.weights = refined_weights;
        
        conv_info.converged = true;
        conv_info.iterations = 1;
        conv_info.message = "ROBPCA completed";
        
        return stats;
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::ROBPCA, ROBPCAEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_ROBPCA_H
