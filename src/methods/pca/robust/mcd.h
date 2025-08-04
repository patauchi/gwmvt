#ifndef GWMVT_METHODS_PCA_ROBUST_MCD_H
#define GWMVT_METHODS_PCA_ROBUST_MCD_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <algorithm>
#include <random>
#include <set>

namespace gwmvt {

// Adaptive Minimum Covariance Determinant (MCD) estimator
class MCDEstimator : public RobustEstimator {
private:
    double h_fraction_;  // Fraction of observations to use
    int n_trials_;       // Number of random subsets to try
    bool adaptive_;      // Use adaptive h selection
    
public:
    // Constructor
    explicit MCDEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), h_fraction_(0.75), n_trials_(500), adaptive_(true) {
        
        // Check for custom h fraction
        auto it = config.params.find("h_fraction");
        if (it != config.params.end()) {
            h_fraction_ = std::max(0.5, std::min(1.0, it->second));
        }
        
        // Check for number of trials
        it = config.params.find("n_trials");
        if (it != config.params.end()) {
            n_trials_ = static_cast<int>(it->second);
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
        
        // Determine h (number of observations to use)
        Vec weights_sorted = arma::sort(weights, "descend");
        Vec cum_weight = arma::cumsum(weights_sorted);
        double target_weight = h_fraction_ * arma::sum(weights);
        
        int h = p + 1;  // Minimum for non-singular covariance
        for (size_t i = 0; i < cum_weight.n_elem; ++i) {
            if (cum_weight(i) >= target_weight) {
                h = i + 1;
                break;
            }
        }
        h = std::min(h, n);
        
        // If adaptive, adjust h based on data characteristics
        if (adaptive_) {
            // Estimate contamination level using robust scale
            Vec robust_scales(p);
            for (int j = 0; j < p; ++j) {
                robust_scales(j) = weighted_mad(data.col(j), weights, 
                                              weighted_median(data.col(j), weights));
            }
            
            // If scales vary a lot, use more conservative h
            double scale_ratio = robust_scales.max() / robust_scales.min();
            if (scale_ratio > 10.0) {
                h = std::max(h, static_cast<int>(0.8 * n));
            }
        }
        
        // Find best subset
        RobustStats best_stats(p);
        double min_det = arma::datum::inf;
        
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Try multiple random subsets
        int actual_trials = std::min(n_trials_, choose_approx(n, h));
        
        for (int trial = 0; trial < actual_trials; ++trial) {
            // Select random h-subset with probability proportional to weights
            UVec subset_idx = weighted_random_subset(n, h, weights, gen);
            
            // Compute statistics for this subset
            Mat data_subset = data.rows(subset_idx);
            Vec weights_subset = weights(subset_idx);
            weights_subset /= arma::sum(weights_subset);
            
            Vec center = weighted_mean(data_subset, weights_subset);
            Mat cov = weighted_covariance(data_subset, weights_subset, center);
            
            // Check if valid
            if (!validate_covariance(cov)) {
                continue;
            }
            
            // Compute determinant
            double det = arma::det(cov);
            
            // Update best if this has smaller determinant
            if (det > 0 && det < min_det) {
                min_det = det;
                best_stats.center = center;
                best_stats.covariance = cov;
            }
        }
        
        // If no valid subset found, fall back to weighted mean/covariance
        if (min_det == arma::datum::inf) {
            best_stats.center = weighted_mean(data, weights);
            best_stats.covariance = weighted_covariance(data, weights, best_stats.center);
            diagnostics_.add_warning("MCD: No valid subset found, using standard estimates");
        }
        
        // Regularize covariance
        regularize_covariance(best_stats.covariance);
        
        // Compute scale for each variable
        best_stats.scale = arma::sqrt(best_stats.covariance.diag());
        
        // Identify outliers using Mahalanobis distances
        Mat cov_inv = safe_inv(best_stats.covariance);
        Vec mahal_dist = mahalanobis_distance(data, best_stats.center, cov_inv);
        
        double chi2_cutoff = R::qchisq(0.975, p, true, false);
        best_stats.outliers = arma::zeros<UVec>(n);
        for (int i = 0; i < n; ++i) {
            if (mahal_dist(i) * mahal_dist(i) > chi2_cutoff) {
                best_stats.outliers(i) = 1;
            }
        }
        
        // Reweight step for final estimates
        best_stats = reweight_step(data, weights, best_stats);
        
        return best_stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return adaptive_ ? "adaptive_mcd" : "mcd"; 
    }
    
    double get_breakdown_point() const override { 
        return 1.0 - h_fraction_; 
    }
    
    double get_efficiency() const override { 
        // MCD efficiency depends on h
        if (h_fraction_ >= 0.9) return 0.95;
        if (h_fraction_ >= 0.8) return 0.85;
        if (h_fraction_ >= 0.7) return 0.75;
        return 0.65;
    }
    
private:
    // Weighted random subset selection
    UVec weighted_random_subset(int n, int h, const Vec& weights, std::mt19937& gen) {
        // Use weighted sampling without replacement
        std::vector<double> probs(weights.begin(), weights.end());
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        
        std::set<int> selected;
        while (selected.size() < static_cast<size_t>(h)) {
            int idx = dist(gen);
            selected.insert(idx);
        }
        
        UVec subset_idx(h);
        int i = 0;
        for (int idx : selected) {
            subset_idx(i++) = idx;
        }
        
        return subset_idx;
    }
    
    // Approximate n choose k for large values
    int choose_approx(int n, int k) {
        if (k > n - k) k = n - k;
        
        double result = 1.0;
        for (int i = 0; i < k; ++i) {
            result *= (n - i);
            result /= (i + 1);
            if (result > 1e6) break;  // Limit number of trials
        }
        
        return static_cast<int>(std::min(result, 1e6));
    }
    
    // Reweighting step for final estimates
    RobustStats reweight_step(const Mat& data, const Vec& weights, 
                             const RobustStats& initial_stats) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Compute Mahalanobis distances using initial estimates
        Mat cov_inv = safe_inv(initial_stats.covariance);
        Vec mahal_dist = mahalanobis_distance(data, initial_stats.center, cov_inv);
        
        // Use chi-square cutoff for reweighting
        double chi2_cutoff = R::qchisq(0.975, p, true, false);
        
        // Create new weights
        Vec new_weights = weights;
        for (int i = 0; i < n; ++i) {
            if (mahal_dist(i) * mahal_dist(i) > chi2_cutoff) {
                new_weights(i) *= 0.01;  // Downweight outliers
            }
        }
        new_weights /= arma::sum(new_weights);
        
        // Compute final estimates with new weights
        RobustStats final_stats(p);
        final_stats.center = weighted_mean(data, new_weights);
        final_stats.covariance = weighted_covariance(data, new_weights, final_stats.center);
        
        // Regularize
        regularize_covariance(final_stats.covariance);
        
        final_stats.scale = arma::sqrt(final_stats.covariance.diag());
        final_stats.weights = new_weights;
        final_stats.outliers = initial_stats.outliers;
        
        return final_stats;
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::ADAPTIVE_MCD, MCDEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_MCD_H