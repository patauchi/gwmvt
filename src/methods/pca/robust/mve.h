#ifndef GWMVT_METHODS_PCA_ROBUST_MVE_H
#define GWMVT_METHODS_PCA_ROBUST_MVE_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <algorithm>
#include <random>
#include <set>
#include <limits>

namespace gwmvt {

// Minimum Volume Ellipsoid (MVE) estimator
class MVEEstimator : public RobustEstimator {
private:
    double h_fraction_;      // Fraction of observations to use
    int n_trials_;          // Number of random subsets to try
    bool fast_mve_;         // Use fast MVE algorithm
    
public:
    // Constructor
    explicit MVEEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), h_fraction_(0.5), n_trials_(1000), fast_mve_(true) {
        
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
        
        // Check for fast MVE flag
        it = config.params.find("fast_mve");
        if (it != config.params.end()) {
            fast_mve_ = (it->second > 0.5);
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Determine h (number of observations to use)
        int h = static_cast<int>(std::ceil(h_fraction_ * n));
        h = std::max(h, p + 1);  // Minimum for non-singular covariance
        h = std::min(h, n);
        
        RobustStats best_stats(p);
        double min_volume = std::numeric_limits<double>::infinity();
        
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if (fast_mve_) {
            // Fast MVE using concentration steps
            best_stats = fast_mve_algorithm(data, weights, h, gen);
        } else {
            // Standard MVE with random sampling
            best_stats = standard_mve_algorithm(data, weights, h, gen);
        }
        
        // Reweighting step for final estimates
        best_stats = reweight_step(data, weights, best_stats);
        
        return best_stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return "mve"; 
    }
    
    double get_breakdown_point() const override { 
        return 1.0 - h_fraction_; 
    }
    
    double get_efficiency() const override { 
        // MVE has low efficiency but high breakdown point
        return 0.37;  // Approximate for multivariate normal
    }
    
private:
    // Standard MVE algorithm with random sampling
    RobustStats standard_mve_algorithm(const Mat& data, const Vec& weights, 
                                      int h, std::mt19937& gen) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        RobustStats best_stats(p);
        double min_volume = std::numeric_limits<double>::infinity();
        
        // Calculate number of trials
        int actual_trials = std::min(n_trials_, choose_approx(n, p + 1));
        
        for (int trial = 0; trial < actual_trials; ++trial) {
            // Select random (p+1)-subset
            UVec subset_idx = random_subset(n, p + 1, weights, gen);
            
            // Compute initial ellipsoid from subset
            Mat data_subset = data.rows(subset_idx);
            Vec weights_subset = weights(subset_idx);
            weights_subset /= arma::sum(weights_subset);
            
            Vec center = weighted_mean(data_subset, weights_subset);
            Mat cov = weighted_covariance(data_subset, weights_subset, center);
            
            // Add small regularization
            cov.diag() += 1e-10;
            
            // Check if valid
            if (!validate_covariance(cov)) {
                continue;
            }
            
            // Expand to h-subset
            RobustStats expanded = expand_to_h_subset(data, weights, center, cov, h);
            
            // Compute volume (proportional to determinant)
            double det = arma::det(expanded.covariance);
            
            if (det > 0 && det < min_volume) {
                min_volume = det;
                best_stats = expanded;
            }
        }
        
        // If no valid subset found, use weighted estimates
        if (min_volume == std::numeric_limits<double>::infinity()) {
            RobustStats robust_center(p);
            for (int j = 0; j < p; ++j) {
                robust_center.center(j) = weighted_median(data.col(j), weights);
                robust_center.scale(j) = weighted_mad(data.col(j), weights, robust_center.center(j));
            }
            robust_center.covariance = arma::diagmat(robust_center.scale % robust_center.scale);
            diagnostics_.add_warning("MVE: No valid subset found, using coordinate-wise robust estimates");
            return robust_center;
        }
        
        return best_stats;
    }
    
    // Fast MVE algorithm with concentration steps
    RobustStats fast_mve_algorithm(const Mat& data, const Vec& weights, 
                                  int h, std::mt19937& gen) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        RobustStats best_stats(p);
        double min_volume = std::numeric_limits<double>::infinity();
        
        // Number of initial subsets
        int n_initial = std::min(20, n_trials_ / 10);
        
        for (int init = 0; init < n_initial; ++init) {
            // Start with random (p+1)-subset
            UVec subset_idx = random_subset(n, p + 1, weights, gen);
            
            // Initial estimates
            Mat data_subset = data.rows(subset_idx);
            Vec center = arma::mean(data_subset, 0).t();
            Mat cov = arma::cov(data_subset);
            cov.diag() += 1e-10;
            
            // Concentration steps
            for (int conc = 0; conc < 5; ++conc) {
                // Compute Mahalanobis distances
                Mat cov_inv = safe_inv(cov);
                Vec distances(n);
                
                for (int i = 0; i < n; ++i) {
                    Vec x = data.row(i).t() - center;
                    distances(i) = arma::as_scalar(x.t() * cov_inv * x);
                }
                
                // Select h observations with smallest distances
                UVec sorted_idx = arma::sort_index(distances);
                UVec h_subset = sorted_idx.head(h);
                
                // Update estimates
                Mat data_h = data.rows(h_subset);
                Vec weights_h = weights(h_subset);
                weights_h /= arma::sum(weights_h);
                
                center = weighted_mean(data_h, weights_h);
                cov = weighted_covariance(data_h, weights_h, center);
                cov.diag() += 1e-10;
            }
            
            // Compute final volume
            double det = arma::det(cov);
            if (det > 0 && det < min_volume) {
                min_volume = det;
                best_stats.center = center;
                best_stats.covariance = cov;
                best_stats.scale = arma::sqrt(cov.diag());
            }
        }
        
        return best_stats;
    }
    
    // Expand ellipsoid to h-subset
    RobustStats expand_to_h_subset(const Mat& data, const Vec& weights,
                                   const Vec& center, const Mat& cov, int h) {
        int n = data.n_rows;
        
        // Compute Mahalanobis distances
        Mat cov_inv = safe_inv(cov);
        Vec distances(n);
        
        for (int i = 0; i < n; ++i) {
            Vec x = data.row(i).t() - center;
            distances(i) = arma::as_scalar(x.t() * cov_inv * x);
        }
        
        // Select h observations with smallest distances
        UVec sorted_idx = arma::sort_index(distances);
        UVec h_subset = sorted_idx.head(h);
        
        // Compute statistics for h-subset
        Mat data_h = data.rows(h_subset);
        Vec weights_h = weights(h_subset);
        weights_h /= arma::sum(weights_h);
        
        RobustStats stats(data.n_cols);
        stats.center = weighted_mean(data_h, weights_h);
        stats.covariance = weighted_covariance(data_h, weights_h, stats.center);
        
        // Apply finite sample correction
        double correction = compute_finite_sample_correction(h, data.n_cols);
        stats.covariance *= correction;
        
        // Regularize
        regularize_covariance(stats.covariance);
        stats.scale = arma::sqrt(stats.covariance.diag());
        
        return stats;
    }
    
    // Reweighting step for efficiency
    RobustStats reweight_step(const Mat& data, const Vec& weights, 
                             const RobustStats& initial_stats) {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Compute distances using initial MVE estimates
        Mat cov_inv = safe_inv(initial_stats.covariance);
        Vec distances(n);
        
        for (int i = 0; i < n; ++i) {
            Vec x = data.row(i).t() - initial_stats.center;
            distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
        }
        
        // Use chi-square cutoff for reweighting
        double chi2_cutoff = std::sqrt(R::qchisq(0.975, p, true, false));
        
        // Create new weights
        Vec new_weights = weights;
        for (int i = 0; i < n; ++i) {
            if (distances(i) > chi2_cutoff) {
                new_weights(i) *= 0.01;  // Downweight outliers
            }
        }
        new_weights /= arma::sum(new_weights);
        
        // Compute final estimates
        RobustStats final_stats(p);
        final_stats.center = weighted_mean(data, new_weights);
        final_stats.covariance = weighted_covariance(data, new_weights, final_stats.center);
        
        // Regularize
        regularize_covariance(final_stats.covariance);
        final_stats.scale = arma::sqrt(final_stats.covariance.diag());
        final_stats.weights = new_weights;
        
        // Identify outliers
        final_stats.outliers = arma::zeros<UVec>(n);
        for (int i = 0; i < n; ++i) {
            if (distances(i) > chi2_cutoff) {
                final_stats.outliers(i) = 1;
            }
        }
        
        return final_stats;
    }
    
    // Random subset selection
    UVec random_subset(int n, int k, const Vec& weights, std::mt19937& gen) {
        // Weighted sampling without replacement
        std::vector<double> probs(weights.begin(), weights.end());
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        
        std::set<int> selected;
        while (selected.size() < static_cast<size_t>(k)) {
            selected.insert(dist(gen));
        }
        
        UVec subset_idx(k);
        int i = 0;
        for (int idx : selected) {
            subset_idx(i++) = idx;
        }
        
        return subset_idx;
    }
    
    // Approximate n choose k
    int choose_approx(int n, int k) {
        if (k > n - k) k = n - k;
        
        double result = 1.0;
        for (int i = 0; i < k; ++i) {
            result *= (n - i);
            result /= (i + 1);
            if (result > 1e6) break;
        }
        
        return static_cast<int>(std::min(result, 1e6));
    }
    
    // Finite sample correction factor
    double compute_finite_sample_correction(int n, int p) {
        // Rousseeuw and van Driessen (1999) correction
        double alpha = (n - p) / static_cast<double>(n);
        double correction = 1.0 / alpha;
        
        // Additional small sample correction
        if (n < 10 * p) {
            correction *= (1.0 + 2.0 / (n - p));
        }
        
        return correction;
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::MVE, MVEEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_MVE_H
