#ifndef GWMVT_METHODS_PCA_ROBUST_LOF_H
#define GWMVT_METHODS_PCA_ROBUST_LOF_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace gwmvt {

// Local Outlier Factor (LOF) based robust estimator
class LOFEstimator : public RobustEstimator {
private:
    int k_;                  // Number of nearest neighbors
    double lof_threshold_;   // LOF threshold for outlier detection
    bool use_weighted_lof_;  // Use weighted version of LOF
    
public:
    // Constructor
    explicit LOFEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), k_(10), lof_threshold_(1.5), use_weighted_lof_(true) {
        
        // Check for custom k
        auto it = config.params.find("k");
        if (it != config.params.end()) {
            k_ = std::max(3, static_cast<int>(it->second));
        }
        
        // Check for LOF threshold
        it = config.params.find("lof_threshold");
        if (it != config.params.end()) {
            lof_threshold_ = std::max(1.0, it->second);
        }
        
        // Check for weighted LOF flag
        it = config.params.find("use_weighted_lof");
        if (it != config.params.end()) {
            use_weighted_lof_ = (it->second > 0.5);
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Compute LOF scores
        Vec lof_scores = compute_lof_scores(data, weights);
        
        // Convert LOF scores to weights
        Vec lof_weights = convert_lof_to_weights(lof_scores, weights);
        
        // Compute robust statistics using LOF weights
        RobustStats stats(p);
        stats.center = weighted_mean(data, lof_weights);
        stats.covariance = weighted_covariance(data, lof_weights, stats.center);
        
        // Regularize covariance
        regularize_covariance(stats.covariance);
        stats.scale = arma::sqrt(stats.covariance.diag());
        
        // Store weights and identify outliers
        stats.weights = lof_weights;
        stats.outliers = arma::zeros<UVec>(n);
        for (int i = 0; i < n; ++i) {
            if (lof_scores(i) > lof_threshold_) {
                stats.outliers(i) = 1;
            }
        }
        
        return stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return "lof"; 
    }
    
    double get_breakdown_point() const override { 
        // LOF doesn't have a classical breakdown point
        // Return approximate value based on k
        return static_cast<double>(k_) / (2.0 * k_ + 1.0);
    }
    
    double get_efficiency() const override { 
        // LOF efficiency depends on data structure
        return 0.7;  // Approximate
    }
    
private:
    // Compute LOF scores for all points
    Vec compute_lof_scores(const Mat& data, const Vec& weights) {
        int n = data.n_rows;
        Vec lof_scores(n, arma::fill::ones);
        
        // Adjust k based on available points
        int k_actual = std::min(k_, static_cast<int>(arma::sum(weights > 0.01)) - 1);
        if (k_actual < 3) {
            diagnostics_.add_warning("LOF: Too few points for reliable estimation");
            return lof_scores;
        }
        
        // Compute pairwise distances (can be optimized with spatial indexing)
        Mat dist_matrix = compute_distance_matrix(data);
        
        // Compute LOF for each point
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            if (weights(i) < 0.01) continue;
            
            // Find k-nearest neighbors
            Vec distances_i = dist_matrix.row(i).t();
            
            // Weight distances by spatial weights if using weighted LOF
            if (use_weighted_lof_) {
                distances_i = distances_i / (weights + 1e-10);
            }
            
            // Get k nearest neighbors (excluding self)
            distances_i(i) = arma::datum::inf;
            UVec sorted_idx = arma::sort_index(distances_i);
            UVec k_neighbors = sorted_idx.head(k_actual);
            
            // Compute local reachability density (LRD)
            double lrd_i = compute_lrd(i, k_neighbors, dist_matrix, k_actual);
            
            // Compute average LRD of neighbors
            double avg_lrd_neighbors = 0.0;
            for (int j : k_neighbors) {
                // Find neighbors of neighbor j
                Vec distances_j = dist_matrix.row(j).t();
                distances_j(j) = arma::datum::inf;
                UVec sorted_idx_j = arma::sort_index(distances_j);
                UVec k_neighbors_j = sorted_idx_j.head(k_actual);
                
                double lrd_j = compute_lrd(j, k_neighbors_j, dist_matrix, k_actual);
                avg_lrd_neighbors += lrd_j;
            }
            avg_lrd_neighbors /= k_actual;
            
            // LOF score
            if (lrd_i > 0) {
                lof_scores(i) = avg_lrd_neighbors / lrd_i;
            } else {
                lof_scores(i) = 2.0;  // Mark as outlier
            }
        }
        
        return lof_scores;
    }
    
    // Compute local reachability density
    double compute_lrd(int point_idx, const UVec& k_neighbors, 
                      const Mat& dist_matrix, int k) {
        double sum_reach_dist = 0.0;
        
        for (int neighbor_idx : k_neighbors) {
            // k-distance of neighbor
            Vec distances_neighbor = dist_matrix.row(neighbor_idx).t();
            distances_neighbor(neighbor_idx) = arma::datum::inf;
            Vec sorted_dist = arma::sort(distances_neighbor);
            double k_dist_neighbor = sorted_dist(k - 1);
            
            // Reachability distance
            double reach_dist = std::max(dist_matrix(point_idx, neighbor_idx), k_dist_neighbor);
            sum_reach_dist += reach_dist;
        }
        
        if (sum_reach_dist > 0) {
            return static_cast<double>(k) / sum_reach_dist;
        } else {
            return arma::datum::inf;
        }
    }
    
    // Compute distance matrix
    Mat compute_distance_matrix(const Mat& data) {
        int n = data.n_rows;
        Mat dist_matrix(n, n, arma::fill::zeros);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dist = arma::norm(data.row(i) - data.row(j), 2);
                dist_matrix(i, j) = dist;
                dist_matrix(j, i) = dist;
            }
        }
        
        return dist_matrix;
    }
    
    // Convert LOF scores to weights
    Vec convert_lof_to_weights(const Vec& lof_scores, const Vec& spatial_weights) {
        int n = lof_scores.n_elem;
        Vec lof_weights(n);
        
        // Different weighting schemes
        if (use_weighted_lof_) {
            // Smooth weighting based on LOF scores
            for (int i = 0; i < n; ++i) {
                double lof = lof_scores(i);
                if (lof <= 1.0) {
                    lof_weights(i) = 1.0;
                } else if (lof < lof_threshold_) {
                    // Linear decrease
                    lof_weights(i) = (lof_threshold_ - lof) / (lof_threshold_ - 1.0);
                } else {
                    lof_weights(i) = 0.01;  // Small weight for outliers
                }
                
                // Combine with spatial weights
                lof_weights(i) *= spatial_weights(i);
            }
        } else {
            // Hard thresholding
            for (int i = 0; i < n; ++i) {
                if (lof_scores(i) <= lof_threshold_) {
                    lof_weights(i) = spatial_weights(i);
                } else {
                    lof_weights(i) = 0.01 * spatial_weights(i);
                }
            }
        }
        
        // Normalize weights
        lof_weights /= arma::sum(lof_weights);
        
        return lof_weights;
    }
    
    // Iterative LOF with refinement
    RobustStats estimate_iterative(const Mat& data, const Vec& weights,
                                  ConvergenceInfo& conv_info) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Initial estimate
        RobustStats stats = estimate(data, weights);
        Vec old_center = stats.center;
        Vec lof_weights = stats.weights;
        
        // Iterative refinement
        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // Transform data using current covariance estimate
            Mat transformed_data = transform_data(data, stats);
            
            // Recompute LOF in transformed space
            Vec lof_scores = compute_lof_scores(transformed_data, weights);
            lof_weights = convert_lof_to_weights(lof_scores, weights);
            
            // Update estimates
            stats.center = weighted_mean(data, lof_weights);
            stats.covariance = weighted_covariance(data, lof_weights, stats.center);
            regularize_covariance(stats.covariance);
            
            // Check convergence
            if (check_convergence(old_center, stats.center, iter + 1, conv_info)) {
                break;
            }
            
            old_center = stats.center;
        }
        
        stats.scale = arma::sqrt(stats.covariance.diag());
        stats.weights = lof_weights;
        
        return stats;
    }
    
    // Transform data to standardized space
    Mat transform_data(const Mat& data, const RobustStats& stats) {
        Mat data_centered = data.each_row() - stats.center.t();
        
        // Compute square root of inverse covariance
        Mat cov_inv = safe_inv(stats.covariance);
        Vec eigenvals;
        Mat eigenvecs;
        arma::eig_sym(eigenvals, eigenvecs, cov_inv);
        
        // Transform: X_new = X * V * D^{1/2}
        Mat transform_matrix = eigenvecs * arma::diagmat(arma::sqrt(eigenvals));
        return data_centered * transform_matrix;
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::LOF, LOFEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_LOF_H
