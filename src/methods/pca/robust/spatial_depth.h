#ifndef GWMVT_METHODS_PCA_ROBUST_SPATIAL_DEPTH_H
#define GWMVT_METHODS_PCA_ROBUST_SPATIAL_DEPTH_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"
#include <algorithm>
#include <cmath>

namespace gwmvt {

// Spatial depth based robust estimator
class SpatialDepthEstimator : public RobustEstimator {
private:
    std::string depth_type_;     // Type of depth: "mahalanobis", "projection", "spatial"
    double depth_threshold_;     // Threshold for outlier detection
    int n_projections_;         // Number of projections for projection depth
    bool use_geo_weights_;      // Use geographic weights in depth calculation
    
public:
    // Constructor
    explicit SpatialDepthEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config), depth_type_("mahalanobis"), 
          depth_threshold_(0.1), n_projections_(100), use_geo_weights_(true) {
        
        // Check for depth type
        auto it = config.params.find("depth_type");
        if (it != config.params.end()) {
            if (it->second == 1.0) depth_type_ = "mahalanobis";
            else if (it->second == 2.0) depth_type_ = "spatial";
            else if (it->second == 3.0) depth_type_ = "projection";
        }
        
        // Check for depth threshold
        it = config.params.find("depth_threshold");
        if (it != config.params.end()) {
            depth_threshold_ = std::max(0.01, std::min(0.5, it->second));
        }
        
        // Check for number of projections
        it = config.params.find("n_projections");
        if (it != config.params.end()) {
            n_projections_ = std::max(10, static_cast<int>(it->second));
        }
        
        // Check for geographic weights flag
        it = config.params.find("use_geo_weights");
        if (it != config.params.end()) {
            use_geo_weights_ = (it->second > 0.5);
        }
    }
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Compute depth values
        Vec depths;
        if (depth_type_ == "mahalanobis") {
            depths = compute_mahalanobis_depth(data, weights);
        } else if (depth_type_ == "spatial") {
            depths = compute_spatial_depth(data, weights);
        } else if (depth_type_ == "projection") {
            depths = compute_projection_depth(data, weights);
        } else {
            diagnostics_.add_error("Unknown depth type: " + depth_type_);
            depths = arma::ones<Vec>(n);
        }
        
        // Convert depths to weights
        Vec depth_weights = depths % weights;
        depth_weights /= arma::sum(depth_weights);
        
        // Compute robust statistics using depth weights
        RobustStats stats(p);
        stats.center = weighted_mean(data, depth_weights);
        stats.covariance = weighted_covariance(data, depth_weights, stats.center);
        
        // Regularize covariance
        regularize_covariance(stats.covariance);
        stats.scale = arma::sqrt(stats.covariance.diag());
        stats.weights = depth_weights;
        
        // Identify outliers based on depth
        stats.outliers = arma::zeros<UVec>(n);
        for (int i = 0; i < n; ++i) {
            if (depths(i) < depth_threshold_) {
                stats.outliers(i) = 1;
            }
        }
        
        return stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return "spatial_depth"; 
    }
    
    double get_breakdown_point() const override { 
        // Depth methods can achieve up to 50% breakdown
        return 0.5;
    }
    
    double get_efficiency() const override { 
        // Efficiency depends on depth type
        if (depth_type_ == "mahalanobis") return 0.85;
        else if (depth_type_ == "spatial") return 0.70;
        else return 0.65;  // projection depth
    }
    
private:
    // Compute Mahalanobis depth
    Vec compute_mahalanobis_depth(const Mat& data, const Vec& weights) {
        int n = data.n_rows;
        int p = data.n_cols;
        Vec depths(n);
        
        // Robust initial estimates
        Vec robust_center(p);
        for (int j = 0; j < p; ++j) {
            Vec valid_data = data.col(j)(arma::find(weights > 0.01));
            robust_center(j) = arma::median(valid_data);
        }
        
        // Robust covariance using weighted quartiles
        Mat data_centered = data.each_row() - robust_center.t();
        UVec valid_idx = arma::find(weights > 0.01);
        
        if (valid_idx.n_elem < data.n_cols + 1) {
            depths.fill(1.0);
            return depths;
        }
        
        Mat valid_data = data_centered.rows(valid_idx);
        Vec valid_weights = weights(valid_idx);
        valid_weights /= arma::sum(valid_weights);
        
        Mat cov_robust = weighted_covariance(valid_data, valid_weights, arma::zeros<Vec>(p));
        regularize_covariance(cov_robust);
        
        // Compute depths
        Mat cov_inv = safe_inv(cov_robust);
        
        for (int i = 0; i < n; ++i) {
            Vec x = data_centered.row(i).t();
            double mahal_dist = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
            depths(i) = 1.0 / (1.0 + mahal_dist);
        }
        
        // Normalize depths
        depths /= arma::max(depths);
        
        return depths;
    }
    
    // Compute spatial depth
    Vec compute_spatial_depth(const Mat& data, const Vec& weights) {
        int n = data.n_rows;
        Vec depths(n, arma::fill::zeros);
        
        // For each point, compute its spatial depth
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            if (weights(i) < 0.01) continue;
            
            Vec point_i = data.row(i).t();
            double depth_sum = 0.0;
            double weight_sum = 0.0;
            
            // Average unit vector pointing from other points to i
            Vec avg_direction(data.n_cols, arma::fill::zeros);
            
            for (int j = 0; j < n; ++j) {
                if (i == j || weights(j) < 0.01) continue;
                
                Vec diff = point_i - data.row(j).t();
                double norm_diff = arma::norm(diff, 2);
                
                if (norm_diff > 1e-10) {
                    Vec unit_dir = diff / norm_diff;
                    avg_direction += weights(j) * unit_dir;
                    weight_sum += weights(j);
                }
            }
            
            if (weight_sum > 0) {
                avg_direction /= weight_sum;
                depths(i) = 1.0 - arma::norm(avg_direction, 2);
            }
        }
        
        // Ensure positive depths
        depths = arma::clamp(depths, 0.0, 1.0);
        
        return depths;
    }
    
    // Compute projection depth
    Vec compute_projection_depth(const Mat& data, const Vec& weights) {
        int n = data.n_rows;
        int p = data.n_cols;
        Vec depths(n, arma::fill::ones);
        
        // Generate random projection directions
        arma::arma_rng::set_seed_random();
        Mat projections(p, n_projections_, arma::fill::randn);
        
        // Normalize projection directions
        for (int j = 0; j < n_projections_; ++j) {
            projections.col(j) /= arma::norm(projections.col(j), 2);
        }
        
        // For each projection direction
        for (int j = 0; j < n_projections_; ++j) {
            Vec direction = projections.col(j);
            
            // Project all points
            Vec projected = data * direction;
            
            // Compute weighted median and MAD
            Vec valid_proj = projected(arma::find(weights > 0.01));
            Vec valid_weights = weights(arma::find(weights > 0.01));
            
            double med = weighted_median(valid_proj, valid_weights);
            double mad = weighted_mad(valid_proj, valid_weights, med);
            
            if (mad < 1e-10) continue;
            
            // Update depths based on outlyingness in this direction
            for (int i = 0; i < n; ++i) {
                double outlyingness = std::abs(projected(i) - med) / mad;
                double depth_j = 1.0 / (1.0 + outlyingness);
                depths(i) = std::min(depths(i), depth_j);
            }
        }
        
        return depths;
    }
    
    // Iterative refinement
    RobustStats estimate_iterative(const Mat& data, const Vec& weights,
                                  ConvergenceInfo& conv_info) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        // Initial estimate
        RobustStats stats = estimate(data, weights);
        Vec old_center = stats.center;
        
        // Iterative deepening
        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // Transform data using current estimate
            Mat data_centered = data.each_row() - stats.center.t();
            Mat cov_sqrt_inv = compute_sqrt_inverse(stats.covariance);
            Mat transformed_data = data_centered * cov_sqrt_inv;
            
            // Recompute depths in transformed space
            Vec depths;
            if (depth_type_ == "mahalanobis") {
                // In transformed space, use Euclidean distance
                depths.set_size(n);
                for (int i = 0; i < n; ++i) {
                    double dist = arma::norm(transformed_data.row(i), 2);
                    depths(i) = 1.0 / (1.0 + dist);
                }
            } else {
                // Use selected depth type
                depths = (depth_type_ == "spatial") ? 
                    compute_spatial_depth(transformed_data, weights) :
                    compute_projection_depth(transformed_data, weights);
            }
            
            // Update weights and recompute statistics
            Vec depth_weights = depths % weights;
            depth_weights /= arma::sum(depth_weights);
            
            stats.center = weighted_mean(data, depth_weights);
            stats.covariance = weighted_covariance(data, depth_weights, stats.center);
            regularize_covariance(stats.covariance);
            
            // Check convergence
            if (check_convergence(old_center, stats.center, iter + 1, conv_info)) {
                break;
            }
            
            old_center = stats.center;
        }
        
        stats.scale = arma::sqrt(stats.covariance.diag());
        return stats;
    }
    
    // Compute matrix square root inverse
    Mat compute_sqrt_inverse(const Mat& cov) {
        Vec eigenvals;
        Mat eigenvecs;
        arma::eig_sym(eigenvals, eigenvecs, cov);
        
        // Compute sqrt inverse
        Vec sqrt_inv_eigenvals = 1.0 / arma::sqrt(arma::clamp(eigenvals, 1e-10, arma::datum::inf));
        return eigenvecs * arma::diagmat(sqrt_inv_eigenvals) * eigenvecs.t();
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::SPATIAL_DEPTH, SpatialDepthEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_SPATIAL_DEPTH_H