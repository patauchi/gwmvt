#ifndef GWMVT_METHODS_PCA_ROBUST_STANDARD_H
#define GWMVT_METHODS_PCA_ROBUST_STANDARD_H

#include "../../base/robust_estimator.h"
#include "../../../core/algebra.h"

namespace gwmvt {

// Standard (non-robust) estimator for baseline comparison
class StandardEstimator : public RobustEstimator {
public:
    // Constructor
    explicit StandardEstimator(const RobustConfig& config = RobustConfig()) 
        : RobustEstimator(config) {}
    
    // Main estimation
    RobustStats estimate(const Mat& data, const Vec& weights) override {
        int n = data.n_rows;
        int p = data.n_cols;
        
        RobustStats stats(p);
        
        // Weighted mean
        stats.center = weighted_mean(data, weights);
        
        // Weighted covariance
        stats.covariance = weighted_covariance(data, weights, stats.center);
        
        // Regularize for numerical stability
        regularize_covariance(stats.covariance);
        
        // Scale (standard deviation)
        stats.scale = arma::sqrt(stats.covariance.diag());
        
        // Store weights
        stats.weights = weights;
        
        // No outliers in standard method
        stats.outliers = arma::zeros<UVec>(n);
        
        return stats;
    }
    
    // Method properties
    std::string get_name() const override { 
        return "standard"; 
    }
    
    double get_breakdown_point() const override { 
        return 0.0;  // No robustness
    }
    
    double get_efficiency() const override { 
        return 1.0;  // Maximum efficiency at normal distribution
    }
};

// Register with factory
// REGISTER_ROBUST_ESTIMATOR(RobustMethod::STANDARD, StandardEstimator);

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_STANDARD_H