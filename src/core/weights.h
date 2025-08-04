#ifndef GWMVT_CORE_WEIGHTS_H
#define GWMVT_CORE_WEIGHTS_H

#include "types.h"
#include "kernels.h"
#include "distances.h"
#include <memory>
#include <vector>

namespace gwmvt {

// Class for computing spatial weights
class SpatialWeights {
private:
    const Mat& coords_;
    GWConfig config_;
    std::unique_ptr<KernelFunction> kernel_;
    std::unique_ptr<DistanceCalculator> distance_calc_;
    
    // Cached adaptive bandwidths if using adaptive kernel
    Vec adaptive_bandwidths_;
    bool has_adaptive_bw_;
    
public:
    // Constructor
    SpatialWeights(const Mat& coords, const GWConfig& config)
        : coords_(coords), config_(config), has_adaptive_bw_(false) {
        
        // Create kernel function
        kernel_ = KernelFactory::create(config.kernel_type);
        
        // Create distance calculator
        distance_calc_ = std::make_unique<EuclideanDistance>();
        
        // Calculate adaptive bandwidths if needed
        if (config.adaptive_bandwidth) {
            calculate_adaptive_bandwidths();
        }
    }
    
    // Compute weights for a specific location
    Vec compute_weights(int location) {
        // Calculate distances from location to all other points
        Vec distances = distance_calc_->calculate_from_point(coords_, location);
        
        // Get bandwidth for this location
        double bandwidth = get_bandwidth(location);
        
        // Compute kernel weights
        Vec weights = kernel_->compute_weights(distances, bandwidth);
        
        // Set self-weight to zero
        weights(location) = 0.0;
        
        // Normalize weights
        double sum_weights = arma::sum(weights);
        if (sum_weights > 0) {
            weights /= sum_weights;
        }
        
        return weights;
    }
    
    // Compute weight matrix for all locations
    Mat compute_weight_matrix() {
        int n = coords_.n_rows;
        Mat W(n, n);
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            W.row(i) = compute_weights(i).t();
        }
        
        return W;
    }
    
    // Compute sparse weight matrix
    SpMat compute_sparse_weight_matrix(double threshold = 1e-10) {
        int n = coords_.n_rows;
        
        // Estimate number of non-zero elements
        int estimated_nnz = n * std::min(100, n);
        std::vector<arma::uword> row_indices;
        std::vector<arma::uword> col_indices;
        std::vector<double> values;
        
        row_indices.reserve(estimated_nnz);
        col_indices.reserve(estimated_nnz);
        values.reserve(estimated_nnz);
        
        // Fill sparse matrix
        for (int i = 0; i < n; ++i) {
            Vec weights = compute_weights(i);
            
            for (int j = 0; j < n; ++j) {
                if (std::abs(weights(j)) > threshold) {
                    row_indices.push_back(i);
                    col_indices.push_back(j);
                    values.push_back(weights(j));
                }
            }
        }
        
        // Create sparse matrix
        arma::umat locations(2, values.size());
        locations.row(0) = arma::conv_to<arma::urowvec>::from(row_indices);
        locations.row(1) = arma::conv_to<arma::urowvec>::from(col_indices);
        
        SpMat W(locations, arma::conv_to<arma::vec>::from(values), n, n);
        
        return W;
    }
    
    // Get bandwidth for a specific location
    double get_bandwidth(int location) const {
        if (has_adaptive_bw_) {
            return adaptive_bandwidths_(location);
        }
        return config_.bandwidth;
    }
    
    // Set fixed bandwidth
    void set_bandwidth(double bandwidth) {
        config_.bandwidth = bandwidth;
        has_adaptive_bw_ = false;
    }
    
    // Set adaptive bandwidths
    void set_adaptive_bandwidths(const Vec& bandwidths) {
        if (bandwidths.n_elem != coords_.n_rows) {
            throw std::invalid_argument("Adaptive bandwidth vector size must match number of locations");
        }
        adaptive_bandwidths_ = bandwidths;
        has_adaptive_bw_ = true;
    }
    
    // Get effective number of neighbors
    Vec get_effective_neighbors() {
        int n = coords_.n_rows;
        Vec eff_neighbors(n);
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Vec weights = compute_weights(i);
            // Effective number = 1 / sum(w_i^2)
            eff_neighbors(i) = 1.0 / arma::dot(weights, weights);
        }
        
        return eff_neighbors;
    }
    
private:
    // Calculate adaptive bandwidths using k-nearest neighbors
    void calculate_adaptive_bandwidths() {
        int n = coords_.n_rows;
        adaptive_bandwidths_.set_size(n);
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Vec distances = distance_calc_->calculate_from_point(coords_, i);
            distances(i) = arma::datum::inf; // Exclude self
            
            // Sort distances
            Vec sorted_dist = arma::sort(distances);
            
            // k-th nearest neighbor distance
            int k = std::min(config_.adaptive_k, n - 1);
            adaptive_bandwidths_(i) = sorted_dist(k - 1);
        }
        
        has_adaptive_bw_ = true;
        
        // If using adaptive kernel, update it
        if (config_.kernel_type == KernelType::ADAPTIVE) {
            if (auto adaptive_kernel = dynamic_cast<AdaptiveKernel*>(kernel_.get())) {
                adaptive_kernel->update_bandwidths(adaptive_bandwidths_);
            }
        }
    }
};

// Bandwidth selection methods
class BandwidthSelector {
public:
    // Cross-validation for bandwidth selection
    static double select_cv(const Mat& data, const Mat& coords, 
                           const Vec& candidates, const GWConfig& config,
                           int n_folds = 10) {
        int n = data.n_rows;
        int n_candidates = candidates.n_elem;
        Vec cv_scores(n_candidates, arma::fill::zeros);
        
        // Create folds
        UVec indices = arma::shuffle(arma::linspace<UVec>(0, n-1, n));
        int fold_size = n / n_folds;
        
        // Test each bandwidth
        #pragma omp parallel for
        for (int b = 0; b < n_candidates; ++b) {
            double cv_error = 0.0;
            
            // Cross-validation
            for (int fold = 0; fold < n_folds; ++fold) {
                // Split data
                int start = fold * fold_size;
                int end = (fold == n_folds - 1) ? n : (fold + 1) * fold_size;
                
                UVec test_idx = indices.subvec(start, end - 1);
                UVec train_idx;
                
                if (fold == 0) {
                    train_idx = indices.subvec(end, n - 1);
                } else if (fold == n_folds - 1) {
                    train_idx = indices.subvec(0, start - 1);
                } else {
                    train_idx = arma::join_cols(indices.subvec(0, start - 1),
                                               indices.subvec(end, n - 1));
                }
                
                // Compute error for this fold
                // This is method-specific and should be implemented by derived classes
                cv_error += compute_cv_error(data, coords, train_idx, test_idx, 
                                            candidates(b), config);
            }
            
            cv_scores(b) = cv_error / n_folds;
        }
        
        // Find optimal bandwidth
        arma::uword opt_idx = cv_scores.index_min();
        return candidates(opt_idx);
    }
    
    // AIC-based selection
    static double select_aic(const Mat& data, const Mat& coords,
                            const Vec& candidates, const GWConfig& config) {
        int n_candidates = candidates.n_elem;
        Vec aic_scores(n_candidates);
        
        #pragma omp parallel for
        for (int b = 0; b < n_candidates; ++b) {
            aic_scores(b) = compute_aic(data, coords, candidates(b), config);
        }
        
        arma::uword opt_idx = aic_scores.index_min();
        return candidates(opt_idx);
    }
    
private:
    // These should be implemented by specific methods
    static double compute_cv_error(const Mat& data, const Mat& coords,
                                  const UVec& train_idx, const UVec& test_idx,
                                  double bandwidth, const GWConfig& config) {
        // Placeholder - should be implemented by specific method
        return 0.0;
    }
    
    static double compute_aic(const Mat& data, const Mat& coords,
                             double bandwidth, const GWConfig& config) {
        // Placeholder - should be implemented by specific method
        return 0.0;
    }
};

} // namespace gwmvt

#endif // GWMVT_CORE_WEIGHTS_H