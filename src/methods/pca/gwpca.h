#ifndef GWMVT_METHODS_PCA_GWPCA_H
#define GWMVT_METHODS_PCA_GWPCA_H

#include "../base/gw_method.h"
#include "../base/robust_estimator.h"
#include "../../core/algebra.h"
#include <memory>
#include <atomic>

namespace gwmvt {

// Geographically Weighted Principal Component Analysis
class GWPCA : public GWMethod {
private:
    // Configuration
    std::shared_ptr<GWPCAConfig> pca_config_;
    
    // Results storage
    std::unique_ptr<GWPCAResult> result_;
    
    // Robust estimator
    std::unique_ptr<RobustEstimator> robust_estimator_;
    
    // Spatial outliers
    UVec spatial_outliers_;
    
    // Data standardization info
    Vec data_mean_;
    Vec data_sd_;
    Mat standardized_data_;
    
protected:
    // Prepare for fitting
    void prepare_fit() override;
    
    // Fit at a single location
    void fit_local(int location) override;
    
    // Finalize results
    void finalize_fit() override;
    
    // Parallel fitting
    void fit_parallel() override;
    
    // Create result object
    std::unique_ptr<GWResult> create_result() override;
    
private:
    // Detect spatial outliers
    void detect_spatial_outliers();
    
    // Standardize data if using correlation
    void standardize_data();
    
    // Fit PCA at a single location with specific method
    LocalPCAResult fit_local_pca(int location, const Vec& weights);
    
    // Standard PCA
    LocalPCAResult fit_standard_pca(const Mat& data, const Vec& weights);
    
    // Robust PCA using selected estimator
    LocalPCAResult fit_robust_pca(const Mat& data, const Vec& weights);
    
    // Calculate scores for a location
    void calculate_scores(int location, const LocalPCAResult& pca_result);
    
    // Calculate variance explained
    void calculate_variance_explained(int location, const LocalPCAResult& pca_result);
    
public:
    // Constructor
    GWPCA(const Mat& data, const Mat& coords, const GWConfig& config);
    
    // Destructor
    ~GWPCA() override;
    
    // Get method name
    std::string get_method_name() const override { return "GWPCA"; }
    
    // Prediction for new data
    Mat predict(const Mat& newdata, const Mat& newcoords) override;
    
    // Get specific results
    const GWPCAResult* get_pca_result() const { 
        return result_.get(); 
    }
    
    // Bandwidth selection specific to GWPCA
    static double select_bandwidth_cv(const Mat& data, const Mat& coords,
                                     const Vec& candidates, const GWPCAConfig& config,
                                     int n_folds = 10);
    
    static double select_bandwidth_aic(const Mat& data, const Mat& coords,
                                      const Vec& candidates, const GWPCAConfig& config);
};

// Factory registration for GWPCA
class GWPCAFactory {
public:
    static std::unique_ptr<GWMethod> create(const Mat& data, const Mat& coords, 
                                            const GWConfig& config) {
        return std::make_unique<GWPCA>(data, coords, config);
    }
    
    static bool register_factory() {
        // This would register with a global factory system
        return true;
    }
};

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_GWPCA_H
