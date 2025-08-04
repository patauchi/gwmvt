#ifndef GWMVT_METHODS_PCA_ROBUST_ALL_ROBUST_METHODS_H
#define GWMVT_METHODS_PCA_ROBUST_ALL_ROBUST_METHODS_H

// Include all robust estimation methods for GWPCA

#include "huber.h"
#include "mcd.h"
#include "mve.h"
#include "spatial_trim.h"
#include "s_estimator.h"
#include "mm_estimator.h"
#include "lof.h"
#include "bacon.h"
#include "spatial_depth.h"
#include "robpca.h"

// Standard (non-robust) estimator
#include "standard.h"

namespace gwmvt {

// Factory function to create the appropriate robust estimator
inline std::unique_ptr<RobustEstimator> create_robust_estimator(RobustMethod method, const RobustConfig& config) {
    switch (method) {
        case RobustMethod::STANDARD:
            return std::make_unique<StandardEstimator>(config);
            
        case RobustMethod::ADAPTIVE_HUBER:
            return std::make_unique<HuberEstimator>(config);
            
        case RobustMethod::ADAPTIVE_MCD:
            return std::make_unique<MCDEstimator>(config);
            
        case RobustMethod::SPATIAL_TRIM:
            return std::make_unique<SpatialTrimEstimator>(config);
            
        case RobustMethod::MVE:
            return std::make_unique<MVEEstimator>(config);
            
        case RobustMethod::S_ESTIMATOR:
            return std::make_unique<SEstimator>(config);
            
        case RobustMethod::MM_ESTIMATOR:
            return std::make_unique<MMEstimator>(config);
            
        case RobustMethod::LOF:
            return std::make_unique<LOFEstimator>(config);
            
        case RobustMethod::BACON:
            return std::make_unique<BACONEstimator>(config);
            
        case RobustMethod::SPATIAL_DEPTH:
            return std::make_unique<SpatialDepthEstimator>(config);
            
        case RobustMethod::ROBPCA:
            return std::make_unique<ROBPCAEstimator>(config);
            
        default:
            throw std::invalid_argument("Unknown robust method");
    }
}

// Get method name as string
inline std::string get_robust_method_name(RobustMethod method) {
    switch (method) {
        case RobustMethod::STANDARD: return "standard";
        case RobustMethod::ADAPTIVE_HUBER: return "adaptive_huber";
        case RobustMethod::ADAPTIVE_MCD: return "adaptive_mcd";
        case RobustMethod::SPATIAL_TRIM: return "spatial_trim";
        case RobustMethod::MVE: return "mve";
        case RobustMethod::S_ESTIMATOR: return "s_estimator";
        case RobustMethod::MM_ESTIMATOR: return "mm_estimator";
        case RobustMethod::LOF: return "lof";
        case RobustMethod::BACON: return "bacon";
        case RobustMethod::SPATIAL_DEPTH: return "spatial_depth";
        case RobustMethod::ROBPCA: return "robpca";
        default: return "unknown";
    }
}

// Convert string to RobustMethod enum
inline RobustMethod parse_robust_method(const std::string& method) {
    if (method == "standard") return RobustMethod::STANDARD;
    if (method == "adaptive_huber") return RobustMethod::ADAPTIVE_HUBER;
    if (method == "adaptive_mcd") return RobustMethod::ADAPTIVE_MCD;
    if (method == "spatial_trim") return RobustMethod::SPATIAL_TRIM;
    if (method == "mve") return RobustMethod::MVE;
    if (method == "s_estimator") return RobustMethod::S_ESTIMATOR;
    if (method == "mm_estimator") return RobustMethod::MM_ESTIMATOR;
    if (method == "lof") return RobustMethod::LOF;
    if (method == "bacon") return RobustMethod::BACON;
    if (method == "spatial_depth") return RobustMethod::SPATIAL_DEPTH;
    if (method == "robpca") return RobustMethod::ROBPCA;
    
    throw std::invalid_argument("Unknown robust method: " + method);
}

} // namespace gwmvt

#endif // GWMVT_METHODS_PCA_ROBUST_ALL_ROBUST_METHODS_H