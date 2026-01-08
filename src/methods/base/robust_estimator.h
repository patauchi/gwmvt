#ifndef GWMVT_METHODS_BASE_ROBUST_ESTIMATOR_H
#define GWMVT_METHODS_BASE_ROBUST_ESTIMATOR_H

#include "core/types.h"
#include <memory>
#include <string>

namespace gwmvt {

// Abstract base class for all robust estimation methods
class RobustEstimator {
protected:
    RobustConfig config_;
    DiagnosticInfo diagnostics_;

public:
    // Constructor
    explicit RobustEstimator(const RobustConfig& config = RobustConfig())
        : config_(config) {}

    virtual ~RobustEstimator() = default;

    // Main estimation interface
    virtual RobustStats estimate(const Mat& data, const Vec& weights) = 0;

    // Optional: iterative estimation with convergence tracking
    virtual RobustStats estimate_iterative(const Mat& data,
                                          const Vec& weights,
                                          ConvergenceInfo& conv_info) {
        // Default implementation: just call estimate
        RobustStats result = estimate(data, weights);
        conv_info.converged = true;
        conv_info.iterations = 1;
        return result;
    }

    // Method properties
    virtual std::string get_name() const = 0;
    virtual double get_breakdown_point() const = 0;
    virtual double get_efficiency() const = 0;

    // Configuration
    void set_config(const RobustConfig& config) {
        config_ = config;
    }

    const RobustConfig& get_config() const {
        return config_;
    }

    // Diagnostics
    const DiagnosticInfo& get_diagnostics() const {
        return diagnostics_;
    }

    void clear_diagnostics() {
        diagnostics_ = DiagnosticInfo();
    }

protected:
    // Helper methods for derived classes

    // Weighted median
    double weighted_median(const Vec& x, const Vec& weights) {
        int n = x.n_elem;
        UVec sorted_idx = arma::sort_index(x);

        Vec sorted_x = x(sorted_idx);
        Vec sorted_w = weights(sorted_idx);

        Vec cum_weights = arma::cumsum(sorted_w);
        double half_weight = arma::sum(sorted_w) * 0.5;

        for (int i = 0; i < n; ++i) {
            if (cum_weights(i) >= half_weight) {
                return sorted_x(i);
            }
        }

        return sorted_x(n-1);
    }

    // Weighted MAD (Median Absolute Deviation)
    double weighted_mad(const Vec& x, const Vec& weights, double center) {
        Vec deviations = arma::abs(x - center);
        return 1.4826 * weighted_median(deviations, weights);
    }

    // Weighted quantile
    double weighted_quantile(const Vec& x, const Vec& weights, double q) {
        int n = x.n_elem;
        UVec sorted_idx = arma::sort_index(x);

        Vec sorted_x = x(sorted_idx);
        Vec sorted_w = weights(sorted_idx);

        Vec cum_weights = arma::cumsum(sorted_w);
        double target_weight = arma::sum(sorted_w) * q;

        for (int i = 0; i < n; ++i) {
            if (cum_weights(i) >= target_weight) {
                return sorted_x(i);
            }
        }

        return sorted_x(n-1);
    }

    // Check convergence
    bool check_convergence(const Vec& old_center, const Vec& new_center,
                          int iter, ConvergenceInfo& info) {
        double error = arma::max(arma::abs(new_center - old_center));
        info.error = error;
        info.iterations = iter;

        if (error < config_.tolerance) {
            info.converged = true;
            info.message = "Converged successfully";
            return true;
        }

        if (iter >= config_.max_iterations) {
            info.converged = false;
            info.message = "Maximum iterations reached";
            diagnostics_.add_warning("Convergence not achieved after " +
                                   std::to_string(iter) + " iterations");
            return true; // Stop anyway
        }

        return false;
    }

    // Regularize covariance matrix
    void regularize_covariance(Mat& cov_mat, double reg_param = 1e-8) {
        // Enforce symmetry to avoid numerical asymmetries before regularization
        cov_mat = arma::symmatu(cov_mat);
        cov_mat.diag() += reg_param;
    }

    // Check if covariance matrix is valid
    bool validate_covariance(const Mat& cov_mat) {
        // Check for NaN/Inf
        if (cov_mat.has_nan() || cov_mat.has_inf()) {
            diagnostics_.add_error("Covariance matrix contains NaN or Inf");
            return false;
        }

        // Check symmetry
        double sym_error = arma::norm(cov_mat - cov_mat.t(), "fro");
        if (sym_error > 1e-10) {
            diagnostics_.add_warning("Covariance matrix is not symmetric");
        }

        // Check positive definiteness
        Vec eigenvals = arma::eig_sym(cov_mat);
        if (eigenvals(0) < -1e-10) {
            diagnostics_.add_error("Covariance matrix is not positive semi-definite");
            return false;
        }

        return true;
    }
};

// Factory for creating robust estimators
// class RobustEstimatorFactory {
// public:
//     using Creator = std::function<std::unique_ptr<RobustEstimator>()>;
//
// private:
//     static std::map<RobustMethod, Creator>& get_creators() {
//         static std::map<RobustMethod, Creator> creators;
//         return creators;
//     }
//
// public:
//     // Register a new estimator type
//     static void register_estimator(RobustMethod method, Creator creator) {
//         get_creators()[method] = creator;
//     }
//
//     // Create an estimator
//     static std::unique_ptr<RobustEstimator> create(RobustMethod method) {
//         auto& creators = get_creators();
//         auto it = creators.find(method);
//
//         if (it == creators.end()) {
//             throw std::runtime_error("Unknown robust method: " +
//                                    std::to_string(static_cast<int>(method)));
//         }
//
//         return it->second();
//     }
//
//     // Get list of available methods
//     static std::vector<RobustMethod> available_methods() {
//         std::vector<RobustMethod> methods;
//         for (const auto& pair : get_creators()) {
//             methods.push_back(pair.first);
//         }
//         return methods;
//     }
// };

// Helper macro for registering estimators
// Macro for automatic registration of robust estimators
// #define REGISTER_ROBUST_ESTIMATOR(method, class_name) \
//     static bool registered_##class_name = []() { \
//         RobustEstimatorFactory::register_estimator( \
//             method, \
//             []() { return std::make_unique<class_name>(); } \
//         ); \
//         return true; \
//     }()

} // namespace gwmvt

#endif // GWMVT_METHODS_BASE_ROBUST_ESTIMATOR_H
