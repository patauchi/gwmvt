#ifndef GWMVT_METHODS_BASE_GW_METHOD_H
#define GWMVT_METHODS_BASE_GW_METHOD_H

#include "../../core/types.h"
#include "../../core/kernels.h"
#include "../../core/weights.h"
#include <memory>
#include <atomic>

namespace gwmvt {

// Abstract base class for all geographically weighted methods
class GWMethod {
protected:
    // Core data
    Mat data_;
    Mat coords_;
    GWConfig config_;
    
    // Spatial weights calculator
    std::unique_ptr<SpatialWeights> spatial_weights_;
    
    // Progress tracking
    std::unique_ptr<ProgressReporter> progress_;
    std::atomic<int> progress_counter_{0};
    
    // Diagnostic information
    DiagnosticInfo diagnostics_;
    
    // Validation methods
    virtual void validate_inputs() {
        if (data_.n_rows != coords_.n_rows) {
            throw std::invalid_argument("Number of data rows must match coordinate rows");
        }
        if (coords_.n_cols != 2) {
            throw std::invalid_argument("Coordinates must have 2 columns");
        }
        if (config_.bandwidth <= 0 && !config_.adaptive_bandwidth) {
            throw std::invalid_argument("Bandwidth must be positive");
        }
    }
    
    // Create appropriate progress reporter
    void setup_progress() {
        if (config_.verbose && config_.show_progress) {
            progress_ = std::make_unique<ConsoleProgress>();
        } else {
            progress_ = std::make_unique<NullProgress>();
        }
    }
    
    // Abstract methods that derived classes must implement
    virtual void fit_local(int location) = 0;
    virtual void prepare_fit() = 0;
    virtual void finalize_fit() = 0;
    
public:
    // Constructor
    GWMethod(const Mat& data, const Mat& coords, const GWConfig& config)
        : data_(data), coords_(coords), config_(config) {
        setup_progress();
    }
    
    virtual ~GWMethod() = default;
    
    // Disable copy constructor and assignment
    GWMethod(const GWMethod&) = delete;
    GWMethod& operator=(const GWMethod&) = delete;
    
    // Enable move semantics
    GWMethod(GWMethod&&) = default;
    GWMethod& operator=(GWMethod&&) = default;
    
    // Main interface methods
    virtual std::unique_ptr<GWResult> fit() {
        // Validate inputs
        validate_inputs();
        
        // Initialize spatial weights
        spatial_weights_ = std::make_unique<SpatialWeights>(coords_, config_);
        
        // Prepare for fitting
        prepare_fit();
        
        // Report initial step
        progress_->report_step("Fitting " + get_method_name());
        
        int n = data_.n_rows;
        
        // Process each location
        bool can_parallel = false;
#ifdef _OPENMP
        can_parallel = (config_.parallel_strategy != ParallelStrategy::SEQUENTIAL && n > 100);
#else
        // Without OpenMP support, always run sequential to preserve progress updates
        can_parallel = false;
#endif
        if (can_parallel) {
            fit_parallel();
        } else {
            fit_sequential();
        }
        
        // Finalize results
        finalize_fit();
        
        // Report completion
        progress_->finish();
        
        // Return result
        return create_result();
    }
    
    // Prediction interface (optional for some methods)
    virtual Mat predict(const Mat& newdata, const Mat& newcoords) {
        throw std::runtime_error("Prediction not implemented for " + get_method_name());
    }
    
    // Accessors
    virtual std::string get_method_name() const = 0;
    const DiagnosticInfo& get_diagnostics() const { return diagnostics_; }
    
    // Configuration updates
    void set_bandwidth(double bandwidth) {
        config_.bandwidth = bandwidth;
        if (spatial_weights_) {
            spatial_weights_->set_bandwidth(bandwidth);
        }
    }
    
    void set_kernel(KernelType kernel) {
        config_.kernel_type = kernel;
        if (spatial_weights_) {
            spatial_weights_ = std::make_unique<SpatialWeights>(coords_, config_);
        }
    }
    
protected:
    // Sequential fitting
    void fit_sequential() {
        int n = data_.n_rows;
        for (int i = 0; i < n; ++i) {
            // Allow user interrupt from R to stop long-running computations
            if ((i & 255) == 0) {
                Rcpp::checkUserInterrupt();
            }
            fit_local(i);
            progress_->report(i + 1, n);
        }
    }
    
    // Parallel fitting
    virtual void fit_parallel() = 0;
    
    // Create result object
    virtual std::unique_ptr<GWResult> create_result() = 0;
    
    // Get weights for a specific location
    Vec get_weights(int location) {
        return spatial_weights_->compute_weights(location);
    }
    
    // Check for numerical issues
    bool check_numerical_stability(const Mat& matrix, const std::string& name) {
        if (matrix.has_nan() || matrix.has_inf()) {
            diagnostics_.add_warning(name + " contains NaN or Inf values");
            diagnostics_.numerical_issues = true;
            return false;
        }
        
        double cond = arma::cond(matrix);
        if (cond > 1e10) {
            diagnostics_.add_warning(name + " is ill-conditioned (condition number: " + 
                                   std::to_string(cond) + ")");
            diagnostics_.numerical_issues = true;
            return false;
        }
        
        return true;
    }
};

} // namespace gwmvt

#endif // GWMVT_METHODS_BASE_GW_METHOD_H
