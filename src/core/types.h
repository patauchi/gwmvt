#ifndef GWMVT_CORE_TYPES_H
#define GWMVT_CORE_TYPES_H

#include <RcppArmadillo.h>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <functional>

namespace gwmvt {

// Type aliases for clarity
using Mat = arma::mat;
using Vec = arma::vec;
using UVec = arma::uvec;
using Cube = arma::cube;
using SpMat = arma::sp_mat;

// Kernel function types
enum class KernelType {
    GAUSSIAN,
    BISQUARE,
    EXPONENTIAL,
    TRICUBE,
    BOXCAR,
    ADAPTIVE
};

// Robust method types
enum class RobustMethod {
    STANDARD,
    ADAPTIVE_HUBER,
    ADAPTIVE_MCD,
    SPATIAL_TRIM,
    MVE,
    S_ESTIMATOR,
    MM_ESTIMATOR,
    LOF,
    BACON,
    SPATIAL_DEPTH,
    ROBPCA
};

// Bandwidth selection methods
enum class BandwidthMethod {
    FIXED,
    ADAPTIVE_QUANTILE,
    ADAPTIVE_NN,
    CV,
    AIC
};

// Parallel execution strategies
enum class ParallelStrategy {
    SEQUENTIAL,
    OPENMP,
    RCPP_PARALLEL,
    AUTO
};

// Forward declarations
class MethodConfig;
class RobustConfig;

// Base configuration structure
struct GWConfig {
    // Spatial parameters
    double bandwidth = 0.0;
    KernelType kernel_type = KernelType::GAUSSIAN;
    bool adaptive_bandwidth = false;
    int adaptive_k = 30;
    
    // Computational parameters
    ParallelStrategy parallel_strategy = ParallelStrategy::AUTO;
    int n_threads = 0;
    int batch_size = 0;
    double memory_limit_mb = 4096.0;
    
    // Progress and logging
    bool verbose = false;
    bool show_progress = true;
    
    // Method-specific configuration
    std::shared_ptr<MethodConfig> method_config;
};

// Base class for method-specific configurations
class MethodConfig {
public:
    virtual ~MethodConfig() = default;
    virtual std::string get_method_name() const = 0;
    virtual std::shared_ptr<MethodConfig> clone() const = 0;
};

// Configuration for robust methods
class RobustConfig : public MethodConfig {
public:
    RobustMethod method = RobustMethod::STANDARD;
    bool detect_outliers = false;
    double outlier_threshold = 2.5;
    double breakdown_point = 0.5;
    double efficiency = 0.95;
    int max_iterations = 50;
    double tolerance = 1e-6;
    
    // Method-specific parameters
    std::map<std::string, double> params;
    
    std::string get_method_name() const override {
        return "robust";
    }
    
    std::shared_ptr<MethodConfig> clone() const override {
        return std::make_shared<RobustConfig>(*this);
    }
};

// GWPCA-specific configuration
class GWPCAConfig : public MethodConfig {
public:
    int n_components = 0;
    bool use_correlation = false;
    RobustMethod robust_method = RobustMethod::STANDARD;
    bool detect_outliers = true;
    double outlier_threshold = 2.5;
    
    // Robust method specific parameters
    double trim_proportion = 0.1;
    double h_fraction = 0.75;
    int lof_k = 10;
    double bacon_alpha = 0.05;
    std::string depth_type = "mahalanobis";
    int robpca_k_max = 10;
    
    std::string get_method_name() const override {
        return "gwpca";
    }
    
    std::shared_ptr<MethodConfig> clone() const override {
        return std::make_shared<GWPCAConfig>(*this);
    }
};

// Structure for robust statistics
struct RobustStats {
    Vec center;
    Mat covariance;
    Vec scale;
    double total_scale;
    Vec weights;
    UVec outliers;
    
    RobustStats() = default;
    
    RobustStats(int p) : 
        center(p, arma::fill::zeros),
        covariance(p, p, arma::fill::zeros),
        scale(p, arma::fill::zeros),
        total_scale(0.0) {}
};

// Structure for PCA results at a single location
struct LocalPCAResult {
    Vec eigenvalues;
    Mat eigenvectors;
    Vec center;
    double total_variance;
    bool converged;
    int iterations;
    std::string message;
    
    LocalPCAResult() = default;
    
    LocalPCAResult(int p) :
        eigenvalues(p),
        eigenvectors(p, p),
        center(p),
        total_variance(0.0),
        converged(true),
        iterations(0) {}
};

// Base result structure
class GWResult {
protected:
    GWConfig config_;
    
public:
    GWResult() = default;
    explicit GWResult(const GWConfig& config) : config_(config) {}
    virtual ~GWResult() = default;
    
    GWConfig get_config() const { return config_; }
    
    virtual std::string get_method_name() const = 0;
};

// GWPCA result structure
class GWPCAResult : public GWResult {
public:
    // Direct access members for efficiency
    Mat eigenvalues;      // n x p matrix
    Cube loadings;        // n x p x k cube
    Mat scores;           // n x k matrix
    Mat var_explained;    // n x k matrix
    Mat coords;           // n x 2 matrix
    Mat centers;          // n x p matrix
    UVec spatial_outliers; // n vector
    
    // Constructor
    GWPCAResult() = default;
    
    GWPCAResult(int n, int p, int k, const GWConfig& config) : 
        GWResult(config),
        eigenvalues(n, p, arma::fill::zeros),
        loadings(n, p, k, arma::fill::zeros),
        scores(n, k, arma::fill::zeros),
        var_explained(n, k, arma::fill::zeros),
        centers(n, p, arma::fill::zeros) {
        
        eigenvalues.fill(arma::datum::nan);
        loadings.fill(arma::datum::nan);
        scores.fill(arma::datum::nan);
        var_explained.fill(arma::datum::nan);
        centers.fill(arma::datum::nan);
    }
    
    std::string get_method_name() const override {
        return "gwpca";
    }
    
    // Convenience accessors
    int n_locations() const { return eigenvalues.n_rows; }
    int n_variables() const { return eigenvalues.n_cols; }
    int n_components() const { return scores.n_cols; }
};

// Progress reporter interface
class ProgressReporter {
public:
    virtual ~ProgressReporter() = default;
    virtual void report(int current, int total) = 0;
    virtual void report_step(const std::string& step) = 0;
    virtual void finish() = 0;
};

// Console progress reporter
class ConsoleProgress : public ProgressReporter {
private:
    int last_percent_ = -1;
    std::string current_step_;
    
public:
    void report(int current, int total) override {
        int percent = (100 * current) / total;
        if (percent != last_percent_) {
            Rcpp::Rcout << "\r" << current_step_ << " Progress: " << percent << "%" << std::flush;
            last_percent_ = percent;
        }
    }
    
    void report_step(const std::string& step) override {
        current_step_ = step;
        Rcpp::Rcout << "\n" << step << std::endl;
    }
    
    void finish() override {
        Rcpp::Rcout << "\r" << current_step_ << " Progress: 100%\n" << std::flush;
    }
};

// Null progress reporter (no output)
class NullProgress : public ProgressReporter {
public:
    void report(int, int) override {}
    void report_step(const std::string&) override {}
    void finish() override {}
};

// Convergence information
struct ConvergenceInfo {
    bool converged = false;
    int iterations = 0;
    double error = 0.0;
    std::string message;
};

// Diagnostic information
struct DiagnosticInfo {
    bool numerical_issues = false;
    bool convergence_issues = false;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    
    void add_warning(const std::string& msg) {
        warnings.push_back(msg);
    }
    
    void add_error(const std::string& msg) {
        errors.push_back(msg);
    }
    
    bool has_issues() const {
        return numerical_issues || convergence_issues || 
               !warnings.empty() || !errors.empty();
    }
};

} // namespace gwmvt

#endif // GWMVT_CORE_TYPES_H
