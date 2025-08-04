#ifndef GWMVT_CORE_KERNELS_H
#define GWMVT_CORE_KERNELS_H

#include "types.h"
#include <cmath>
#include <memory>

namespace gwmvt {

// Abstract base class for kernel functions
class KernelFunction {
public:
    virtual ~KernelFunction() = default;
    
    // Compute weights for a vector of distances
    virtual Vec compute_weights(const Vec& distances, double bandwidth) const = 0;
    
    // Compute single weight
    virtual double compute_weight(double distance, double bandwidth) const = 0;
    
    // Get kernel name
    virtual std::string name() const = 0;
    
    // Check if kernel has compact support
    virtual bool has_compact_support() const = 0;
    
    // Get support radius (for compact kernels)
    virtual double get_support_radius(double bandwidth) const {
        return bandwidth;
    }
};

// Gaussian kernel
class GaussianKernel : public KernelFunction {
public:
    Vec compute_weights(const Vec& distances, double bandwidth) const override {
        return arma::exp(-(distances % distances) / (2.0 * bandwidth * bandwidth));
    }
    
    double compute_weight(double distance, double bandwidth) const override {
        return std::exp(-(distance * distance) / (2.0 * bandwidth * bandwidth));
    }
    
    std::string name() const override { return "gaussian"; }
    
    bool has_compact_support() const override { return false; }
};

// Bisquare kernel
class BisquareKernel : public KernelFunction {
public:
    Vec compute_weights(const Vec& distances, double bandwidth) const override {
        Vec u = distances / bandwidth;
        Vec weights(distances.n_elem);
        
        for (size_t i = 0; i < u.n_elem; ++i) {
            if (u(i) <= 1.0) {
                double temp = 1.0 - u(i) * u(i);
                weights(i) = temp * temp;
            } else {
                weights(i) = 0.0;
            }
        }
        
        return weights;
    }
    
    double compute_weight(double distance, double bandwidth) const override {
        double u = distance / bandwidth;
        if (u <= 1.0) {
            double temp = 1.0 - u * u;
            return temp * temp;
        }
        return 0.0;
    }
    
    std::string name() const override { return "bisquare"; }
    
    bool has_compact_support() const override { return true; }
};

// Exponential kernel
class ExponentialKernel : public KernelFunction {
public:
    Vec compute_weights(const Vec& distances, double bandwidth) const override {
        return arma::exp(-distances / bandwidth);
    }
    
    double compute_weight(double distance, double bandwidth) const override {
        return std::exp(-distance / bandwidth);
    }
    
    std::string name() const override { return "exponential"; }
    
    bool has_compact_support() const override { return false; }
};

// Tricube kernel
class TricubeKernel : public KernelFunction {
public:
    Vec compute_weights(const Vec& distances, double bandwidth) const override {
        Vec u = distances / bandwidth;
        Vec weights(distances.n_elem);
        
        for (size_t i = 0; i < u.n_elem; ++i) {
            if (u(i) <= 1.0) {
                double temp = 1.0 - u(i) * u(i) * u(i);
                weights(i) = temp * temp * temp;
            } else {
                weights(i) = 0.0;
            }
        }
        
        return weights;
    }
    
    double compute_weight(double distance, double bandwidth) const override {
        double u = distance / bandwidth;
        if (u <= 1.0) {
            double temp = 1.0 - u * u * u;
            return temp * temp * temp;
        }
        return 0.0;
    }
    
    std::string name() const override { return "tricube"; }
    
    bool has_compact_support() const override { return true; }
};

// Boxcar (uniform) kernel
class BoxcarKernel : public KernelFunction {
public:
    Vec compute_weights(const Vec& distances, double bandwidth) const override {
        Vec weights(distances.n_elem);
        
        for (size_t i = 0; i < distances.n_elem; ++i) {
            weights(i) = (distances(i) <= bandwidth) ? 1.0 : 0.0;
        }
        
        return weights;
    }
    
    double compute_weight(double distance, double bandwidth) const override {
        return (distance <= bandwidth) ? 1.0 : 0.0;
    }
    
    std::string name() const override { return "boxcar"; }
    
    bool has_compact_support() const override { return true; }
};

// Adaptive kernel wrapper
class AdaptiveKernel : public KernelFunction {
private:
    std::unique_ptr<KernelFunction> base_kernel_;
    Vec adaptive_bandwidths_;
    
public:
    AdaptiveKernel(std::unique_ptr<KernelFunction> base_kernel, 
                   const Vec& adaptive_bandwidths)
        : base_kernel_(std::move(base_kernel)), 
          adaptive_bandwidths_(adaptive_bandwidths) {}
    
    Vec compute_weights(const Vec& distances, double) const override {
        // Note: bandwidth parameter is ignored, using adaptive bandwidths
        Vec weights(distances.n_elem);
        
        for (size_t i = 0; i < distances.n_elem; ++i) {
            weights(i) = base_kernel_->compute_weight(distances(i), 
                                                     adaptive_bandwidths_(i));
        }
        
        return weights;
    }
    
    double compute_weight(double distance, double bandwidth) const override {
        // For single weight, use provided bandwidth
        return base_kernel_->compute_weight(distance, bandwidth);
    }
    
    std::string name() const override { 
        return "adaptive_" + base_kernel_->name(); 
    }
    
    bool has_compact_support() const override { 
        return base_kernel_->has_compact_support(); 
    }
    
    void update_bandwidths(const Vec& new_bandwidths) {
        adaptive_bandwidths_ = new_bandwidths;
    }
};

// Factory for creating kernel functions
class KernelFactory {
public:
    static std::unique_ptr<KernelFunction> create(KernelType type) {
        switch (type) {
            case KernelType::GAUSSIAN:
                return std::make_unique<GaussianKernel>();
            case KernelType::BISQUARE:
                return std::make_unique<BisquareKernel>();
            case KernelType::EXPONENTIAL:
                return std::make_unique<ExponentialKernel>();
            case KernelType::TRICUBE:
                return std::make_unique<TricubeKernel>();
            case KernelType::BOXCAR:
                return std::make_unique<BoxcarKernel>();
            case KernelType::ADAPTIVE:
                // Default to adaptive gaussian
                return std::make_unique<AdaptiveKernel>(
                    std::make_unique<GaussianKernel>(), 
                    Vec());
            default:
                throw std::invalid_argument("Unknown kernel type");
        }
    }
    
    static std::unique_ptr<KernelFunction> create(const std::string& name) {
        if (name == "gaussian") return create(KernelType::GAUSSIAN);
        if (name == "bisquare") return create(KernelType::BISQUARE);
        if (name == "exponential") return create(KernelType::EXPONENTIAL);
        if (name == "tricube") return create(KernelType::TRICUBE);
        if (name == "boxcar") return create(KernelType::BOXCAR);
        if (name == "adaptive") return create(KernelType::ADAPTIVE);
        
        throw std::invalid_argument("Unknown kernel name: " + name);
    }
};

} // namespace gwmvt

#endif // GWMVT_CORE_KERNELS_H