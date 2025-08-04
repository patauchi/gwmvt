#ifndef GWMVT_CORE_DISTANCES_H
#define GWMVT_CORE_DISTANCES_H

#include "types.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gwmvt {

// Abstract base class for distance calculations
class DistanceCalculator {
public:
    virtual ~DistanceCalculator() = default;
    
    // Calculate distances from one point to all others
    virtual Vec calculate_from_point(const Mat& coords, int focal_idx) const = 0;
    
    // Calculate full distance matrix
    virtual Mat calculate_matrix(const Mat& coords) const = 0;
    
    // Calculate distances between two sets of coordinates
    virtual Mat calculate_cross(const Mat& coords1, const Mat& coords2) const = 0;
    
    // Get distance type name
    virtual std::string get_name() const = 0;
};

// Euclidean distance calculator
class EuclideanDistance : public DistanceCalculator {
public:
    Vec calculate_from_point(const Mat& coords, int focal_idx) const override {
        int n = coords.n_rows;
        Vec distances(n);
        
        double focal_x = coords(focal_idx, 0);
        double focal_y = coords(focal_idx, 1);
        
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            double dx = coords(i, 0) - focal_x;
            double dy = coords(i, 1) - focal_y;
            distances(i) = std::sqrt(dx * dx + dy * dy);
        }
        
        return distances;
    }
    
    Mat calculate_matrix(const Mat& coords) const override {
        int n = coords.n_rows;
        Mat dist_mat(n, n, arma::fill::zeros);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dx = coords(i, 0) - coords(j, 0);
                double dy = coords(i, 1) - coords(j, 1);
                double dist = std::sqrt(dx * dx + dy * dy);
                dist_mat(i, j) = dist;
                dist_mat(j, i) = dist;
            }
        }
        
        return dist_mat;
    }
    
    Mat calculate_cross(const Mat& coords1, const Mat& coords2) const override {
        int n1 = coords1.n_rows;
        int n2 = coords2.n_rows;
        Mat dist_mat(n1, n2);
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                double dx = coords1(i, 0) - coords2(j, 0);
                double dy = coords1(i, 1) - coords2(j, 1);
                dist_mat(i, j) = std::sqrt(dx * dx + dy * dy);
            }
        }
        
        return dist_mat;
    }
    
    std::string get_name() const override { return "euclidean"; }
};

// Manhattan distance calculator
class ManhattanDistance : public DistanceCalculator {
public:
    Vec calculate_from_point(const Mat& coords, int focal_idx) const override {
        int n = coords.n_rows;
        Vec distances(n);
        
        double focal_x = coords(focal_idx, 0);
        double focal_y = coords(focal_idx, 1);
        
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            double dx = std::abs(coords(i, 0) - focal_x);
            double dy = std::abs(coords(i, 1) - focal_y);
            distances(i) = dx + dy;
        }
        
        return distances;
    }
    
    Mat calculate_matrix(const Mat& coords) const override {
        int n = coords.n_rows;
        Mat dist_mat(n, n, arma::fill::zeros);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dx = std::abs(coords(i, 0) - coords(j, 0));
                double dy = std::abs(coords(i, 1) - coords(j, 1));
                double dist = dx + dy;
                dist_mat(i, j) = dist;
                dist_mat(j, i) = dist;
            }
        }
        
        return dist_mat;
    }
    
    Mat calculate_cross(const Mat& coords1, const Mat& coords2) const override {
        int n1 = coords1.n_rows;
        int n2 = coords2.n_rows;
        Mat dist_mat(n1, n2);
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                double dx = std::abs(coords1(i, 0) - coords2(j, 0));
                double dy = std::abs(coords1(i, 1) - coords2(j, 1));
                dist_mat(i, j) = dx + dy;
            }
        }
        
        return dist_mat;
    }
    
    std::string get_name() const override { return "manhattan"; }
};

// Great Circle distance calculator (for geographic coordinates)
class GreatCircleDistance : public DistanceCalculator {
private:
    static constexpr double EARTH_RADIUS_KM = 6371.0;
    
    double deg2rad(double deg) const {
        return deg * M_PI / 180.0;
    }
    
    double haversine(double lat1, double lon1, double lat2, double lon2) const {
        double dlat = deg2rad(lat2 - lat1);
        double dlon = deg2rad(lon2 - lon1);
        
        double a = std::sin(dlat/2) * std::sin(dlat/2) +
                   std::cos(deg2rad(lat1)) * std::cos(deg2rad(lat2)) *
                   std::sin(dlon/2) * std::sin(dlon/2);
        
        double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a));
        return EARTH_RADIUS_KM * c;
    }
    
public:
    Vec calculate_from_point(const Mat& coords, int focal_idx) const override {
        int n = coords.n_rows;
        Vec distances(n);
        
        double focal_lon = coords(focal_idx, 0);
        double focal_lat = coords(focal_idx, 1);
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            distances(i) = haversine(focal_lat, focal_lon, coords(i, 1), coords(i, 0));
        }
        
        return distances;
    }
    
    Mat calculate_matrix(const Mat& coords) const override {
        int n = coords.n_rows;
        Mat dist_mat(n, n, arma::fill::zeros);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dist = haversine(coords(i, 1), coords(i, 0), 
                                      coords(j, 1), coords(j, 0));
                dist_mat(i, j) = dist;
                dist_mat(j, i) = dist;
            }
        }
        
        return dist_mat;
    }
    
    Mat calculate_cross(const Mat& coords1, const Mat& coords2) const override {
        int n1 = coords1.n_rows;
        int n2 = coords2.n_rows;
        Mat dist_mat(n1, n2);
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                dist_mat(i, j) = haversine(coords1(i, 1), coords1(i, 0),
                                         coords2(j, 1), coords2(j, 0));
            }
        }
        
        return dist_mat;
    }
    
    std::string get_name() const override { return "great_circle"; }
};

// Minkowski distance calculator
class MinkowskiDistance : public DistanceCalculator {
private:
    double p_;
    
public:
    explicit MinkowskiDistance(double p = 2.0) : p_(p) {}
    
    Vec calculate_from_point(const Mat& coords, int focal_idx) const override {
        int n = coords.n_rows;
        Vec distances(n);
        
        double focal_x = coords(focal_idx, 0);
        double focal_y = coords(focal_idx, 1);
        
        if (p_ == 1.0) {
            // Manhattan distance
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                double dx = std::abs(coords(i, 0) - focal_x);
                double dy = std::abs(coords(i, 1) - focal_y);
                distances(i) = dx + dy;
            }
        } else if (p_ == 2.0) {
            // Euclidean distance
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                double dx = coords(i, 0) - focal_x;
                double dy = coords(i, 1) - focal_y;
                distances(i) = std::sqrt(dx * dx + dy * dy);
            }
        } else if (std::isinf(p_)) {
            // Chebyshev distance
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                double dx = std::abs(coords(i, 0) - focal_x);
                double dy = std::abs(coords(i, 1) - focal_y);
                distances(i) = std::max(dx, dy);
            }
        } else {
            // General Minkowski distance
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                double dx = std::abs(coords(i, 0) - focal_x);
                double dy = std::abs(coords(i, 1) - focal_y);
                distances(i) = std::pow(std::pow(dx, p_) + std::pow(dy, p_), 1.0/p_);
            }
        }
        
        return distances;
    }
    
    Mat calculate_matrix(const Mat& coords) const override {
        int n = coords.n_rows;
        Mat dist_mat(n, n, arma::fill::zeros);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            Vec dists = calculate_from_point(coords, i);
            dist_mat.row(i) = dists.t();
        }
        
        return dist_mat;
    }
    
    Mat calculate_cross(const Mat& coords1, const Mat& coords2) const override {
        int n1 = coords1.n_rows;
        int n2 = coords2.n_rows;
        Mat dist_mat(n1, n2);
        
        #pragma omp parallel for
        for (int i = 0; i < n1; ++i) {
            Vec dists = calculate_from_point(coords2, i);
            dist_mat.row(i) = dists.t();
        }
        
        return dist_mat;
    }
    
    std::string get_name() const override { 
        return "minkowski_p" + std::to_string(p_); 
    }
};

// Factory for creating distance calculators
class DistanceFactory {
public:
    static std::unique_ptr<DistanceCalculator> create(const std::string& type) {
        if (type == "euclidean") {
            return std::make_unique<EuclideanDistance>();
        } else if (type == "manhattan") {
            return std::make_unique<ManhattanDistance>();
        } else if (type == "great_circle" || type == "haversine") {
            return std::make_unique<GreatCircleDistance>();
        } else if (type == "minkowski") {
            return std::make_unique<MinkowskiDistance>(2.0);
        } else {
            throw std::invalid_argument("Unknown distance type: " + type);
        }
    }
    
    static std::unique_ptr<DistanceCalculator> create_minkowski(double p) {
        return std::make_unique<MinkowskiDistance>(p);
    }
};

} // namespace gwmvt

#endif // GWMVT_CORE_DISTANCES_H