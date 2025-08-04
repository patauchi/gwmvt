#ifndef GWMVT_CORE_ALGEBRA_H
#define GWMVT_CORE_ALGEBRA_H

#include "types.h"
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gwmvt {

// Weighted mean calculation
inline Vec weighted_mean(const Mat& X, const Vec& weights) {
    return X.t() * weights / arma::sum(weights);
}

// Weighted covariance matrix
inline Mat weighted_covariance(const Mat& X, const Vec& weights, const Vec& center) {
    int n = X.n_rows;
    int p = X.n_cols;
    
    // Center the data
    Mat X_centered = X.each_row() - center.t();
    
    // Compute weighted covariance
    Mat cov(p, p, arma::fill::zeros);
    double sum_weights = arma::sum(weights);
    
    for (int i = 0; i < n; ++i) {
        Vec x = X_centered.row(i).t();
        cov += weights(i) * (x * x.t());
    }
    
    return cov / sum_weights;
}

// Parallel weighted covariance computation
inline Mat weighted_covariance_parallel(const Mat& X, const Vec& weights, const Vec& center) {
    int n = X.n_rows;
    int p = X.n_cols;
    
    Mat X_centered = X.each_row() - center.t();
    
    // Use thread-local storage for accumulation
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    
    std::vector<Mat> local_covs(n_threads, Mat(p, p, arma::fill::zeros));
    
    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            Vec x = X_centered.row(i).t();
            local_covs[tid] += weights(i) * (x * x.t());
        }
    }
    
    // Reduce
    Mat cov = local_covs[0];
    for (int i = 1; i < n_threads; ++i) {
        cov += local_covs[i];
    }
    
    return cov / arma::sum(weights);
}

// Memory-efficient eigendecomposition for symmetric matrices
inline void eigen_decomp_symmetric(const Mat& A, Vec& eigenvalues, Mat& eigenvectors) {
    arma::eig_sym(eigenvalues, eigenvectors, A);
    
    // Sort in descending order
    UVec indices = arma::sort_index(eigenvalues, "descend");
    eigenvalues = eigenvalues(indices);
    eigenvectors = eigenvectors.cols(indices);
}

// Truncated SVD for large matrices
inline void truncated_svd(const Mat& X, Mat& U, Vec& s, Mat& V, int k) {
    // Use Armadillo's SVD for economy size decomposition
    Mat U_full, V_full;
    Vec s_full;
    
    bool success = arma::svd_econ(U_full, s_full, V_full, X, "both", "std");
    
    if (!success) {
        throw std::runtime_error("SVD decomposition failed");
    }
    
    // Truncate to k components
    k = std::min(k, static_cast<int>(s_full.n_elem));
    
    U = U_full.cols(0, k-1);
    s = s_full.subvec(0, k-1);
    V = V_full.cols(0, k-1);
}

// Robust scale estimator (MAD)
inline double mad_scale(const Vec& x, const Vec& weights, double center) {
    Vec deviations = arma::abs(x - center);
    
    // Weighted median of deviations
    UVec sorted_idx = arma::sort_index(deviations);
    Vec sorted_dev = deviations(sorted_idx);
    Vec sorted_weights = weights(sorted_idx);
    
    Vec cum_weights = arma::cumsum(sorted_weights);
    double half_weight = arma::sum(sorted_weights) * 0.5;
    
    double weighted_median = 0.0;
    for (size_t i = 0; i < cum_weights.n_elem; ++i) {
        if (cum_weights(i) >= half_weight) {
            weighted_median = sorted_dev(i);
            break;
        }
    }
    
    return 1.4826 * weighted_median;
}

// Matrix condition number check
inline bool check_condition_number(const Mat& A, double max_cond = 1e10) {
    double cond = arma::cond(A);
    return cond < max_cond;
}

// Safe matrix inversion with regularization
inline Mat safe_inv(const Mat& A, double reg_param = 1e-8) {
    Mat A_reg = A;
    A_reg.diag() += reg_param;
    
    Mat A_inv;
    bool success = arma::inv_sympd(A_inv, A_reg);
    
    if (!success) {
        // Try with more regularization
        A_reg.diag() += reg_param * 10;
        success = arma::inv_sympd(A_inv, A_reg);
        
        if (!success) {
            throw std::runtime_error("Matrix inversion failed even with regularization");
        }
    }
    
    return A_inv;
}

// Mahalanobis distance calculation
inline Vec mahalanobis_distance(const Mat& X, const Vec& center, const Mat& cov_inv) {
    int n = X.n_rows;
    Vec distances(n);
    
    Mat X_centered = X.each_row() - center.t();
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        Vec x = X_centered.row(i).t();
        distances(i) = std::sqrt(arma::as_scalar(x.t() * cov_inv * x));
    }
    
    return distances;
}

// Weighted quantile calculation
inline double weighted_quantile(const Vec& x, const Vec& weights, double q) {
    int n = x.n_elem;
    UVec sorted_idx = arma::sort_index(x);
    
    Vec sorted_x = x(sorted_idx);
    Vec sorted_w = weights(sorted_idx);
    
    Vec cum_weights = arma::cumsum(sorted_w);
    double target_weight = arma::sum(sorted_w) * q;
    
    for (int i = 0; i < n; ++i) {
        if (cum_weights(i) >= target_weight) {
            // Linear interpolation for more accurate quantile
            if (i > 0 && cum_weights(i-1) < target_weight) {
                double w1 = cum_weights(i-1);
                double w2 = cum_weights(i);
                double alpha = (target_weight - w1) / (w2 - w1);
                return sorted_x(i-1) * (1 - alpha) + sorted_x(i) * alpha;
            }
            return sorted_x(i);
        }
    }
    
    return sorted_x(n-1);
}

// Weighted median (special case of quantile)
inline double weighted_median(const Vec& x, const Vec& weights) {
    return weighted_quantile(x, weights, 0.5);
}

// Efficient matrix multiplication for sparse weights
inline Mat sparse_weighted_multiply(const SpMat& W, const Mat& X) {
    return W * X;
}

// Check for numerical issues in matrix
inline bool check_numerical_stability(const Mat& A, DiagnosticInfo& diag) {
    // Check for NaN or Inf
    if (A.has_nan()) {
        diag.add_error("Matrix contains NaN values");
        diag.numerical_issues = true;
        return false;
    }
    
    if (A.has_inf()) {
        diag.add_error("Matrix contains Inf values");
        diag.numerical_issues = true;
        return false;
    }
    
    // Check condition number
    double cond = arma::cond(A);
    if (cond > 1e12) {
        diag.add_warning("Matrix is severely ill-conditioned (condition number: " + 
                        std::to_string(cond) + ")");
        diag.numerical_issues = true;
        return false;
    } else if (cond > 1e8) {
        diag.add_warning("Matrix is ill-conditioned (condition number: " + 
                        std::to_string(cond) + ")");
    }
    
    return true;
}

// Robust covariance regularization
inline void regularize_covariance_adaptive(Mat& cov, const DiagnosticInfo& diag) {
    double reg_param = 1e-8;
    
    // Increase regularization if numerical issues detected
    if (diag.numerical_issues) {
        reg_param = 1e-6;
    }
    
    // Check minimum eigenvalue
    Vec eigenvals = arma::eig_sym(cov);
    double min_eigenval = eigenvals.min();
    
    if (min_eigenval < reg_param) {
        // Add regularization to make positive definite
        cov.diag() += std::abs(min_eigenval) + reg_param;
    } else {
        cov.diag() += reg_param;
    }
}

// Fast computation of X'WX for diagonal weight matrix
inline Mat fast_weighted_crossprod(const Mat& X, const Vec& weights) {
    Mat X_weighted = X.each_col() % arma::sqrt(weights);
    return X_weighted.t() * X_weighted;
}

// Parallel matrix-vector multiplication
inline Vec parallel_matvec(const Mat& A, const Vec& x) {
    int n = A.n_rows;
    Vec result(n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        result(i) = arma::dot(A.row(i), x);
    }
    
    return result;
}

// Efficient trace computation
inline double efficient_trace(const Mat& A) {
    return arma::sum(A.diag());
}

// Woodbury matrix identity for efficient updates
inline Mat woodbury_update(const Mat& A_inv, const Mat& U, const Mat& V) {
    // (A + UV')^{-1} = A^{-1} - A^{-1}U(I + V'A^{-1}U)^{-1}V'A^{-1}
    Mat VA_inv = V.t() * A_inv;
    Mat I_plus_VAU = arma::eye(U.n_cols, U.n_cols) + VA_inv * U;
    Mat I_plus_VAU_inv = arma::inv(I_plus_VAU);
    
    return A_inv - A_inv * U * I_plus_VAU_inv * VA_inv;
}

// QR decomposition for numerical stability
inline void stable_qr_decomp(const Mat& X, Mat& Q, Mat& R) {
    arma::qr_econ(Q, R, X);
}

// Cholesky decomposition with error handling
inline Mat safe_chol(const Mat& A, bool& success) {
    Mat L;
    success = arma::chol(L, A, "lower");
    
    if (!success) {
        // Try with regularization
        Mat A_reg = A;
        A_reg.diag() += 1e-8;
        success = arma::chol(L, A_reg, "lower");
    }
    
    return L;
}

} // namespace gwmvt

#endif // GWMVT_CORE_ALGEBRA_H