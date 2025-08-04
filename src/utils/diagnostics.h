#ifndef GWMVT_UTILS_DIAGNOSTICS_H
#define GWMVT_UTILS_DIAGNOSTICS_H

#include "../core/types.h"
#include <Rcpp.h>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace gwmvt {

// Diagnostic level enumeration
enum class DiagnosticLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

// Diagnostic entry
struct DiagnosticEntry {
    DiagnosticLevel level;
    std::string message;
    std::string location;
    std::chrono::system_clock::time_point timestamp;
    
    DiagnosticEntry(DiagnosticLevel lvl, const std::string& msg, const std::string& loc = "")
        : level(lvl), message(msg), location(loc), timestamp(std::chrono::system_clock::now()) {}
};

// Enhanced diagnostic collector
class DiagnosticCollector {
private:
    std::vector<DiagnosticEntry> entries_;
    DiagnosticLevel min_level_ = DiagnosticLevel::INFO;
    bool print_immediate_ = false;
    
public:
    // Set minimum diagnostic level
    void set_level(DiagnosticLevel level) {
        min_level_ = level;
    }
    
    // Enable/disable immediate printing
    void set_immediate_print(bool immediate) {
        print_immediate_ = immediate;
    }
    
    // Add diagnostic entry
    void add(DiagnosticLevel level, const std::string& message, const std::string& location = "") {
        if (level >= min_level_) {
            entries_.emplace_back(level, message, location);
            
            if (print_immediate_) {
                print_entry(entries_.back());
            }
        }
    }
    
    // Convenience methods
    void debug(const std::string& message, const std::string& location = "") {
        add(DiagnosticLevel::DEBUG, message, location);
    }
    
    void info(const std::string& message, const std::string& location = "") {
        add(DiagnosticLevel::INFO, message, location);
    }
    
    void warning(const std::string& message, const std::string& location = "") {
        add(DiagnosticLevel::WARNING, message, location);
    }
    
    void error(const std::string& message, const std::string& location = "") {
        add(DiagnosticLevel::ERROR, message, location);
    }
    
    // Get entries by level
    std::vector<DiagnosticEntry> get_entries(DiagnosticLevel level) const {
        std::vector<DiagnosticEntry> result;
        for (const auto& entry : entries_) {
            if (entry.level == level) {
                result.push_back(entry);
            }
        }
        return result;
    }
    
    // Get all entries
    const std::vector<DiagnosticEntry>& get_all_entries() const {
        return entries_;
    }
    
    // Clear entries
    void clear() {
        entries_.clear();
    }
    
    // Check if there are entries at or above a level
    bool has_level(DiagnosticLevel level) const {
        for (const auto& entry : entries_) {
            if (entry.level >= level) {
                return true;
            }
        }
        return false;
    }
    
    // Print all entries
    void print_all() const {
        for (const auto& entry : entries_) {
            print_entry(entry);
        }
    }
    
    // Convert to R list
    Rcpp::List to_R_list() const {
        Rcpp::List result;
        
        std::vector<std::string> levels;
        std::vector<std::string> messages;
        std::vector<std::string> locations;
        std::vector<std::string> timestamps;
        
        for (const auto& entry : entries_) {
            levels.push_back(level_to_string(entry.level));
            messages.push_back(entry.message);
            locations.push_back(entry.location);
            timestamps.push_back(format_timestamp(entry.timestamp));
        }
        
        result["level"] = levels;
        result["message"] = messages;
        result["location"] = locations;
        result["timestamp"] = timestamps;
        
        return result;
    }
    
private:
    // Print a single entry
    void print_entry(const DiagnosticEntry& entry) const {
        std::string prefix;
        switch (entry.level) {
            case DiagnosticLevel::DEBUG:
                prefix = "[DEBUG]";
                break;
            case DiagnosticLevel::INFO:
                prefix = "[INFO]";
                break;
            case DiagnosticLevel::WARNING:
                prefix = "[WARNING]";
                break;
            case DiagnosticLevel::ERROR:
                prefix = "[ERROR]";
                break;
        }
        
        Rcpp::Rcout << prefix << " " << entry.message;
        if (!entry.location.empty()) {
            Rcpp::Rcout << " (at " << entry.location << ")";
        }
        Rcpp::Rcout << std::endl;
    }
    
    // Convert level to string
    std::string level_to_string(DiagnosticLevel level) const {
        switch (level) {
            case DiagnosticLevel::DEBUG: return "DEBUG";
            case DiagnosticLevel::INFO: return "INFO";
            case DiagnosticLevel::WARNING: return "WARNING";
            case DiagnosticLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
    
    // Format timestamp
    std::string format_timestamp(const std::chrono::system_clock::time_point& tp) const {
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

// Numerical diagnostics
class NumericalDiagnostics {
public:
    // Check matrix condition number
    static bool check_condition_number(const Mat& A, DiagnosticInfo& diag, double max_cond = 1e10) {
        double cond = arma::cond(A);
        
        if (cond > max_cond) {
            diag.add_error("Matrix is severely ill-conditioned (condition number: " + 
                          std::to_string(cond) + ")");
            diag.numerical_issues = true;
            return false;
        } else if (cond > max_cond / 100) {
            diag.add_warning("Matrix is ill-conditioned (condition number: " + 
                            std::to_string(cond) + ")");
            diag.numerical_issues = true;
        }
        
        return true;
    }
    
    // Check for near-singular matrix
    static bool check_singularity(const Mat& A, DiagnosticInfo& diag, double tol = 1e-10) {
        Vec eigenvals = arma::eig_sym(A);
        double min_eigenval = eigenvals.min();
        
        if (std::abs(min_eigenval) < tol) {
            diag.add_error("Matrix is near-singular (minimum eigenvalue: " + 
                          std::to_string(min_eigenval) + ")");
            diag.numerical_issues = true;
            return false;
        }
        
        return true;
    }
    
    // Check for numerical overflow/underflow
    static bool check_numerical_range(const Mat& A, DiagnosticInfo& diag) {
        double max_val = A.max();
        double min_val = A.min();
        
        if (max_val > 1e100) {
            diag.add_warning("Matrix contains very large values (max: " + 
                            std::to_string(max_val) + ")");
            diag.numerical_issues = true;
        }
        
        if (std::abs(min_val) < 1e-100 && min_val != 0) {
            diag.add_warning("Matrix contains very small values (min: " + 
                            std::to_string(min_val) + ")");
            diag.numerical_issues = true;
        }
        
        return !diag.numerical_issues;
    }
    
    // Check covariance matrix properties
    static bool validate_covariance_matrix(const Mat& cov, DiagnosticInfo& diag) {
        // Check symmetry
        double sym_error = arma::norm(cov - cov.t(), "fro") / arma::norm(cov, "fro");
        if (sym_error > 1e-10) {
            diag.add_warning("Covariance matrix is not perfectly symmetric (error: " + 
                            std::to_string(sym_error) + ")");
        }
        
        // Check positive semi-definiteness
        Vec eigenvals = arma::eig_sym(cov);
        if (eigenvals.min() < -1e-10) {
            diag.add_error("Covariance matrix is not positive semi-definite");
            diag.numerical_issues = true;
            return false;
        }
        
        // Check condition number
        return check_condition_number(cov, diag);
    }
};

// Performance diagnostics
class PerformanceDiagnostics {
private:
    struct TimingEntry {
        std::string name;
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point end;
        bool completed;
    };
    
    std::vector<TimingEntry> timings_;
    
public:
    // Start timing a section
    void start_timing(const std::string& name) {
        timings_.push_back({name, std::chrono::steady_clock::now(), 
                           std::chrono::steady_clock::time_point(), false});
    }
    
    // End timing a section
    void end_timing(const std::string& name) {
        for (auto& entry : timings_) {
            if (entry.name == name && !entry.completed) {
                entry.end = std::chrono::steady_clock::now();
                entry.completed = true;
                break;
            }
        }
    }
    
    // Get timing summary
    Rcpp::List get_timing_summary() const {
        std::vector<std::string> names;
        std::vector<double> durations;
        
        for (const auto& entry : timings_) {
            if (entry.completed) {
                names.push_back(entry.name);
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    entry.end - entry.start).count();
                durations.push_back(duration / 1000.0);  // Convert to seconds
            }
        }
        
        return Rcpp::List::create(
            Rcpp::Named("section") = names,
            Rcpp::Named("duration_sec") = durations
        );
    }
    
    // Clear timings
    void clear() {
        timings_.clear();
    }
};

// Convergence diagnostics
class ConvergenceDiagnostics {
public:
    // Check if algorithm has converged
    static bool check_convergence(const Vec& old_params, const Vec& new_params, 
                                 double tol, ConvergenceInfo& info) {
        double max_change = arma::max(arma::abs(new_params - old_params));
        double rel_change = max_change / (arma::norm(old_params, 2) + 1e-10);
        
        info.error = max_change;
        
        if (max_change < tol) {
            info.converged = true;
            info.message = "Converged (absolute tolerance)";
            return true;
        }
        
        if (rel_change < tol * 0.1) {
            info.converged = true;
            info.message = "Converged (relative tolerance)";
            return true;
        }
        
        return false;
    }
    
    // Track convergence history
    static void update_convergence_history(std::vector<double>& history, 
                                         double current_value,
                                         DiagnosticInfo& diag) {
        history.push_back(current_value);
        
        // Check for oscillation
        if (history.size() > 5) {
            bool oscillating = true;
            for (size_t i = history.size() - 4; i < history.size() - 1; ++i) {
                if ((history[i] - history[i-1]) * (history[i+1] - history[i]) >= 0) {
                    oscillating = false;
                    break;
                }
            }
            
            if (oscillating) {
                diag.add_warning("Convergence appears to be oscillating");
                diag.convergence_issues = true;
            }
        }
        
        // Check for stagnation
        if (history.size() > 10) {
            double recent_change = std::abs(history.back() - history[history.size() - 10]);
            if (recent_change < 1e-10) {
                diag.add_warning("Convergence has stagnated");
                diag.convergence_issues = true;
            }
        }
    }
};

} // namespace gwmvt

#endif // GWMVT_UTILS_DIAGNOSTICS_H