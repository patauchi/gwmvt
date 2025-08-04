#ifndef GWMVT_UTILS_PROGRESS_H
#define GWMVT_UTILS_PROGRESS_H

#include "../core/types.h"
#include <Rcpp.h>
#include <chrono>
#include <string>
#include <atomic>

namespace gwmvt {

// Enhanced progress reporter with time estimation
class TimedProgressReporter : public ProgressReporter {
private:
    int last_percent_ = -1;
    std::string current_step_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point step_start_time_;
    bool show_time_estimate_ = true;
    
public:
    TimedProgressReporter(bool show_time_estimate = true) 
        : show_time_estimate_(show_time_estimate) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    void report(int current, int total) override {
        int percent = (100 * current) / total;
        if (percent != last_percent_) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - step_start_time_).count();
            
            Rcpp::Rcout << "\r" << current_step_ << " Progress: " << percent << "%";
            
            if (show_time_estimate_ && percent > 0) {
                // Estimate remaining time
                double seconds_per_percent = static_cast<double>(elapsed) / percent;
                int remaining_seconds = static_cast<int>(seconds_per_percent * (100 - percent));
                
                if (remaining_seconds > 0) {
                    Rcpp::Rcout << " (ETA: ";
                    if (remaining_seconds > 3600) {
                        Rcpp::Rcout << remaining_seconds / 3600 << "h ";
                        remaining_seconds %= 3600;
                    }
                    if (remaining_seconds > 60) {
                        Rcpp::Rcout << remaining_seconds / 60 << "m ";
                        remaining_seconds %= 60;
                    }
                    Rcpp::Rcout << remaining_seconds << "s)";
                }
            }
            
            Rcpp::Rcout << std::flush;
            last_percent_ = percent;
        }
    }
    
    void report_step(const std::string& step) override {
        current_step_ = step;
        step_start_time_ = std::chrono::steady_clock::now();
        last_percent_ = -1;
        Rcpp::Rcout << "\n" << step << std::endl;
    }
    
    void finish() override {
        auto end_time = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time_).count();
        
        Rcpp::Rcout << "\r" << current_step_ << " Progress: 100%";
        
        if (show_time_estimate_) {
            Rcpp::Rcout << " (Total time: ";
            if (total_elapsed > 3600) {
                Rcpp::Rcout << total_elapsed / 3600 << "h ";
                total_elapsed %= 3600;
            }
            if (total_elapsed > 60) {
                Rcpp::Rcout << total_elapsed / 60 << "m ";
                total_elapsed %= 60;
            }
            Rcpp::Rcout << total_elapsed << "s)";
        }
        
        Rcpp::Rcout << "\n" << std::flush;
    }
};

// Progress reporter for batch processing
class BatchProgressReporter : public ProgressReporter {
private:
    int batch_size_;
    int current_batch_ = 0;
    int total_batches_;
    std::string current_step_;
    
public:
    BatchProgressReporter(int total_items, int batch_size) 
        : batch_size_(batch_size) {
        total_batches_ = (total_items + batch_size - 1) / batch_size;
    }
    
    void report(int current, int total) override {
        int current_batch = current / batch_size_;
        if (current_batch != current_batch_) {
            current_batch_ = current_batch;
            Rcpp::Rcout << "\r" << current_step_ 
                        << " Batch " << (current_batch_ + 1) 
                        << "/" << total_batches_ 
                        << " (" << (100 * current / total) << "%)" 
                        << std::flush;
        }
    }
    
    void report_step(const std::string& step) override {
        current_step_ = step;
        current_batch_ = 0;
        Rcpp::Rcout << "\n" << step << std::endl;
    }
    
    void finish() override {
        Rcpp::Rcout << "\r" << current_step_ 
                    << " Batch " << total_batches_ 
                    << "/" << total_batches_ 
                    << " (100%)\n" 
                    << std::flush;
    }
};

// Thread-safe progress counter for parallel operations
class ParallelProgressCounter {
private:
    std::atomic<int> counter_{0};
    int total_;
    ProgressReporter* reporter_;
    int report_interval_;
    
public:
    ParallelProgressCounter(int total, ProgressReporter* reporter, int report_interval = 100)
        : total_(total), reporter_(reporter), report_interval_(report_interval) {}
    
    void increment() {
        int current = counter_.fetch_add(1) + 1;
        if (current % report_interval_ == 0 || current == total_) {
            reporter_->report(current, total_);
        }
    }
    
    int get_current() const {
        return counter_.load();
    }
    
    double get_progress() const {
        return static_cast<double>(counter_.load()) / total_;
    }
};

// Factory for creating appropriate progress reporter
inline std::unique_ptr<ProgressReporter> create_progress_reporter(const GWConfig& config) {
    if (!config.verbose || !config.show_progress) {
        return std::make_unique<NullProgress>();
    }
    
    if (config.batch_size > 0) {
        // Batch processing
        return std::make_unique<BatchProgressReporter>(0, config.batch_size);
    } else {
        // Regular progress with time estimation
        return std::make_unique<TimedProgressReporter>(true);
    }
}

} // namespace gwmvt

#endif // GWMVT_UTILS_PROGRESS_H