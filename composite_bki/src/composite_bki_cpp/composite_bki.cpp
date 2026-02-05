#include "composite_bki.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <cstring>
#include <set>
#include <numeric>

namespace composite_bki {

// Polygon methods
bool Polygon::contains(const Point2D& p) const {
    if (points.size() < 3) return false;
    
    // Ray casting algorithm
    bool inside = false;
    for (size_t i = 0, j = points.size() - 1; i < points.size(); j = i++) {
        if ((points[i].y > p.y) != (points[j].y > p.y) &&
            p.x < (points[j].x - points[i].x) * (p.y - points[i].y) / 
                  (points[j].y - points[i].y) + points[i].x) {
            inside = !inside;
        }
    }
    return inside;
}

float Polygon::distance(const Point2D& p) const {
    if (points.empty()) return std::numeric_limits<float>::max();
    
    float min_dist = std::numeric_limits<float>::max();
    
    // Go through each edge and find the shortest distance. Then
    // return the shortest distance to all edges, which is the shortest
    // distance to the polygon.
    for (size_t i = 0; i < points.size(); i++) {
        size_t j = (i + 1) % points.size();
        const Point2D& p1 = points[i];
        const Point2D& p2 = points[j];
        
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float len_sq = dx*dx + dy*dy;
        
        if (len_sq < 1e-10f) {
            // Points are the same
            float d = std::sqrt((p.x - p1.x)*(p.x - p1.x) + (p.y - p1.y)*(p.y - p1.y));
            min_dist = std::min(min_dist, d);
            continue;
        }
        
        float t = std::max(0.0f, std::min(1.0f, 
            ((p.x - p1.x) * dx + (p.y - p1.y) * dy) / len_sq));
        
        float proj_x = p1.x + t * dx;
        float proj_y = p1.y + t * dy;
        
        float d = std::sqrt((p.x - proj_x)*(p.x - proj_x) + (p.y - proj_y)*(p.y - proj_y));
        min_dist = std::min(min_dist, d);
    }
    
    return min_dist;
}

float Polygon::boundaryDistance(const Point2D& p) const {
    return distance(p);
}

// SemanticBKI Implementation
SemanticBKI::SemanticBKI(const Config& config,
                         const OSMData& osm_data,
                         float l_scale,
                         float sigma_0,
                         float prior_delta,
                         float height_sigma,
                         bool use_semantic_kernel,
                         bool use_spatial_kernel,
                         int num_threads)
    : confusion_matrix_(config.confusion_matrix),
      osm_data_(osm_data),
      l_scale_(l_scale),
      sigma_0_(sigma_0),
      delta_(prior_delta),
      height_sigma_(height_sigma),
      epsilon_(1e-6f),
      use_semantic_kernel_(use_semantic_kernel),
      use_spatial_kernel_(use_spatial_kernel),
      num_threads_(num_threads) {
    
    K_pred_ = confusion_matrix_.size();
    K_prior_ = confusion_matrix_.empty() ? 0 : confusion_matrix_[0].size();
    class_map_ = config.osm_class_map;
    
    // Print kernel configuration
    std::cout << "Kernel Configuration:" << std::endl;
    std::cout << "  - Semantic Kernel: " << (use_semantic_kernel_ ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "  - Spatial Kernel: " << (use_spatial_kernel_ ? "ENABLED" : "DISABLED") << std::endl;
    
    // Set number of OpenMP threads
#ifdef _OPENMP
    if (num_threads_ < 0) {
        num_threads_ = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads_);
    std::cout << "Using " << num_threads_ << " OpenMP threads" << std::endl;
#else
    num_threads_ = 1;
    std::cout << "OpenMP not available, using single thread" << std::endl;
#endif
}

float SemanticBKI::computeDistanceToClass(float x, float y, int class_idx) const {
    Point2D p(x, y);
    
    auto it = osm_data_.geometries.find(class_idx);
    if (it == osm_data_.geometries.end() || it->second.empty()) {
        return 50.0f; // Default distance if no geometries
    }
    
    float min_dist = std::numeric_limits<float>::max();
    
    for (const auto& poly : it->second) {
        float dist = poly.distance(p);
        
        // Check if point is inside polygon
        if (dist < 1e-6f && poly.contains(p)) {
            dist = -1.0f * poly.boundaryDistance(p);
        }
        
        min_dist = std::min(min_dist, dist);
    }
    
    return min_dist;
}

std::vector<float> SemanticBKI::getOSMPrior(float x, float y) const {
    std::vector<float> s_scores(K_prior_, 0.0f);
    
    for (int k = 0; k < K_prior_; k++) {
        float dist = computeDistanceToClass(x, y, k);
        s_scores[k] = 1.0f / (1.0f + std::exp((dist / delta_) - 4.6)); // -4.6 is -ln(0.01); ensures probability is 1 if in polygon (i.e. dist <= 0).
    }
    
    // Normalize
    float sum = std::accumulate(s_scores.begin(), s_scores.end(), 0.0f);
    std::vector<float> m_i(K_prior_);
    for (int k = 0; k < K_prior_; k++) {
        m_i[k] = s_scores[k] / (sum + epsilon_);
    }
    
    return m_i;
}

float SemanticBKI::getSemanticKernel(int matrix_idx, const std::vector<float>& m_i) const {
    if (matrix_idx < 0 || matrix_idx >= K_pred_) return 1.0f;
    
    // Get the confidence for the OSM prior by assuming the highest probability class.
    float c_xi = *std::max_element(m_i.begin(), m_i.end());
    
    // Compute expected_obs = M @ m_i
    std::vector<float> expected_obs(K_pred_, 0.0f);
    for (int i = 0; i < K_pred_; i++) {
        for (int j = 0; j < K_prior_; j++) {
            expected_obs[i] += confusion_matrix_[i][j] * m_i[j];
        }
    }
    
    float numerator = expected_obs[matrix_idx];
    float denominator = *std::max_element(expected_obs.begin(), expected_obs.end()) + epsilon_;
    float s_i = numerator / denominator;
    
    return (1.0f - c_xi) + (c_xi * s_i);
}

std::vector<float> SemanticBKI::computeSpatialKernelVectorized(const std::vector<float>& dists) const {
    std::vector<float> k_val(dists.size());
    
    for (size_t i = 0; i < dists.size(); i++) {
        float xi = dists[i] / l_scale_;
        
        if (xi < 1.0f) {
            float term1 = (1.0f/3.0f) * (2.0f + std::cos(2.0f * M_PI * xi)) * (1.0f - xi);
            float term2 = (1.0f/(2.0f * M_PI)) * std::sin(2.0f * M_PI * xi);
            k_val[i] = sigma_0_ * (term1 + term2);
        } else {
            k_val[i] = 0.0f;
        }
    }
    
    return k_val;
}

// Compute semantic weights (PARALLELIZED)
// NOTE: This function also computes dense_labels which are needed for voting.
// The k_sem_all values are computed but may not be used if location-based semantic
// kernel is enabled (see runBKIUpdate for location-based computation).
std::vector<float> computeSemanticWeights(
    const std::vector<Point3D>& points,
    const std::vector<uint32_t>& input_labels,
    const SemanticBKI& bki,
    const Config& config,
    std::vector<int>& dense_labels) {
    
    size_t n = points.size();
    std::vector<float> k_sem_all(n, 1.0f);
    dense_labels.resize(n, -1);
    
    std::cout << "Computing semantic weights and dense labels (parallelized)..." << std::endl;
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1000)
#endif
    for (size_t i = 0; i < n; i++) {
        uint32_t raw_label = input_labels[i];
        
        // Map to dense label
        auto it_dense = config.raw_to_dense.find(raw_label);
        if (it_dense != config.raw_to_dense.end()) {
            dense_labels[i] = it_dense->second;
        }
        
        // Get matrix index
        // TODO: This can be extended to work with confidence-weighted semantic labels.
        //       For now, we just use the raw label to find the matrix index. i.e. this assumes
        //       prediction vector = {0, 1, 0, 0, 0, 0, 0, 0}
        //                  vs
        //       prediction vector = {0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0}
        //       This is a limitation of the current implementation and should be addressed in the future.
        auto it_matrix = config.label_to_matrix_idx.find(raw_label);
        if (it_matrix != config.label_to_matrix_idx.end()) {
            int matrix_idx = it_matrix->second;
            std::vector<float> m_i = bki.getOSMPrior(points[i].x, points[i].y);
            k_sem_all[i] = bki.getSemanticKernel(matrix_idx, m_i);
        }
        
#ifdef _OPENMP
        if (i % 10000 == 0 && i > 0 && omp_get_thread_num() == 0) {
#else
        if (i % 10000 == 0 && i > 0) {
#endif
            std::cout << "  Pre-computed " << i << " semantic weights..." << std::endl;
        }
    }
    
    return k_sem_all;
}

// Parallel neighbor search (OPTIMIZED)
std::vector<std::vector<size_t>> findNeighbors(
    const std::vector<Point3D>& points,
    float radius) {
    
    size_t n = points.size();
    std::vector<std::vector<size_t>> neighbors(n);
    float radius_sq = radius * radius;  // Use squared distance to avoid sqrt
    
    std::cout << "Finding neighbors (parallelized, radius=" << radius << ")..." << std::endl;
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 100)
#endif
    for (size_t i = 0; i < n; i++) {
        std::vector<size_t> local_neighbors;
        local_neighbors.reserve(100);  // Reserve space
        
        const Point3D& pi = points[i];
        
        for (size_t j = 0; j < n; j++) {
            const Point3D& pj = points[j];
            // Use squared distance to avoid expensive sqrt
            float dx = pi.x - pj.x;
            float dy = pi.y - pj.y;
            float dz = pi.z - pj.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (dist_sq <= radius_sq) {
                local_neighbors.push_back(j);
            }
        }
        
        neighbors[i] = std::move(local_neighbors);
        
#ifdef _OPENMP
        if (i % 5000 == 0 && i > 0 && omp_get_thread_num() == 0) {
#else
        if (i % 5000 == 0 && i > 0) {
#endif
            std::cout << "  Found neighbors for " << i << " points..." << std::endl;
        }
    }
    
    return neighbors;
}

// Run BKI Update (PARALLELIZED)
// This function implements the Bayesian Kernel Inference update step.
// For each point, it:
// 1. Finds neighbors within radius l_scale
// 2. Computes spatial kernel k_sp based on distance
// 3. Computes semantic kernel k_sem on-the-fly based on target location and neighbor's class
//    (Location-based approach: uses target point's OSM prior, not neighbor's label-based pre-computed value)
// 4. Combines weights: combined_weight = k_sp * k_sem
// 5. Accumulates weights per class and selects best class
std::vector<uint32_t> runBKIUpdate(
    const std::vector<Point3D>& points,
    const std::vector<uint32_t>& input_labels,
    const std::vector<float>& k_sem_all,
    const std::vector<int>& dense_labels,
    const SemanticBKI& bki,
    const Config& config,
    float alpha_0) {
    
    size_t n = points.size();
    std::vector<uint32_t> final_labels = input_labels;
    
    // Find neighbors (parallelized inside)
    auto neighbors_list = findNeighbors(points, bki.getLScale());
    
    std::cout << "Running BKI update with spatial-semantic weighting (location-based semantic kernel)..." << std::endl;
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 500)
#endif
    for (size_t j = 0; j < n; j++) {
        std::vector<float> alpha_star(config.num_total_classes, alpha_0);
        
        // Filter valid neighbors (those with valid labels)
        const auto& neighbor_indices = neighbors_list[j];
        std::vector<size_t> valid_neighbors;
        valid_neighbors.reserve(neighbor_indices.size());
        
        for (size_t idx : neighbor_indices) {
            if (dense_labels[idx] != -1) {
                valid_neighbors.push_back(idx);
            }
        }
        
        if (valid_neighbors.empty()) {
            final_labels[j] = input_labels[j];
        } else {
            const Point3D& target_point = points[j];
            
            // Pre-compute OSM prior at target point location (used for all neighbors)
            // This is the key change: semantic kernel is based on WHERE we're predicting,
            // not where the neighbor is or what label the neighbor has
            std::vector<float> target_osm_prior;
            if (bki.getUseSemanticKernel()) {
                target_osm_prior = bki.getOSMPrior(target_point.x, target_point.y);
                
                // GEOMETRIC GATING (Smart Probabilistic Approach)
                // Modulate OSM priors based on height relative to local ground
                if (!valid_neighbors.empty()) {
                    // 1. Find local ground height (min_z)
                    float min_z = target_point.z;
                    for (size_t idx : valid_neighbors) {
                        if (points[idx].z < min_z) min_z = points[idx].z;
                    }
                    
                    float height = target_point.z - min_z;
                    
                    // 2. Apply Gaussian decay to Ground Classes
                    // If point is high above ground, probability of it being a ground class drops.
                    
                    if (!config.height_filter_map.empty()) {
                        // Use per-class configuration from YAML
                        for (const auto& kv : config.height_filter_map) {
                            int class_idx = kv.first;
                            float sigma = kv.second;
                            if (class_idx >= 0 && class_idx < (int)target_osm_prior.size()) {
                                float decay = std::exp(-(height * height) / (2.0f * sigma * sigma));
                                target_osm_prior[class_idx] *= decay;
                            }
                        }
                    } else {
                        // Fallback to legacy hardcoded behavior (Roads, Parking, Sidewalks)
                        // Uses the global height_sigma from CLI/constructor
                        float sigma = bki.getHeightSigma(); 
                        float decay = std::exp(-(height * height) / (2.0f * sigma * sigma));
                        
                        // Prior indices: 0=Roads, 1=Parking, 2=Sidewalks
                        if (target_osm_prior.size() > 0) target_osm_prior[0] *= decay;
                        if (target_osm_prior.size() > 1) target_osm_prior[1] *= decay;
                        if (target_osm_prior.size() > 2) target_osm_prior[2] *= decay;
                    }
                    
                    // 3. Re-normalize
                    float sum = 0.0f;
                    for (float v : target_osm_prior) sum += v;
                    if (sum > 1e-6f) {
                        for (float &v : target_osm_prior) v /= sum;
                    }
                }
            }
            
            // Prepare spatial kernel values (if enabled)
            std::vector<float> k_sp_values;
            if (bki.getUseSpatialKernel()) {
                // Compute distances to valid neighbors
                std::vector<float> distances;
                distances.reserve(valid_neighbors.size());
                
                for (size_t idx : valid_neighbors) {
                    const Point3D& neighbor = points[idx];
                    float dx = neighbor.x - target_point.x;
                    float dy = neighbor.y - target_point.y;
                    float dz = neighbor.z - target_point.z;
                    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    distances.push_back(dist);
                }
                
                // Compute spatial kernels (vectorized)
                k_sp_values = bki.computeSpatialKernelVectorized(distances);
            } else {
                // If spatial kernel disabled, use uniform weight of 1.0
                k_sp_values.resize(valid_neighbors.size(), 1.0f);
            }
            
            // Combine spatial and semantic weights, add to alpha_star
            for (size_t i = 0; i < valid_neighbors.size(); i++) {
                size_t idx = valid_neighbors[i];
                int dense_class = dense_labels[idx];
                
                // Compute semantic kernel on-the-fly based on:
                // 1. Target point's location (for OSM prior)
                // 2. Neighbor's class (for confusion matrix lookup)
                float k_sem = 1.0f;
                if (bki.getUseSemanticKernel()) {
                    // Get the raw label for this neighbor to find matrix index
                    auto it_raw = config.dense_to_raw.find(dense_class);
                    if (it_raw != config.dense_to_raw.end()) {
                        uint32_t raw_label = it_raw->second;
                        auto it_matrix = config.label_to_matrix_idx.find(raw_label);
                        if (it_matrix != config.label_to_matrix_idx.end()) {
                            int matrix_idx = it_matrix->second;
                            // Use target point's OSM prior, not neighbor's
                            k_sem = bki.getSemanticKernel(matrix_idx, target_osm_prior);
                        }
                    }
                }
                
                // Combine weights
                float combined_weight = k_sp_values[i] * k_sem;
                alpha_star[dense_class] += combined_weight;
            }
            
            // Find best class
            int best_idx = std::max_element(alpha_star.begin(), alpha_star.end()) - alpha_star.begin();
            auto it = config.dense_to_raw.find(best_idx);
            if (it != config.dense_to_raw.end()) {
                final_labels[j] = it->second;
            }
        }
        
#ifdef _OPENMP
        if (j % 10000 == 0 && j > 0 && omp_get_thread_num() == 0) {
#else
        if (j % 10000 == 0 && j > 0) {
#endif
            std::cout << "  Processed " << j << " points..." << std::endl;
        }
    }
    
    return final_labels;
}

} // namespace composite_bki
