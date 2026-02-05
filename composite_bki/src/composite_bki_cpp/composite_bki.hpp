#ifndef COMPOSITE_BKI_HPP
#define COMPOSITE_BKI_HPP

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace composite_bki {

// Forward declarations
struct Point2D {
    float x, y;
    Point2D() : x(0), y(0) {}
    Point2D(float x_, float y_) : x(x_), y(y_) {}
};

struct Point3D {
    float x, y, z;
    Point3D() : x(0), y(0), z(0) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    float distance(const Point3D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

struct Polygon {
    std::vector<Point2D> points;
    
    bool contains(const Point2D& p) const;
    float distance(const Point2D& p) const;
    float boundaryDistance(const Point2D& p) const;
};

struct OSMData {
    std::map<int, std::vector<Polygon>> geometries; // class_idx -> polygons
};

// Configuration for label mappings
struct Config {
    std::map<int, std::string> labels;
    std::vector<std::vector<float>> confusion_matrix;
    std::map<int, int> label_to_matrix_idx;
    std::map<std::string, int> osm_class_map;
    std::map<int, float> height_filter_map; // osm_class_idx -> sigma
    std::vector<std::string> osm_categories;
    std::map<int, int> raw_to_dense;
    std::map<int, int> dense_to_raw;
    int num_total_classes;
};

class SemanticBKI {
public:
    SemanticBKI(const Config& config, 
                const OSMData& osm_data,
                float l_scale = 3.0f,
                float sigma_0 = 1.0f,
                float prior_delta = 5.0f,
                float height_sigma = 0.3f,
                bool use_semantic_kernel = true,
                bool use_spatial_kernel = true,
                int num_threads = -1);  // -1 = auto-detect
    
    // Get OSM prior probabilities for a point
    std::vector<float> getOSMPrior(float x, float y) const;
    
    // Get semantic kernel value
    float getSemanticKernel(int matrix_idx, const std::vector<float>& m_i) const;
    
    // Compute spatial kernel (vectorized version)
    std::vector<float> computeSpatialKernelVectorized(const std::vector<float>& dists) const;
    
    // Getters
    float getLScale() const { return l_scale_; }
    float getSigma0() const { return sigma_0_; }
    float getHeightSigma() const { return height_sigma_; }
    int getNumThreads() const { return num_threads_; }
    bool getUseSpatialKernel() const { return use_spatial_kernel_; }
    bool getUseSemanticKernel() const { return use_semantic_kernel_; }
    
private:
    std::vector<std::vector<float>> confusion_matrix_;
    int K_pred_;
    int K_prior_;
    std::map<std::string, int> class_map_;
    OSMData osm_data_;
    
    float l_scale_;
    float sigma_0_;
    float delta_;
    float height_sigma_;
    float epsilon_;
    bool use_semantic_kernel_;
    bool use_spatial_kernel_;
    int num_threads_;
    
    // Helper functions
    float computeDistanceToClass(float x, float y, int class_idx) const;
};

// Main processing functions
OSMData loadOSMBinary(const std::string& filename,
                      const std::map<std::string, int>& osm_class_map,
                      const std::vector<std::string>& osm_categories);

std::vector<float> computeSemanticWeights(
    const std::vector<Point3D>& points,
    const std::vector<uint32_t>& input_labels,
    const SemanticBKI& bki,
    const Config& config,
    std::vector<int>& dense_labels);

std::vector<uint32_t> runBKIUpdate(
    const std::vector<Point3D>& points,
    const std::vector<uint32_t>& input_labels,
    const std::vector<float>& k_sem_all,
    const std::vector<int>& dense_labels,
    const SemanticBKI& bki,
    const Config& config,
    float alpha_0 = 0.01f);

// Metrics computation
struct Metrics {
    float accuracy;
    float miou;
    std::map<int, float> per_class_iou;
    std::map<int, float> per_class_acc;
};

Metrics computeMetrics(
    const std::vector<uint32_t>& pred_labels,
    const std::vector<uint32_t>& gt_labels);

// Configuration loading
Config loadConfigFromYAML(const std::string& config_path);

} // namespace composite_bki

#endif // COMPOSITE_BKI_HPP
