#include "composite_bki.hpp"
#include "yaml_parser.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>

namespace composite_bki {

// OSM Binary Loader
OSMData loadOSMBinary(const std::string& filename,
                      const std::map<std::string, int>& osm_class_map,
                      const std::vector<std::string>& osm_categories) {
    OSMData data;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OSM file: " + filename);
    }

    if (osm_categories.empty()) {
        throw std::runtime_error("OSM categories missing from config.");
    }

    // Categories in order for binary file layout (from config)
    const std::vector<std::string>& categories = osm_categories;

    for (const auto& cat : categories) {
        uint32_t num_items;
        file.read(reinterpret_cast<char*>(&num_items), sizeof(uint32_t));
        
        if (!file.good()) break;

        auto class_it = osm_class_map.find(cat);
        bool has_class = (class_it != osm_class_map.end());
        if (!has_class) {
            std::cerr << "OSM class map missing category: " << cat
                      << ". Skipping assignment for this category." << std::endl;
        }
        int class_idx = has_class ? class_it->second : -1;
        
        for (uint32_t i = 0; i < num_items; i++) {
            uint32_t n_pts;
            file.read(reinterpret_cast<char*>(&n_pts), sizeof(uint32_t));
            
            if (!file.good()) break;
            
            Polygon poly;
            poly.points.reserve(n_pts);
            
            for (uint32_t j = 0; j < n_pts; j++) {
                float x, y;
                file.read(reinterpret_cast<char*>(&x), sizeof(float));
                file.read(reinterpret_cast<char*>(&y), sizeof(float));
                poly.points.push_back(Point2D(x, y));
            }
            
            if (has_class) {
                data.geometries[class_idx].push_back(poly);
            }
        }
    }
    
    file.close();
    return data;
}

// Metrics computation
Metrics computeMetrics(
    const std::vector<uint32_t>& pred_labels,
    const std::vector<uint32_t>& gt_labels) {

    Metrics metrics;

    std::map<int, int> intersection;
    std::map<int, int> union_map;
    std::map<int, int> correct;
    std::map<int, int> total;

    int total_correct = 0;
    int total_valid = 0;

    std::set<int> unique_classes;
    for (auto label : gt_labels) {
        if (label != 0) unique_classes.insert(label);
    }

    for (int cls : unique_classes) {
        int inter = 0, uni = 0, count = 0;

        for (size_t i = 0; i < gt_labels.size(); i++) {
            bool gt_match = (gt_labels[i] == cls);
            bool pred_match = (pred_labels[i] == cls);

            if (gt_match && pred_match) inter++;
            if (gt_match || pred_match) uni++;
            if (gt_match) count++;
        }

        intersection[cls] = inter;
        union_map[cls] = uni;
        correct[cls] = inter;
        total[cls] = count;

        total_correct += inter;
        total_valid += count;
    }

    // Compute per-class IoU and accuracy
    std::vector<float> iou_list;
    for (const auto& kv : intersection) {
        int cls = kv.first;
        if (union_map[cls] > 0) {
            float iou = static_cast<float>(intersection[cls]) / union_map[cls];
            metrics.per_class_iou[cls] = iou;
            iou_list.push_back(iou);
        }

        if (total[cls] > 0) {
            metrics.per_class_acc[cls] = static_cast<float>(correct[cls]) / total[cls];
        }
    }

    metrics.miou = iou_list.empty() ? 0.0f :
        std::accumulate(iou_list.begin(), iou_list.end(), 0.0f) / iou_list.size();
    metrics.accuracy = total_valid > 0 ?
        static_cast<float>(total_correct) / total_valid : 0.0f;

    return metrics;
}

// Load configuration from YAML file
Config loadConfigFromYAML(const std::string& config_path) {
    Config config;

    try {
        yaml_parser::YAMLNode yaml;
        yaml.parseFile(config_path);

        // Load labels
        config.labels = yaml.getLabels();

        // Load confusion matrix
        config.confusion_matrix = yaml.getConfusionMatrix();

        // Load label to matrix index mapping
        config.label_to_matrix_idx = yaml.getLabelToMatrixIdx();

        // Load OSM class map
        config.osm_class_map = yaml.getOSMClassMap();
        config.osm_categories = yaml.getOSMCategories();

        // Load OSM height filter and convert to int indices
        auto height_filter_str = yaml.getOSMHeightFilter();
        for (const auto& kv : height_filter_str) {
            auto it = config.osm_class_map.find(kv.first);
            if (it != config.osm_class_map.end()) {
                config.height_filter_map[it->second] = kv.second;
            } else {
                std::cerr << "Warning: osm_height_filter class '" << kv.first 
                          << "' not found in osm_class_map." << std::endl;
            }
        }

        // Build dense mappings
        std::vector<int> all_classes;
        for (const auto& kv : config.labels) {
            all_classes.push_back(kv.first);
        }

        for (size_t i = 0; i < all_classes.size(); i++) {
            config.raw_to_dense[all_classes[i]] = i;
            config.dense_to_raw[i] = all_classes[i];
        }

        config.num_total_classes = all_classes.size();

        std::cout << "Loaded config from: " << config_path << std::endl;
        std::cout << "  Labels: " << config.labels.size() << std::endl;
        std::cout << "  Confusion matrix: " << config.confusion_matrix.size()
                  << "x" << (config.confusion_matrix.empty() ? 0 : config.confusion_matrix[0].size()) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading config from " << config_path << ": " << e.what() << std::endl;
        throw;
    }

    return config;
}

} // namespace composite_bki
