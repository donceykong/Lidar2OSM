#include "continuous_bki.hpp"
#include "yaml_parser.hpp"
#include <fstream>
#include <limits>
#include <cstring>
#include <numeric>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace continuous_bki {

// --- Polygon Implementation ---
void Polygon::computeBounds() {
    if (points.empty()) {
        min_x = max_x = min_y = max_y = 0.0f;
        return;
    }
    min_x = max_x = points[0].x;
    min_y = max_y = points[0].y;
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].x < min_x) min_x = points[i].x;
        if (points[i].x > max_x) max_x = points[i].x;
        if (points[i].y < min_y) min_y = points[i].y;
        if (points[i].y > max_y) max_y = points[i].y;
    }
}

bool Polygon::contains(const Point2D& p) const {
    if (points.size() < 3) return false;
    if (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y) return false;

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
    for (size_t i = 0; i < points.size(); i++) {
        size_t j = (i + 1) % points.size();
        const Point2D& p1 = points[i];
        const Point2D& p2 = points[j];
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float len_sq = dx*dx + dy*dy;
        if (len_sq < 1e-10f) {
            float d = std::sqrt((p.x - p1.x)*(p.x - p1.x) + (p.y - p1.y)*(p.y - p1.y));
            min_dist = std::min(min_dist, d);
            continue;
        }
        float t = std::max(0.0f, std::min(1.0f, ((p.x - p1.x) * dx + (p.y - p1.y) * dy) / len_sq));
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

// --- ContinuousBKI Implementation ---

ContinuousBKI::ContinuousBKI(const Config& config,
              const OSMData& osm_data,
              float resolution,
              float l_scale,
              float sigma_0,
              float prior_delta,
              float height_sigma,
              bool use_semantic_kernel,
              bool use_spatial_kernel,
              int num_threads,
              float alpha0,
              bool seed_osm_prior,
              float osm_prior_strength,
              bool osm_fallback_in_infer)
    : config_(config),
      osm_data_(osm_data),
      resolution_(resolution),
      l_scale_(l_scale),
      sigma_0_(sigma_0),
      delta_(prior_delta),
      height_sigma_(height_sigma),
      epsilon_(1e-6f),
      use_semantic_kernel_(use_semantic_kernel),
      use_spatial_kernel_(use_spatial_kernel),
      num_threads_(num_threads),
      alpha0_(alpha0),
      seed_osm_prior_(seed_osm_prior),
      osm_prior_strength_(osm_prior_strength),
      osm_fallback_in_infer_(osm_fallback_in_infer)
{
    K_pred_ = config.confusion_matrix.size();
    K_prior_ = config.confusion_matrix.empty() ? 0 : static_cast<int>(config.confusion_matrix[0].size());

#ifdef _OPENMP
    if (num_threads_ < 0) {
        num_threads_ = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads_);
#else
    num_threads_ = 1;
#endif

    block_shards_.resize(static_cast<size_t>(num_threads_));
}

void ContinuousBKI::clear() {
    for (auto& shard : block_shards_) {
        shard.clear();
    }
}

int ContinuousBKI::getShardIndex(const BlockKey& k) const {
    BlockKeyHasher hasher;
    return static_cast<int>(hasher(k) % block_shards_.size());
}

VoxelKey ContinuousBKI::pointToKey(const Point3D& p) const {
    return VoxelKey{
        static_cast<int>(std::floor(p.x / resolution_)),
        static_cast<int>(std::floor(p.y / resolution_)),
        static_cast<int>(std::floor(p.z / resolution_))
    };
}

Point3D ContinuousBKI::keyToPoint(const VoxelKey& k) const {
    return Point3D(
        (k.x + 0.5f) * resolution_,
        (k.y + 0.5f) * resolution_,
        (k.z + 0.5f) * resolution_
    );
}

BlockKey ContinuousBKI::voxelToBlockKey(const VoxelKey& vk) const {
    return BlockKey{
        div_floor(vk.x, BLOCK_SIZE),
        div_floor(vk.y, BLOCK_SIZE),
        div_floor(vk.z, BLOCK_SIZE)
    };
}

void ContinuousBKI::voxelToLocal(const VoxelKey& vk, int& lx, int& ly, int& lz) const {
    lx = mod_floor(vk.x, BLOCK_SIZE);
    ly = mod_floor(vk.y, BLOCK_SIZE);
    lz = mod_floor(vk.z, BLOCK_SIZE);
}

Block& ContinuousBKI::getOrCreateBlock(std::unordered_map<BlockKey, Block, BlockKeyHasher>& shard_map, const BlockKey& bk) const {
    auto it = shard_map.find(bk);
    if (it != shard_map.end())
        return it->second;

    Block blk;
    const size_t total = static_cast<size_t>(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * config_.num_total_classes);
    blk.alpha.resize(total, alpha0_);

    if (seed_osm_prior_ && osm_prior_strength_ > 0.0f) {
        for (int lz = 0; lz < BLOCK_SIZE; lz++) {
            for (int ly = 0; ly < BLOCK_SIZE; ly++) {
                for (int lx = 0; lx < BLOCK_SIZE; lx++) {
                    int vx = bk.x * BLOCK_SIZE + lx;
                    int vy = bk.y * BLOCK_SIZE + ly;
                    int vz = bk.z * BLOCK_SIZE + lz;
                    Point3D center((vx + 0.5f) * resolution_, (vy + 0.5f) * resolution_, (vz + 0.5f) * resolution_);
                    initVoxelAlpha(blk, lx, ly, lz, center);
                }
            }
        }
    }

    auto inserted = shard_map.emplace(bk, std::move(blk));
    return inserted.first->second;
}

const Block* ContinuousBKI::getBlockConst(const std::unordered_map<BlockKey, Block, BlockKeyHasher>& shard_map, const BlockKey& bk) const {
    auto it = shard_map.find(bk);
    if (it == shard_map.end()) return nullptr;
    return &it->second;
}

void ContinuousBKI::initVoxelAlpha(Block& b, int lx, int ly, int lz, const Point3D& center) const {
    const int K = config_.num_total_classes;
    for (int c = 0; c < K; c++) {
        int idx = flatIndex(lx, ly, lz, c);
        b.alpha[idx] = alpha0_;
    }
    if (seed_osm_prior_ && osm_prior_strength_ > 0.0f && K_pred_ > 0 && K_prior_ > 0) {
        std::vector<float> p_pred;
        computePredPriorFromOSM(center.x, center.y, p_pred);
        if (p_pred.size() == static_cast<size_t>(K)) {
            for (int c = 0; c < K; c++) {
                int idx = flatIndex(lx, ly, lz, c);
                b.alpha[idx] += osm_prior_strength_ * p_pred[c];
            }
        }
    }
}

void ContinuousBKI::computePredPriorFromOSM(float x, float y, std::vector<float>& p_pred_out) const {
    std::vector<float> m_i(K_prior_, 0.0f);
    getOSMPrior(x, y, m_i);
    p_pred_out.resize(static_cast<size_t>(K_pred_), 0.0f);
    for (int i = 0; i < K_pred_; i++) {
        for (int j = 0; j < K_prior_; j++) {
            p_pred_out[i] += config_.confusion_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] * m_i[j];
        }
    }
    float sum = std::accumulate(p_pred_out.begin(), p_pred_out.end(), 0.0f);
    if (sum > epsilon_)
        for (float& v : p_pred_out) v /= sum;
}

float ContinuousBKI::computeSpatialKernel(float dist_sq) const {
    if (!use_spatial_kernel_) return 1.0f;

    float dist = std::sqrt(dist_sq);
    float xi = dist / l_scale_;

    if (xi < 1.0f) {
        float term1 = (1.0f/3.0f) * (2.0f + std::cos(2.0f * static_cast<float>(M_PI) * xi)) * (1.0f - xi);
        float term2 = (1.0f/(2.0f * static_cast<float>(M_PI))) * std::sin(2.0f * static_cast<float>(M_PI) * xi);
        return sigma_0_ * (term1 + term2);
    }
    return 0.0f;
}

float ContinuousBKI::computeDistanceToClass(float x, float y, int class_idx) const {
    Point2D p(x, y);
    auto it = osm_data_.geometries.find(class_idx);
    if (it == osm_data_.geometries.end() || it->second.empty()) return 50.0f;

    float min_dist = std::numeric_limits<float>::max();
    for (const auto& poly : it->second) {
        float dist_bbox_x = std::max(0.0f, std::max(poly.min_x - p.x, p.x - poly.max_x));
        float dist_bbox_y = std::max(0.0f, std::max(poly.min_y - p.y, p.y - poly.max_y));
        float dist_bbox_sq = dist_bbox_x * dist_bbox_x + dist_bbox_y * dist_bbox_y;
        if (dist_bbox_sq > min_dist * min_dist) continue;

        float dist = poly.distance(p);
        if (dist < 1e-6f && poly.contains(p)) {
            dist = -1.0f * poly.boundaryDistance(p);
        }
        min_dist = std::min(min_dist, dist);
    }
    return min_dist;
}

void ContinuousBKI::getOSMPrior(float x, float y, std::vector<float>& m_i) const {
    if (m_i.size() != static_cast<size_t>(K_prior_)) m_i.resize(static_cast<size_t>(K_prior_));

    std::vector<float> s_scores(static_cast<size_t>(K_prior_), 0.0f);
    for (int k = 0; k < K_prior_; k++) {
        float dist = computeDistanceToClass(x, y, k);
        s_scores[static_cast<size_t>(k)] = 1.0f / (1.0f + std::exp((dist / delta_) - 4.6f));
    }
    float sum = std::accumulate(s_scores.begin(), s_scores.end(), 0.0f);
    for (int k = 0; k < K_prior_; k++) {
        m_i[static_cast<size_t>(k)] = s_scores[static_cast<size_t>(k)] / (sum + epsilon_);
    }
}

float ContinuousBKI::getSemanticKernel(int matrix_idx, const std::vector<float>& m_i) const {
    if (!use_semantic_kernel_) return 1.0f;
    if (matrix_idx < 0 || matrix_idx >= K_pred_) return 1.0f;

    float c_xi = *std::max_element(m_i.begin(), m_i.end());
    std::vector<float> expected_obs(static_cast<size_t>(K_pred_), 0.0f);
    for (int i = 0; i < K_pred_; i++) {
        for (int j = 0; j < K_prior_; j++) {
            expected_obs[static_cast<size_t>(i)] += config_.confusion_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] * m_i[static_cast<size_t>(j)];
        }
    }
    float numerator = expected_obs[static_cast<size_t>(matrix_idx)];
    float denominator = *std::max_element(expected_obs.begin(), expected_obs.end()) + epsilon_;
    float s_i = numerator / denominator;
    return (1.0f - c_xi) + (c_xi * s_i);
}

void ContinuousBKI::update(const std::vector<uint32_t>& labels, const std::vector<Point3D>& points) {
    if (labels.size() != points.size()) {
        std::cerr << "Mismatch in points and labels size" << std::endl;
        return;
    }

    size_t n = points.size();
    std::vector<float> point_k_sem(n, 1.0f);

    if (use_semantic_kernel_) {
#ifdef _OPENMP
#pragma omp parallel
        {
            std::vector<float> temp_m_i(static_cast<size_t>(K_prior_));
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; i++) {
                getOSMPrior(points[i].x, points[i].y, temp_m_i);
                uint32_t raw_label = labels[i];
                auto it_matrix = config_.label_to_matrix_idx.find(raw_label);
                if (it_matrix != config_.label_to_matrix_idx.end()) {
                    point_k_sem[i] = getSemanticKernel(it_matrix->second, temp_m_i);
                }
            }
        }
#else
        std::vector<float> temp_m_i(static_cast<size_t>(K_prior_));
        for (size_t i = 0; i < n; i++) {
            getOSMPrior(points[i].x, points[i].y, temp_m_i);
            uint32_t raw_label = labels[i];
            auto it_matrix = config_.label_to_matrix_idx.find(raw_label);
            if (it_matrix != config_.label_to_matrix_idx.end()) {
                point_k_sem[i] = getSemanticKernel(it_matrix->second, temp_m_i);
            }
        }
#endif
    }

    int num_shards = static_cast<int>(block_shards_.size());
    int radius = static_cast<int>(std::ceil(l_scale_ / resolution_));

#ifdef _OPENMP
#pragma omp parallel num_threads(num_shards)
    {
#pragma omp for schedule(static, 1)
#endif
    for (int s = 0; s < num_shards; ++s) {
        auto& shard = block_shards_[static_cast<size_t>(s)];

        for (size_t i = 0; i < n; i++) {
            const Point3D& p = points[i];
            VoxelKey vk_p = pointToKey(p);
            BlockKey bk_p = voxelToBlockKey(vk_p);
            if (getShardIndex(bk_p) != s) continue;

            uint32_t raw_label = labels[i];
            auto it_dense = config_.raw_to_dense.find(raw_label);
            if (it_dense == config_.raw_to_dense.end()) continue;
            int dense_label = it_dense->second;
            float k_sem = point_k_sem[i];

            for (int dx = -radius; dx <= radius; dx++) {
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dz = -radius; dz <= radius; dz++) {
                        VoxelKey vk = {vk_p.x + dx, vk_p.y + dy, vk_p.z + dz};
                        BlockKey bk = voxelToBlockKey(vk);
                        if (getShardIndex(bk) != s) continue;

                        Point3D v_center = keyToPoint(vk);
                        float dist_sq = p.dist_sq(v_center);
                        if (dist_sq > l_scale_ * l_scale_) continue;

                        float k_sp = computeSpatialKernel(dist_sq);
                        if (k_sp <= 1e-6f) continue;

                        Block& blk = getOrCreateBlock(shard, bk);
                        int lx, ly, lz;
                        voxelToLocal(vk, lx, ly, lz);
                        int idx = flatIndex(lx, ly, lz, dense_label);
                        blk.alpha[static_cast<size_t>(idx)] += k_sp * k_sem;
                    }
                }
            }
        }
    }
#ifdef _OPENMP
    }
#endif
}

void ContinuousBKI::update(const std::vector<std::vector<float>>& probs, const std::vector<Point3D>& points, const std::vector<float>& weights) {
    if (probs.size() != points.size()) {
        std::cerr << "Mismatch in points and probs size" << std::endl;
        return;
    }
    bool use_weights = !weights.empty();
    if (use_weights && weights.size() != points.size()) {
        std::cerr << "Mismatch in points and weights size" << std::endl;
        return;
    }

    size_t n = points.size();
    int num_shards = static_cast<int>(block_shards_.size());
    int radius = static_cast<int>(std::ceil(l_scale_ / resolution_)) + 1;

#ifdef _OPENMP
#pragma omp parallel num_threads(num_shards)
    {
#pragma omp for schedule(static, 1)
#endif
    for (int s = 0; s < num_shards; ++s) {
        auto& shard = block_shards_[static_cast<size_t>(s)];

        for (size_t i = 0; i < n; i++) {
            const Point3D& p = points[i];
            VoxelKey vk_p = pointToKey(p);
            BlockKey bk_p = voxelToBlockKey(vk_p);
            if (getShardIndex(bk_p) != s) continue;

            const std::vector<float>& prob = probs[i];
            float w_i = use_weights ? weights[i] : 1.0f;

            for (int dx = -radius; dx <= radius; dx++) {
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dz = -radius; dz <= radius; dz++) {
                        VoxelKey vk = {vk_p.x + dx, vk_p.y + dy, vk_p.z + dz};
                        BlockKey bk = voxelToBlockKey(vk);
                        if (getShardIndex(bk) != s) continue;

                        Point3D v_center = keyToPoint(vk);
                        float dist_sq = p.dist_sq(v_center);
                        if (dist_sq > l_scale_ * l_scale_) continue;

                        float k_sp = computeSpatialKernel(dist_sq);
                        if (k_sp <= 1e-6f) continue;

                        Block& blk = getOrCreateBlock(shard, bk);
                        int lx, ly, lz;
                        voxelToLocal(vk, lx, ly, lz);

                        for (size_t c = 0; c < prob.size(); c++) {
                            int idx = flatIndex(lx, ly, lz, static_cast<int>(c));
                            blk.alpha[static_cast<size_t>(idx)] += w_i * k_sp * prob[c];
                        }
                    }
                }
            }
        }
    }
#ifdef _OPENMP
    }
#endif
}

int ContinuousBKI::size() const {
    int total = 0;
    for (const auto& shard : block_shards_) {
        total += static_cast<int>(shard.size()) * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
    }
    return total;
}

void ContinuousBKI::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for saving: " << filename << std::endl;
        return;
    }

    const uint8_t version = 2;
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint8_t));
    out.write(reinterpret_cast<const char*>(&resolution_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&l_scale_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&sigma_0_), sizeof(float));

    size_t num_blocks = 0;
    for (const auto& shard : block_shards_) num_blocks += shard.size();
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(size_t));

    const size_t block_alpha_size = static_cast<size_t>(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * config_.num_total_classes);
    for (const auto& shard : block_shards_) {
        for (const auto& kv : shard) {
            const BlockKey& bk = kv.first;
            const Block& blk = kv.second;
            out.write(reinterpret_cast<const char*>(&bk.x), sizeof(int));
            out.write(reinterpret_cast<const char*>(&bk.y), sizeof(int));
            out.write(reinterpret_cast<const char*>(&bk.z), sizeof(int));
            if (blk.alpha.size() == block_alpha_size) {
                out.write(reinterpret_cast<const char*>(blk.alpha.data()), block_alpha_size * sizeof(float));
            }
        }
    }
    out.close();
}

void ContinuousBKI::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading: " << filename << std::endl;
        return;
    }

    uint8_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(uint8_t));
    if (version != 2) {
        std::cerr << "Unsupported map file version: " << static_cast<int>(version) << " (expected 2)" << std::endl;
        return;
    }

    float res, l, s0;
    in.read(reinterpret_cast<char*>(&res), sizeof(float));
    in.read(reinterpret_cast<char*>(&l), sizeof(float));
    in.read(reinterpret_cast<char*>(&s0), sizeof(float));
    if (std::abs(res - resolution_) > 1e-4f) std::cerr << "Warning: Loaded resolution mismatch" << std::endl;

    size_t num_blocks = 0;
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(size_t));

    clear();
    const size_t block_alpha_size = static_cast<size_t>(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * config_.num_total_classes);

    for (size_t i = 0; i < num_blocks; i++) {
        BlockKey bk;
        in.read(reinterpret_cast<char*>(&bk.x), sizeof(int));
        in.read(reinterpret_cast<char*>(&bk.y), sizeof(int));
        in.read(reinterpret_cast<char*>(&bk.z), sizeof(int));
        Block blk;
        blk.alpha.resize(block_alpha_size);
        in.read(reinterpret_cast<char*>(blk.alpha.data()), block_alpha_size * sizeof(float));
        int s = getShardIndex(bk);
        block_shards_[static_cast<size_t>(s)][bk] = std::move(blk);
    }
    in.close();
}

std::vector<uint32_t> ContinuousBKI::infer(const std::vector<Point3D>& points) const {
    std::vector<uint32_t> results;
    results.reserve(points.size());

    std::vector<float> fallback_pred;
    for (const auto& p : points) {
        VoxelKey k = pointToKey(p);
        BlockKey bk = voxelToBlockKey(k);
        int s = getShardIndex(bk);
        const Block* blk = getBlockConst(block_shards_[static_cast<size_t>(s)], bk);

        if (blk != nullptr) {
            int lx, ly, lz;
            voxelToLocal(k, lx, ly, lz);
            float sum = 0.0f;
            int best_idx = 0;
            float best_val = -1.0f;
            for (int c = 0; c < config_.num_total_classes; c++) {
                float v = blk->alpha[static_cast<size_t>(flatIndex(lx, ly, lz, c))];
                sum += v;
                if (v > best_val) { best_val = v; best_idx = c; }
            }
            if (sum > epsilon_) {
                auto it_raw = config_.dense_to_raw.find(best_idx);
                if (it_raw != config_.dense_to_raw.end()) {
                    results.push_back(it_raw->second);
                } else {
                    results.push_back(0);
                }
                continue;
            }
        }

        if (osm_fallback_in_infer_ && K_pred_ > 0) {
            computePredPriorFromOSM(p.x, p.y, fallback_pred);
            if (!fallback_pred.empty()) {
                int best = static_cast<int>(std::max_element(fallback_pred.begin(), fallback_pred.end()) - fallback_pred.begin());
                if (best < config_.num_total_classes) {
                    auto it_raw = config_.dense_to_raw.find(best);
                    if (it_raw != config_.dense_to_raw.end()) {
                        results.push_back(it_raw->second);
                    } else {
                        results.push_back(0);
                    }
                } else {
                    results.push_back(0);
                }
            } else {
                results.push_back(0);
            }
        } else {
            results.push_back(0);
        }
    }
    return results;
}

std::vector<std::vector<float>> ContinuousBKI::infer_probs(const std::vector<Point3D>& points) const {
    std::vector<std::vector<float>> results;
    results.reserve(points.size());

    const int K = config_.num_total_classes;
    std::vector<float> fallback_pred;
    std::vector<float> uniform(K, 1.0f / static_cast<float>(K));

    for (const auto& p : points) {
        VoxelKey k = pointToKey(p);
        BlockKey bk = voxelToBlockKey(k);
        int s = getShardIndex(bk);
        const Block* blk = getBlockConst(block_shards_[static_cast<size_t>(s)], bk);

        if (blk != nullptr) {
            int lx, ly, lz;
            voxelToLocal(k, lx, ly, lz);
            std::vector<float> probs(static_cast<size_t>(K), 0.0f);
            float sum = 0.0f;
            for (int c = 0; c < K; c++) {
                probs[static_cast<size_t>(c)] = blk->alpha[static_cast<size_t>(flatIndex(lx, ly, lz, c))];
                sum += probs[static_cast<size_t>(c)];
            }
            if (sum > epsilon_) {
                for (float& v : probs) v /= sum;
                results.push_back(probs);
                continue;
            }
        }

        if (osm_fallback_in_infer_ && K_pred_ > 0) {
            computePredPriorFromOSM(p.x, p.y, fallback_pred);
            if (fallback_pred.size() == static_cast<size_t>(K)) {
                results.push_back(fallback_pred);
            } else {
                results.push_back(uniform);
            }
        } else {
            results.push_back(uniform);
        }
    }
    return results;
}

// --- Loader Implementations ---

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

    for (const auto& cat : osm_categories) {
        uint32_t num_items;
        file.read(reinterpret_cast<char*>(&num_items), sizeof(uint32_t));
        if (!file.good()) break;

        auto class_it = osm_class_map.find(cat);
        bool has_class = (class_it != osm_class_map.end());
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
            poly.computeBounds();
            if (has_class) {
                data.geometries[class_idx].push_back(poly);
            }
        }
    }

    file.close();
    return data;
}

Config loadConfigFromYAML(const std::string& config_path) {
    Config config;
    try {
        yaml_parser::YAMLNode yaml;
        yaml.parseFile(config_path);

        config.labels = yaml.getLabels();
        config.confusion_matrix = yaml.getConfusionMatrix();
        config.label_to_matrix_idx = yaml.getLabelToMatrixIdx();
        config.osm_class_map = yaml.getOSMClassMap();
        config.osm_categories = yaml.getOSMCategories();

        auto height_filter_str = yaml.getOSMHeightFilter();
        for (const auto& kv : height_filter_str) {
            auto it = config.osm_class_map.find(kv.first);
            if (it != config.osm_class_map.end()) {
                config.height_filter_map[it->second] = kv.second;
            }
        }

        std::vector<int> all_classes;
        for (const auto& kv : config.labels) {
            all_classes.push_back(kv.first);
        }
        for (size_t i = 0; i < all_classes.size(); i++) {
            config.raw_to_dense[all_classes[i]] = static_cast<int>(i);
            config.dense_to_raw[static_cast<int>(i)] = all_classes[i];
        }
        config.num_total_classes = static_cast<int>(all_classes.size());

    } catch (const std::exception& e) {
        std::cerr << "Error loading config from " << config_path << ": " << e.what() << std::endl;
        throw;
    }
    return config;
}

} // namespace continuous_bki
