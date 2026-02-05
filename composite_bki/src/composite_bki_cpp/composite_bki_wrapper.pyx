# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
import numpy as np
cimport numpy as np

# C++ declarations
cdef extern from "composite_bki.hpp" namespace "composite_bki":
    cdef cppclass Point3D:
        float x, y, z
        Point3D() except +
        Point3D(float, float, float) except +
    
    cdef cppclass Config:
        map[int, string] labels
        vector[vector[float]] confusion_matrix
        map[int, int] label_to_matrix_idx
        map[string, int] osm_class_map
        vector[string] osm_categories
        map[int, int] raw_to_dense
        map[int, int] dense_to_raw
        int num_total_classes
    
    cdef cppclass OSMData:
        pass
    
    Config loadConfigFromYAML(const string& config_path) except +
    
    cdef cppclass SemanticBKI:
        SemanticBKI(const Config&, const OSMData&, float, float, float, float, bint, bint, int) except +
        vector[float] getOSMPrior(float x, float y) except +
        float getSemanticKernel(int matrix_idx, const vector[float]& m_i) except +
        vector[float] computeSpatialKernelVectorized(const vector[float]& dists) except +
        float getLScale() except +
        float getSigma0() except +
        int getNumThreads() except +
        bint getUseSpatialKernel() except +
        bint getUseSemanticKernel() except +
    
    cdef cppclass Metrics:
        float accuracy
        float miou
        map[int, float] per_class_iou
        map[int, float] per_class_acc
    
    OSMData loadOSMBinary(const string& filename,
                          const map[string, int]& osm_class_map,
                          const vector[string]& osm_categories) except +
    
    vector[float] computeSemanticWeights(
        const vector[Point3D]& points,
        const vector[unsigned int]& input_labels,
        const SemanticBKI& bki,
        const Config& config,
        vector[int]& dense_labels) except +
    
    vector[unsigned int] runBKIUpdate(
        const vector[Point3D]& points,
        const vector[unsigned int]& input_labels,
        const vector[float]& k_sem_all,
        const vector[int]& dense_labels,
        const SemanticBKI& bki,
        const Config& config,
        float alpha_0) except +
    
    Metrics computeMetrics(
        const vector[unsigned int]& pred_labels,
        const vector[unsigned int]& gt_labels) except +
    
# Python wrapper class
cdef class PySemanticBKI:
    cdef SemanticBKI* bki_ptr
    cdef OSMData osm_data
    cdef Config config
    
    def __cinit__(self, str osm_path, str config_path, 
                  float l_scale=3.0, float sigma_0=1.0, float prior_delta=5.0,
                  float height_sigma=0.3,
                  bool use_semantic_kernel=True, bool use_spatial_kernel=True,
                  int num_threads=-1):
        """
        Initialize Semantic BKI processor.
        
        Args:
            osm_path: Path to OSM binary file
            config_path: Path to YAML config file
            l_scale: Spatial kernel scale parameter
            sigma_0: Spatial kernel amplitude
            prior_delta: OSM prior distance scaling
            height_sigma: Sigma for height-based gating of ground classes
            use_semantic_kernel: Enable semantic kernel (OSM priors, default: True)
            use_spatial_kernel: Enable spatial kernel (distance-based, default: True)
            num_threads: Number of OpenMP threads (-1 = auto-detect, use all cores)
        """
        if not config_path:
            raise ValueError("config_path is required. Provide a YAML config file.")
        
        # Get config
        # Load from YAML file
        self.config = loadConfigFromYAML(config_path.encode('utf-8'))
        
        # Load OSM data (use config-defined class mapping)
        self.osm_data = loadOSMBinary(osm_path.encode('utf-8'),
                                      self.config.osm_class_map,
                                      self.config.osm_categories)
        
        # Create BKI instance with kernel flags and threading
        self.bki_ptr = new SemanticBKI(self.config, self.osm_data, 
                                       l_scale, sigma_0, prior_delta, height_sigma,
                                       use_semantic_kernel, use_spatial_kernel,
                                       num_threads)
    
    def __dealloc__(self):
        if self.bki_ptr != NULL:
            del self.bki_ptr
    
    def process_point_cloud(self, 
                           np.ndarray[np.float32_t, ndim=2] points,
                           np.ndarray[np.uint32_t, ndim=1] labels,
                           float alpha_0=0.01):
        """
        Process a point cloud with semantic labels.
        
        Args:
            points: Nx3 array of point coordinates (x, y, z)
            labels: N array of semantic labels
            alpha_0: Prior weight for Dirichlet distribution
            
        Returns:
            refined_labels: N array of refined semantic labels
        """
        cdef vector[Point3D] cpp_points
        cdef vector[unsigned int] cpp_labels
        cdef vector[float] k_sem_all
        cdef vector[int] dense_labels
        cdef vector[unsigned int] result
        
        # Convert numpy arrays to C++ vectors
        cdef int n = points.shape[0]
        cpp_points.reserve(n)
        cpp_labels.reserve(n)
        
        for i in range(n):
            cpp_points.push_back(Point3D(points[i, 0], points[i, 1], points[i, 2]))
            cpp_labels.push_back(labels[i])
        
        print("Computing semantic weights...")
        k_sem_all = computeSemanticWeights(cpp_points, cpp_labels, 
                                           self.bki_ptr[0], self.config, dense_labels)
        
        print("Running BKI update...")
        result = runBKIUpdate(cpp_points, cpp_labels, k_sem_all, 
                             dense_labels, self.bki_ptr[0], self.config, alpha_0)
        
        # Convert result to numpy array
        cdef np.ndarray[np.uint32_t, ndim=1] output = np.zeros(n, dtype=np.uint32)
        for i in range(n):
            output[i] = result[i]
        
        return output
    
    def evaluate_metrics(self,
                        np.ndarray[np.uint32_t, ndim=1] pred_labels,
                        np.ndarray[np.uint32_t, ndim=1] gt_labels):
        """
        Compute evaluation metrics.
        
        Args:
            pred_labels: Predicted labels
            gt_labels: Ground truth labels
            
        Returns:
            dict with keys: 'accuracy', 'miou', 'per_class_iou', 'per_class_acc'
        """
        cdef vector[unsigned int] cpp_pred
        cdef vector[unsigned int] cpp_gt
        
        cdef int n = len(pred_labels)
        cpp_pred.reserve(n)
        cpp_gt.reserve(n)
        
        for i in range(n):
            cpp_pred.push_back(pred_labels[i])
            cpp_gt.push_back(gt_labels[i])
        
        cdef Metrics metrics = computeMetrics(cpp_pred, cpp_gt)
        
        return {
            'accuracy': metrics.accuracy,
            'miou': metrics.miou,
            'per_class_iou': dict(metrics.per_class_iou),
            'per_class_acc': dict(metrics.per_class_acc)
        }


def run_pipeline(str lidar_path, 
                str label_path,
                str osm_path,
                str config_path,
                str output_path=None,
                str ground_truth_path=None,
                float l_scale=3.0,
                float sigma_0=1.0,
                float prior_delta=5.0,
                float alpha_0=0.01,
                float height_sigma=0.3,
                bool use_semantic_kernel=True,
                bool use_spatial_kernel=True,
                int num_threads=-1):
    """
    Run the complete Semantic-Spatial BKI pipeline (PARALLELIZED).
    
    Args:
        lidar_path: Path to .bin lidar point cloud file
        label_path: Path to .label or .bin semantic labels
        osm_path: Path to .bin OSM map file
        config_path: Path to YAML config file (overrides use_kitti)
        output_path: Optional path to save refined labels
        ground_truth_path: Optional path to ground truth for evaluation
        l_scale: Spatial kernel scale (default: 3.0)
        sigma_0: Spatial kernel amplitude (default: 1.0)
        prior_delta: OSM prior distance scaling (default: 5.0)
        alpha_0: Prior weight for Dirichlet (default: 0.01)
        height_sigma: Sigma for height-based gating (default: 0.3)
        use_semantic_kernel: Enable semantic kernel (OSM priors, default: True)
        use_spatial_kernel: Enable spatial kernel (distance-based, default: True)
        num_threads: Number of OpenMP threads (-1 = auto, use all cores)
        
    Returns:
        refined_labels: NumPy array of refined labels
    """
    import numpy as np
    import os
    
    print(f"Loading point cloud from {lidar_path}...")
    scan = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3].astype(np.float32)
    
    print(f"Loading labels from {label_path}...")
    labels_raw = np.fromfile(label_path, dtype=np.uint32)
    input_labels = (labels_raw & 0xFFFF).astype(np.uint32)
    
    print(f"Initializing BKI with OSM data from {osm_path}...")
    bki = PySemanticBKI(osm_path, config_path, l_scale, sigma_0, prior_delta, height_sigma,
                        use_semantic_kernel, use_spatial_kernel, num_threads)
    
    print(f"Processing {len(points)} points...")
    refined_labels = bki.process_point_cloud(points, input_labels, alpha_0)
    
    # Evaluation
    if ground_truth_path and os.path.exists(ground_truth_path):
        print(f"Loading ground truth from {ground_truth_path}...")
        gt_raw = np.fromfile(ground_truth_path, dtype=np.uint32)
        gt_labels = (gt_raw & 0xFFFF).astype(np.uint32)
        
        print("Computing baseline metrics...")
        baseline_metrics = bki.evaluate_metrics(input_labels, gt_labels)
        print(f"Baseline -> Accuracy: {baseline_metrics['accuracy']*100:.2f}%, "
              f"mIoU: {baseline_metrics['miou']*100:.2f}%")
        
        print("Computing refined metrics...")
        refined_metrics = bki.evaluate_metrics(refined_labels, gt_labels)
        print(f"Refined -> Accuracy: {refined_metrics['accuracy']*100:.2f}%, "
              f"mIoU: {refined_metrics['miou']*100:.2f}%")
        
        acc_delta = refined_metrics['accuracy'] - baseline_metrics['accuracy']
        miou_delta = refined_metrics['miou'] - baseline_metrics['miou']
        print(f"Improvement -> Acc: {acc_delta*100:+.2f}%, mIoU: {miou_delta*100:+.2f}%")
    
    # Save output
    if output_path:
        print(f"Saving refined labels to {output_path}...")
        refined_labels.tofile(output_path)
        
        num_changed = np.sum(refined_labels != input_labels)
        print(f"Changed {num_changed} points ({num_changed/len(refined_labels)*100:.2f}%)")
    
    return refined_labels
