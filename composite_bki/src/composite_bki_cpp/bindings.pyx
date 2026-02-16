# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
import numpy as np
cimport numpy as np

# C++ Declarations
cdef extern from "continuous_bki.hpp" namespace "continuous_bki":
    cdef cppclass Point3D:
        float x, y, z
        Point3D() except +
        Point3D(float, float, float) except +
    
    cdef cppclass Config:
        map[int, string] labels
        map[string, int] osm_class_map
        vector[string] osm_categories
        vector[vector[float]] confusion_matrix
        map[int, int] label_to_matrix_idx
        map[int, int] raw_to_dense
        map[int, int] dense_to_raw
        int num_total_classes
    
    cdef cppclass OSMData:
        pass
    
    cdef cppclass ContinuousBKI:
        ContinuousBKI(const Config&, const OSMData&, float, float, float, float, float, bool, bool, int,
                      float, bool, float, bool, float, float) except +
        
        void update(const vector[unsigned int]& labels, const vector[Point3D]& points) except +
        void update(const vector[vector[float]]& probs, const vector[Point3D]& points, const vector[float]& weights) except +
        
        vector[unsigned int] infer(const vector[Point3D]& points) except +
        vector[vector[float]] infer_probs(const vector[Point3D]& points) except +
        
        void save(const string& filename) except +
        void load(const string& filename) except +
        
        void clear() except +
        int size() except +

    # Re-declare loaders to access them
    Config loadConfigFromYAML(const string& config_path) except +
    OSMData loadOSMBinary(const string& filename,
                          const map[string, int]& osm_class_map,
                          const vector[string]& osm_categories) except +
    OSMData loadOSM(const string& filename,
                    const Config& config) except +

# Python Wrapper
cdef class PyContinuousBKI:
    cdef ContinuousBKI* bki_ptr
    cdef OSMData osm_data
    cdef Config config
    
    def __cinit__(self, str osm_path, str config_path,
                  float resolution=0.1,
                  float l_scale=0.3,
                  float sigma_0=1.0,
                  float prior_delta=5.0,
                  float height_sigma=0.3,
                  bool use_semantic_kernel=True,
                  bool use_spatial_kernel=True,
                  int num_threads=-1,
                  float alpha0=1.0,
                  bool seed_osm_prior=False,
                  float osm_prior_strength=0.0,
                  bool osm_fallback_in_infer=True,
                  float lambda_min=0.8,
                  float lambda_max=0.99):
        
        if not config_path:
            raise ValueError("config_path is required")
            
        self.config = loadConfigFromYAML(config_path.encode('utf-8'))
        
        self.osm_data = loadOSM(osm_path.encode('utf-8'),
                                self.config)
                                      
        self.bki_ptr = new ContinuousBKI(self.config, self.osm_data,
                                         resolution, l_scale, sigma_0, prior_delta, height_sigma,
                                         use_semantic_kernel, use_spatial_kernel, num_threads,
                                         alpha0, seed_osm_prior, osm_prior_strength, osm_fallback_in_infer,
                                         lambda_min, lambda_max)
                                         
    def __dealloc__(self):
        if self.bki_ptr != NULL:
            del self.bki_ptr
            
    def update(self, 
               np.ndarray[np.uint32_t, ndim=1] labels, 
               np.ndarray[np.float32_t, ndim=2] points):
        
        cdef int n = points.shape[0]
        cdef vector[Point3D] cpp_points
        cdef vector[unsigned int] cpp_labels
        cpp_points.reserve(n)
        cpp_labels.reserve(n)
        
        for i in range(n):
            cpp_points.push_back(Point3D(points[i, 0], points[i, 1], points[i, 2]))
            cpp_labels.push_back(labels[i])
            
        self.bki_ptr.update(cpp_labels, cpp_points)
        
    def update_soft(self,
                    np.ndarray[np.float32_t, ndim=2] probs,
                    np.ndarray[np.float32_t, ndim=2] points,
                    np.ndarray[np.float32_t, ndim=1] weights=None):
        
        cdef int n = points.shape[0]
        cdef vector[Point3D] cpp_points
        cdef vector[vector[float]] cpp_probs
        cdef vector[float] cpp_weights
        
        cpp_points.reserve(n)
        cpp_probs.reserve(n)
        
        cdef int num_classes = probs.shape[1]
        cdef vector[float] temp_prob
        
        for i in range(n):
            cpp_points.push_back(Point3D(points[i, 0], points[i, 1], points[i, 2]))
            temp_prob.clear()
            temp_prob.reserve(num_classes)
            for j in range(num_classes):
                temp_prob.push_back(probs[i, j])
            cpp_probs.push_back(temp_prob)
            
        if weights is not None:
            if weights.shape[0] != n:
                raise ValueError("Weights size mismatch")
            cpp_weights.reserve(n)
            for i in range(n):
                cpp_weights.push_back(weights[i])
        
        self.bki_ptr.update(cpp_probs, cpp_points, cpp_weights)

    def infer(self, np.ndarray[np.float32_t, ndim=2] points):
        cdef int n = points.shape[0]
        cdef vector[Point3D] cpp_points
        cpp_points.reserve(n)
        
        for i in range(n):
            cpp_points.push_back(Point3D(points[i, 0], points[i, 1], points[i, 2]))
            
        cdef vector[unsigned int] result = self.bki_ptr.infer(cpp_points)
        
        cdef np.ndarray[np.uint32_t, ndim=1] output = np.zeros(n, dtype=np.uint32)
        for i in range(n):
            output[i] = result[i]
            
        return output

    def infer_probs(self, np.ndarray[np.float32_t, ndim=2] points):
        cdef int n = points.shape[0]
        cdef vector[Point3D] cpp_points
        cpp_points.reserve(n)
        
        for i in range(n):
            cpp_points.push_back(Point3D(points[i, 0], points[i, 1], points[i, 2]))
            
        cdef vector[vector[float]] result = self.bki_ptr.infer_probs(cpp_points)
        
        # Determine num classes from first result or config (safe way is from result[0] if n>0)
        cdef int num_classes = 0
        if n > 0 and result.size() > 0:
            num_classes = result[0].size()
        else:
            return np.zeros((0, 0), dtype=np.float32)

        cdef np.ndarray[np.float32_t, ndim=2] output = np.zeros((n, num_classes), dtype=np.float32)
        for i in range(n):
            for j in range(num_classes):
                output[i, j] = result[i][j]
            
        return output
        
    def get_size(self):
        return self.bki_ptr.size()
        
    def clear(self):
        self.bki_ptr.clear()

    def save(self, str filename):
        self.bki_ptr.save(filename.encode('utf-8'))

    def load(self, str filename):
        self.bki_ptr.load(filename.encode('utf-8'))
