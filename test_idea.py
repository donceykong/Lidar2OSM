import struct
import os
import argparse
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

# --- 1. Compact OSM Loader ---
def load_osm_bin(bin_file):
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"OSM bin file not found: {bin_file}")

    data = {}
    categories = ['buildings', 'roads', 'grasslands', 'trees', 'wood', 'parking', 'sidewalks', 'fences']

    with open(bin_file, 'rb') as f:
        for cat in categories:
            try:
                num_items_bytes = f.read(4)
                if not num_items_bytes: break 
                num_items = struct.unpack('I', num_items_bytes)[0]
                
                items = []
                for _ in range(num_items):
                    n_pts_bytes = f.read(4)
                    if not n_pts_bytes: break
                    n_pts = struct.unpack('I', n_pts_bytes)[0]
                    
                    bytes_data = f.read(n_pts * 2 * 4)
                    floats = struct.unpack(f'{n_pts * 2}f', bytes_data)
                    poly_coords = list(zip(floats[0::2], floats[1::2]))
                    items.append(poly_coords)
                
                data[cat] = items
            except struct.error:
                print(f"Warning: Failed to load {cat}")
                break
    return data

# --- 2. BKI Configurations ---

class BaseConfig:
    CONFUSION_MATRIX = np.array([
        # Osm: Rd    Park  Sw    Veg   Bld   Fence
        [0.85, 0.10, 0.05, 0.00, 0.00, 0.00], # 0: Pred Road
        [0.10, 0.85, 0.05, 0.00, 0.00, 0.00], # 1: Pred Parking
        [0.05, 0.05, 0.80, 0.05, 0.05, 0.00], # 2: Pred Sidewalk
        [0.01, 0.01, 0.05, 0.90, 0.02, 0.01], # 3: Pred Veg
        [0.01, 0.01, 0.02, 0.05, 0.90, 0.01], # 4: Pred Building
        [0.00, 0.00, 0.01, 0.05, 0.10, 0.84], # 5: Pred Fence
        [0.40, 0.40, 0.05, 0.01, 0.01, 0.00], # 6: Pred VEHICLE (Dynamic)
        [0.15, 0.15, 0.50, 0.05, 0.05, 0.05]  # 7: Pred OBJECT (Pole, Sign)
    ])
    
    OSM_CLASS_MAP = {
        'roads': 0, 'parking': 1, 'sidewalks': 2,
        'grasslands': 3, 'trees': 3, 'wood': 3,
        'buildings': 4, 'fences': 5
    }

class MCDConfig(BaseConfig):
    # MCD / MultiCampus Dataset Label Mapping
    LABELS = {
        0: "barrier", 1: "bike", 2: "building", 3: "chair", 4: "cliff",
        5: "container", 6: "curb", 7: "fence", 8: "hydrant", 9: "infosign",
        10: "lanemarking", 11: "noise", 12: "other", 13: "parkinglot", 14: "pedestrian",
        15: "pole", 16: "road", 17: "shelter", 18: "sidewalk", 19: "stairs",
        20: "structure-other", 21: "traffic-cone", 22: "traffic-sign", 23: "trashbin",
        24: "treetrunk", 25: "vegetation", 26: "vehicle-dynamic",
        27: "vehicle-other", 28: "vehicle-static"
    }
    
    LABEL_TO_MATRIX_IDX = {
        16: 0, 10: 0,                   # Road-like (Row 0)
        13: 1,                          # Parking (Row 1)
        18: 2, 6: 2, 19: 2, 4: 2,       # Sidewalk-like (Row 2)
        25: 3, 24: 3,                   # Veg-like (Row 3)
        2: 4, 20: 4, 17: 4,             # Building-like (Row 4)
        7: 5, 0: 5,                     # Fence (Row 5)
        1: 6, 14: 6, 26: 6, 27: 6, 28: 6, # Vehicle/Dynamic (Row 6)
        15: 7, 22: 7, 9: 7, 12: 7, 3: 7, 5: 7, 8: 7, 21: 7, 23: 7, 11: 7 # Objects (Row 7)
    }

class KittiConfig(BaseConfig):
    LABELS = {
        0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 16: "on-rails", 18: "truck", 20: "other-vehicle",
        30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
        50: "building", 51: "fence", 52: "other-structure", 60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain",
        80: "pole", 81: "traffic-sign", 99: "other-object",
        252: "moving-car", 253: "moving-bicyclist", 254: "moving-person", 255: "moving-motorcyclist",
        256: "moving-on-rails", 257: "moving-bus", 258: "moving-truck", 259: "moving-other-vehicle"
    }
    
    LABEL_TO_MATRIX_IDX = {
        40: 0, 60: 0,                   # Road-like -> 0
        44: 1,                          # Parking -> 1
        48: 2, 49: 2, 72: 2,            # Sidewalk-like -> 2
        70: 3, 71: 3,                   # Veg -> 3
        50: 4, 52: 4,                   # Building -> 4
        51: 5,                          # Fence -> 5
        10: 6, 11: 6, 13: 6, 15: 6, 16: 6, 18: 6, 20: 6, # Vehicles -> 6
        252: 6, 253: 6, 254: 6, 255: 6, 256: 6, 257: 6, 258: 6, 259: 6,
        30: 7, 31: 7, 32: 7, 80: 7, 81: 7, 99: 7 # Objects -> 7
    }

def get_config(use_kitti=False):
    config_cls = KittiConfig if use_kitti else MCDConfig
    all_classes = list(config_cls.LABELS.keys())
    
    class Config:
        LABELS = config_cls.LABELS
        CONFUSION_MATRIX = config_cls.CONFUSION_MATRIX
        LABEL_TO_MATRIX_IDX = config_cls.LABEL_TO_MATRIX_IDX
        OSM_CLASS_MAP = config_cls.OSM_CLASS_MAP
        ALL_CLASSES = all_classes
        RAW_TO_DENSE = {raw: i for i, raw in enumerate(all_classes)}
        DENSE_TO_RAW = {i: raw for i, raw in enumerate(all_classes)}
        NUM_TOTAL_CLASSES = len(all_classes)
        
    return Config

# --- 3. BKI Logic ---
class SemanticBKI:
    def __init__(self, config, osm_data, 
                 l_scale=0.5, sigma_0=1.0, prior_delta=5.0):
        self.M = config.CONFUSION_MATRIX
        self.K_pred = self.M.shape[0]
        self.K_prior = self.M.shape[1]
        self.class_map = config.OSM_CLASS_MAP
        self.l = l_scale
        self.sigma_0 = sigma_0
        self.delta = prior_delta
        self.epsilon = 1e-6
        
        self.osm_geoms = self._prep_geometry(osm_data)
        self.osm_trees = self._build_rtrees(self.osm_geoms)

    def _prep_geometry(self, osm_data):
        geoms = {k: [] for k in range(self.K_prior)}
        for cat_name, items in osm_data.items():
            if cat_name == 'bounds': continue
            idx = self.class_map.get(cat_name)
            if idx is not None:
                for coords in items:
                    if len(coords) < 2: continue
                    if len(coords) < 3:
                        geoms[idx].append(LineString(coords))
                    else:
                        geoms[idx].append(Polygon(coords))
        return geoms

    def _build_rtrees(self, geoms_by_class):
        trees = {}
        for k, geom_list in geoms_by_class.items():
            if geom_list:
                trees[k] = STRtree(geom_list)
        return trees

    def get_osm_prior(self, x, y):
        p = Point(x, y)
        s_scores = np.zeros(self.K_prior)
        
        for k in range(self.K_prior):
            geoms = self.osm_geoms[k]
            tree = self.osm_trees.get(k)
            
            if not geoms or tree is None:
                dist = 50.0 
            else:
                nearest_idx = tree.nearest(p)
                if hasattr(tree, 'geometries'):
                     nearest_geom = tree.geometries.take(nearest_idx)
                else:
                     nearest_geom = geoms[nearest_idx] if isinstance(nearest_idx, (int, np.integer)) else geoms[nearest_idx[0]]

                dist = p.distance(nearest_geom)
                if isinstance(nearest_geom, Polygon) and dist == 0.0:
                    if nearest_geom.contains(p):
                         dist = -1.0 * nearest_geom.boundary.distance(p)
            
            s_scores[k] = 1.0 / (1.0 + np.exp(dist / self.delta))
        
        m_i = s_scores / (np.sum(s_scores) + self.epsilon)
        return m_i

    def get_semantic_kernel(self, matrix_idx, m_i):
        c_xi = np.max(m_i)
        expected_obs = self.M @ m_i
        numerator = expected_obs[matrix_idx]
        denominator = np.max(expected_obs) + self.epsilon
        s_i = numerator / denominator
        return (1 - c_xi) + (c_xi * s_i)

    def compute_spatial_kernel_vectorized(self, dists):
        xi = dists / self.l
        mask = xi < 1.0
        k_val = np.zeros_like(dists)
        xi_m = xi[mask]
        
        term1 = (1.0/3.0) * (2.0 + np.cos(2.0 * np.pi * xi_m)) * (1.0 - xi_m)
        term2 = (1.0/(2.0 * np.pi)) * np.sin(2.0 * np.pi * xi_m)
        k_val[mask] = self.sigma_0 * (term1 + term2)
        return k_val

def compute_metrics(pred_labels, gt_labels, valid_classes):
    """
    Compute mIoU and Accuracy given prediction and ground truth.
    Ignores 0 (unlabeled) and classes not in valid_classes.
    """
    intersection = {}
    union = {}
    total_correct = 0
    total_valid = 0
    
    # We use valid_classes to know what is "interesting" to evaluate
    # But usually mIoU is over all classes present in GT
    unique_gt = np.unique(gt_labels)
    
    for cls in unique_gt:
        if cls == 0: continue # Skip unlabeled in GT
        
        # Mask for this class
        gt_mask = (gt_labels == cls)
        pred_mask = (pred_labels == cls)
        
        inter = np.sum(gt_mask & pred_mask)
        uni = np.sum(gt_mask | pred_mask)
        
        intersection[cls] = inter
        union[cls] = uni
        
        total_correct += inter
        total_valid += np.sum(gt_mask)
        
    iou_list = []
    for cls in intersection:
        if union[cls] > 0:
            iou_list.append(intersection[cls] / union[cls])
            
    miou = np.mean(iou_list) if iou_list else 0.0
    accuracy = total_correct / total_valid if total_valid > 0 else 0.0
    
    return accuracy, miou

def map_kitti_to_mcd(kitti_labels):
    """
    Best-effort mapping from SemanticKITTI IDs to MCD IDs for evaluation.
    Source: KittiConfig.LABELS -> MCDConfig.LABELS logic
    """
    # This is tricky because the user wants to compare "Kitti" input to "MCD" ground truth.
    # We need a mapping function.
    # Kitti 40 (Road) -> MCD 16 (Road)
    # Kitti 48 (Sidewalk) -> MCD 18 (Sidewalk)
    # ...
    # Simplified mapping for common classes:
    mapping = {
        40: 16, # Road
        44: 13, # Parking -> Parkinglot
        48: 18, # Sidewalk
        49: 18, # Other-ground -> Sidewalk? or 12 Other?
        70: 25, # Vegetation
        71: 24, # Trunk -> Treetrunk
        72: 25, # Terrain -> Vegetation? or 4 Cliff? Let's map to Veg for now
        50: 2,  # Building
        51: 7,  # Fence
        52: 20, # Other-structure -> Structure-other
        10: 26, 11: 1, 13: 26, 15: 26, 16: 26, 18: 26, 20: 27, # Vehicles -> Vehicle-dynamic/other/Bike
        30: 14, 31: 1, 32: 26, # Person/Bicyclist
        80: 15, # Pole
        81: 22, # Traffic-sign
        99: 12  # Other-object -> Other
    }
    
    mapped = np.zeros_like(kitti_labels)
    for k, v in mapping.items():
        mapped[kitti_labels == k] = v
        
    return mapped

# --- 4. Hybrid Pipeline ---
def run_pipeline(lidar_path, sem_path, osm_path, output_path=None, ground_truth_path=None, use_kitti=False):
    
    config = get_config(use_kitti)
    print(f"Using {'SemanticKITTI' if use_kitti else 'MCD'} Label Configuration.")
    
    print("Loading data...")
    scan = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3] 
    
    labels_raw = np.fromfile(sem_path, dtype=np.uint32).reshape((-1))
    input_labels = labels_raw & 0xFFFF 
    
    osm_data = load_osm_bin(osm_path)
    bki = SemanticBKI(config, osm_data)
    
    # ---------------------------------------------------------
    # Optional: Load Ground Truth for Evaluation
    # ---------------------------------------------------------
    gt_labels = None
    if ground_truth_path:
        if os.path.exists(ground_truth_path):
            print(f"Loading Ground Truth: {ground_truth_path}")
            gt_raw = np.fromfile(ground_truth_path, dtype=np.uint32).reshape((-1))
            gt_labels = gt_raw & 0xFFFF
            
            # If input is Kitti and GT is MCD, we need to map Input -> MCD to evaluate Input Accuracy
            if use_kitti:
                print("Mapping Input (Kitti) to GT (MCD) for baseline evaluation...")
                eval_input_labels = map_kitti_to_mcd(input_labels)
            else:
                eval_input_labels = input_labels
                
            acc_in, miou_in = compute_metrics(eval_input_labels, gt_labels, list(MCDConfig.LABELS.keys()))
            print(f"Baseline Input -> Accuracy: {acc_in*100:.2f}%, mIoU: {miou_in*100:.2f}%")
        else:
            print(f"Warning: GT path provided but not found: {ground_truth_path}")

    # ---------------------------------------------------------
    # STEP A: Pre-compute Semantic Weights (k_sem)
    # ---------------------------------------------------------
    print(f"Pre-computing kernels for {len(points)} points...")
    
    k_sem_all = np.ones(len(points), dtype=np.float32)
    dense_labels = np.full(len(points), -1, dtype=int)
    
    for i in range(len(points)):
        raw_label = input_labels[i]
        
        d_idx = config.RAW_TO_DENSE.get(raw_label)
        if d_idx is not None:
            dense_labels[i] = d_idx
        
        matrix_idx = config.LABEL_TO_MATRIX_IDX.get(raw_label)
        
        if matrix_idx is not None:
            m_i = bki.get_osm_prior(points[i,0], points[i,1])
            k_sem_all[i] = bki.get_semantic_kernel(matrix_idx, m_i)

    # ---------------------------------------------------------
    # STEP B: Spatial Indexing
    # ---------------------------------------------------------
    print("Building KD-Tree...")
    tree = cKDTree(points)
    
    # ---------------------------------------------------------
    # STEP C: Composite Update
    # ---------------------------------------------------------
    print("Running Hybrid BKI Update...")
    
    neighbors_list = tree.query_ball_point(points, r=bki.l)
    
    results = []
    final_labels = np.copy(input_labels) # Store for evaluation
    alpha_0 = 0.01 
    
    for j, neighbors_idx in enumerate(neighbors_list):
        if j % 10000 == 0: print(f"  Processed {j}...")
        
        alpha_star = np.full(config.NUM_TOTAL_CLASSES, alpha_0)
        
        valid_n_idx = [idx for idx in neighbors_idx if dense_labels[idx] != -1]
        
        # If no valid neighbors, keep original
        if not valid_n_idx:
            new_raw_class = input_labels[j]
        else:
            valid_n_idx = np.array(valid_n_idx)
            dists = np.linalg.norm(points[valid_n_idx] - points[j], axis=1)
            k_sp_values = bki.compute_spatial_kernel_vectorized(dists)
            k_sem_values = k_sem_all[valid_n_idx]
            combined_weights = k_sp_values * k_sem_values
            
            neighbor_dense_classes = dense_labels[valid_n_idx]
            np.add.at(alpha_star, neighbor_dense_classes, combined_weights)
            
            best_dense_idx = np.argmax(alpha_star)
            new_raw_class = config.DENSE_TO_RAW[best_dense_idx]
        
        final_labels[j] = new_raw_class
        
        # Only save to CSV if requested (optimization)
        if output_path:
            results.append({
                'point_idx': j, 
                'new_class': new_raw_class,
                'orig_class': input_labels[j],
                'new_class_name': config.LABELS.get(new_raw_class, "Unknown"),
                'orig_class_name': config.LABELS.get(input_labels[j], "Unknown"),
                'k_sem': k_sem_all[j],
                'k_spatial_avg': 0.0, # Placeholder if not computing mean
                'changed': new_raw_class != input_labels[j]
            })

    # ---------------------------------------------------------
    # STEP D: Evaluation & Output
    # ---------------------------------------------------------
    if gt_labels is not None:
        if use_kitti:
            print("Mapping Refined (Kitti) to GT (MCD) for evaluation...")
            eval_final_labels = map_kitti_to_mcd(final_labels)
        else:
            eval_final_labels = final_labels
            
        acc_out, miou_out = compute_metrics(eval_final_labels, gt_labels, list(MCDConfig.LABELS.keys()))
        print(f"Refined Output -> Accuracy: {acc_out*100:.2f}%, mIoU: {miou_out*100:.2f}%")
        print(f"Improvement -> Acc: {(acc_out-acc_in)*100:+.2f}%, mIoU: {(miou_out-miou_in)*100:+.2f}%")

    if output_path:
        df = pd.DataFrame(results)
        print(f"\nChanged {df['changed'].sum()} points.")
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return final_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Semantic-Spatial BKI Update")
    
    parser.add_argument("--scan", type=str, required=True, help="Path to .bin lidar point cloud")
    parser.add_argument("--label", type=str, required=True, help="Path to .label or .bin semantic labels")
    parser.add_argument("--osm", type=str, required=True, help="Path to .bin OSM map file")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results as CSV")
    parser.add_argument("--ground_truth", type=str, default=None, help="Optional path to GT labels for evaluation")
    parser.add_argument("--kitti_labels", action="store_true", help="Use SemanticKITTI label mapping")

    args = parser.parse_args()

    run_pipeline(args.scan, args.label, args.osm, args.output, args.ground_truth, args.kitti_labels)
