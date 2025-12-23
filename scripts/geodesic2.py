import open3d as o3d
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq

def astar_geodesic_path(mesh, source_idx, target_idx):
    """
    Find the geodesic path between source and target vertices using A*.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        source_idx (int): Index of the source vertex.
        target_idx (int): Index of the target vertex.

    Returns:
        list: List of vertex indices forming the geodesic path.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    
    # Build adjacency list
    adjacency_list = {i: set() for i in range(len(V))}
    for tri in F:
        adjacency_list[tri[0]].update([tri[1], tri[2]])
        adjacency_list[tri[1]].update([tri[0], tri[2]])
        adjacency_list[tri[2]].update([tri[0], tri[1]])

    # A* search
    open_set = []
    heapq.heappush(open_set, (0, source_idx))  # (priority, vertex)
    came_from = {}
    g_score = {i: float('inf') for i in range(len(V))}
    g_score[source_idx] = 0
    f_score = {i: float('inf') for i in range(len(V))}
    f_score[source_idx] = np.linalg.norm(V[source_idx] - V[target_idx])  # Heuristic

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target_idx:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(source_idx)
            path.reverse()
            return path

        for neighbor in adjacency_list[current]:
            tentative_g_score = g_score[current] + np.linalg.norm(V[current] - V[neighbor])
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + np.linalg.norm(V[neighbor] - V[target_idx])
                if neighbor not in [v[1] for v in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Return empty if no path is found

def compute_heat_geodesic(mesh, source_idx, t=1e-2):
    """
    Compute geodesic distances on a mesh using the Heat Method.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        source_idx (int): Index of the source vertex.
        t (float): Heat diffusion time step.

    Returns:
        np.ndarray: Geodesic distances from the source vertex to all vertices.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    L = compute_cotangent_laplacian(V, F)

    # Step 1: Solve heat diffusion equation
    heat_source = np.zeros(V.shape[0])
    heat_source[source_idx] = 1  # Heat source at the source vertex
    heat = spsolve(csr_matrix(np.eye(V.shape[0]) - t * L), heat_source)

    # Step 2: Compute the normalized gradient
    grad_u = compute_gradient(V, F, heat)
    grad_u_norm = np.linalg.norm(grad_u, axis=1)
    div = compute_divergence(V, F, grad_u / grad_u_norm[:, None])

    # Step 3: Solve Poisson equation
    distances = spsolve(L, div)
    distances -= distances.min()  # Ensure non-negative distances
    return distances

def compute_cotangent_laplacian(V, F):
    """
    Compute the cotangent Laplace-Beltrami operator for the mesh.

    Parameters:
        V (np.ndarray): Vertices of the mesh (N x 3).
        F (np.ndarray): Faces of the mesh (M x 3).

    Returns:
        csr_matrix: Sparse Laplace-Beltrami operator (N x N).
    """
    # Compute cotangent weights
    I, J, W = [], [], []
    for tri in F:
        for i in range(3):
            i0, i1, i2 = tri[i], tri[(i+1)%3], tri[(i+2)%3]
            v0, v1, v2 = V[i0], V[i1], V[i2]
            cot_alpha = cotangent(v1, v0, v2)
            cot_beta = cotangent(v2, v1, v0)
            W.extend([cot_alpha, cot_beta])
            I.extend([i1, i2])
            J.extend([i2, i1])
    W = np.array(W)
    I = np.array(I)
    J = np.array(J)
    L = csr_matrix((W, (I, J)), shape=(V.shape[0], V.shape[0]))
    return L

def cotangent(v0, v1, v2):
    """Compute cotangent of the angle opposite to edge v1-v2."""
    u = v1 - v0
    v = v2 - v0
    return np.dot(u, v) / np.linalg.norm(np.cross(u, v))

def compute_gradient(V, F, u):
    """Compute the gradient of a scalar field u on the mesh."""
    grad = np.zeros((F.shape[0], 3))
    for i, tri in enumerate(F):
        i0, i1, i2 = tri
        v0, v1, v2 = V[i0], V[i1], V[i2]
        area = np.linalg.norm(np.cross(v1-v0, v2-v0)) / 2
        grad[i] = (u[i1] - u[i0]) * np.cross(v2-v0, [0, 0, 1]) + (u[i2] - u[i0]) * np.cross(v0-v1, [0, 0, 1])
        grad[i] /= (2 * area)
    return grad

def compute_divergence(V, F, grad_u):
    """Compute the divergence of a vector field grad_u on the mesh."""
    div = np.zeros(V.shape[0])
    for i, tri in enumerate(F):
        for j in range(3):
            v0, v1 = tri[j], tri[(j+1)%3]
            div[v0] += np.dot(grad_u[i], V[v1] - V[v0])
    return div

def find_geodesic_path(mesh, geodesic_distances, source_idx, target_idx):
    """
    Find the geodesic path between the source and target vertices.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        geodesic_distances (np.ndarray): Geodesic distances from the source vertex.
        source_idx (int): Index of the source vertex.
        target_idx (int): Index of the target vertex.

    Returns:
        list: List of vertex indices forming the geodesic path.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)

    # Start from the target vertex
    path = [target_idx]
    current_vertex = target_idx

    while current_vertex != source_idx:
        neighbors = []
        for tri in F:
            if current_vertex in tri:
                neighbors.extend(tri)
        neighbors = set(neighbors) - {current_vertex}

        # Find the neighbor with the smallest geodesic distance
        current_vertex = min(neighbors, key=lambda v: geodesic_distances[v])
        path.append(current_vertex)

    path.reverse()  # Reverse the path to start from the source
    return path


# def visualize_geodesic_path(mesh, path):
#     """
#     Visualize the geodesic path on the mesh.

#     Parameters:
#         mesh (o3d.geometry.TriangleMesh): The input mesh.
#         path (list): List of vertex indices forming the geodesic path.
#     """
#     V = np.asarray(mesh.vertices)

#     # Create lines for the path
#     path_edges = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
#     path_edges = np.array(path_edges)

#     # Create a LineSet for the path
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(V),
#         lines=o3d.utility.Vector2iVector(path_edges),
#     )
#     line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in path_edges])  # Red for path

#     # Visualize the mesh and the geodesic path
#     o3d.visualization.draw_geometries([mesh, line_set])

# # Main Execution
# # Create a sphere mesh
# mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=100)
# mesh.compute_vertex_normals()

# print(len(mesh.vertices))

# # Source and target vertices
# source_idx = 10
# target_idx = 19450

# # Compute geodesic distances
# geodesic_distances = compute_heat_geodesic(mesh, source_idx)
# print("D")
# # Highlight geodesic path on the sphere
# path = astar_geodesic_path(mesh, source_idx, target_idx)
# # path = find_geodesic_path(mesh, geodesic_distances, source_idx, target_idx)
# visualize_geodesic_path(mesh, path)
# # highlight_geodesic_path(mesh, source_vertex_idx, target_vertex_idx, geodesic_distances)
