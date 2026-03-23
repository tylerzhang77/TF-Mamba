# graph/hybrik.py
"""
HybrIK / SMPL 24-joint skeleton graph definition.

Joint indices (0-based, SMPL standard order):
    0: Pelvis (root)
    1: L_Hip
    2: R_Hip
    3: Spine1
    4: L_Knee
    5: R_Knee
    6: Spine2
    7: L_Ankle
    8: R_Ankle
    9: Spine3
    10: L_Foot (left toes)
    11: R_Foot (right toes)
    12: Neck
    13: L_Collar
    14: R_Collar
    15: Head
    16: L_Shoulder
    17: R_Shoulder
    18: L_Elbow
    19: R_Elbow
    20: L_Wrist
    21: R_Wrist
    22: L_Hand
    23: R_Hand

Skeleton layout:
                        Head (15)
                          |
                       Neck (12)
                          |
                       Spine3 (9)
                       /  |  \
          L_Collar(13)  Spine2(6)  R_Collar(14)
              |           |             |
        L_Shoulder(16) Spine1(3)  R_Shoulder(17)
              |           |             |
         L_Elbow(18)  Pelvis(0)   R_Elbow(19)
              |        /    \           |
         L_Wrist(20) L_Hip(1) R_Hip(2) R_Wrist(21)
              |        |        |           |
         L_Hand(22) L_Knee(4) R_Knee(5) R_Hand(23)
                      |          |
                 L_Ankle(7)  R_Ankle(8)
                      |          |
                 L_Foot(10)  R_Foot(11)
"""

import numpy as np
import sys
sys.path.extend(['../'])

try:
    from graph import tools
except ImportError:
    try:
        from . import tools
    except:
        print("[WARNING] Could not import graph.tools, using fallback implementations")


# ========== 24 joints ==========
num_node = 24

# Joint names (SMPL order)
joint_names = [
    'Pelvis',      # 0  - pelvis (root)
    'L_Hip',       # 1  - left hip
    'R_Hip',       # 2  - right hip
    'Spine1',      # 3  - spine1
    'L_Knee',      # 4  - left knee
    'R_Knee',      # 5  - right knee
    'Spine2',      # 6  - spine2
    'L_Ankle',     # 7  - left ankle
    'R_Ankle',     # 8  - right ankle
    'Spine3',      # 9  - spine3
    'L_Foot',      # 10 - left foot / toes
    'R_Foot',      # 11 - right foot / toes
    'Neck',        # 12 - neck
    'L_Collar',    # 13 - left collar
    'R_Collar',    # 14 - right collar
    'Head',        # 15 - head
    'L_Shoulder',  # 16 - left shoulder
    'R_Shoulder',  # 17 - right shoulder
    'L_Elbow',     # 18 - left elbow
    'R_Elbow',     # 19 - right elbow
    'L_Wrist',     # 20 - left wrist
    'R_Wrist',     # 21 - right wrist
    'L_Hand',      # 22 - left hand
    'R_Hand',      # 23 - right hand
]

# Self-loops
self_link = [(i, i) for i in range(num_node)]

# Bone edges (child -> parent)
# Format: (child, parent)
# Standard SMPL parent table
inward = [
    # ========== Torso (head to pelvis) ==========
    (15, 12),  # Head -> Neck
    (12, 9),   # Neck -> Spine3
    (9, 6),    # Spine3 -> Spine2
    (6, 3),    # Spine2 -> Spine1
    (3, 0),    # Spine1 -> Pelvis

    # ========== Left leg (foot to pelvis) ==========
    (10, 7),   # L_Foot -> L_Ankle
    (7, 4),    # L_Ankle -> L_Knee
    (4, 1),    # L_Knee -> L_Hip
    (1, 0),    # L_Hip -> Pelvis

    # ========== Right leg (foot to pelvis) ==========
    (11, 8),   # R_Foot -> R_Ankle
    (8, 5),    # R_Ankle -> R_Knee
    (5, 2),    # R_Knee -> R_Hip
    (2, 0),    # R_Hip -> Pelvis

    # ========== Left arm (hand to torso) ==========
    (13, 12),  # L_Collar -> Neck
    (16, 13),  # L_Shoulder -> L_Collar
    (18, 16),  # L_Elbow -> L_Shoulder
    (20, 18),  # L_Wrist -> L_Elbow
    (22, 20),  # L_Hand -> L_Wrist

    # ========== Right arm (hand to torso) ==========
    (14, 12),  # R_Collar -> Neck
    (17, 14),  # R_Shoulder -> R_Collar
    (19, 17),  # R_Elbow -> R_Shoulder
    (21, 19),  # R_Wrist -> R_Elbow
    (23, 21),  # R_Hand -> R_Wrist
]

# Reverse edges (parent -> child)
outward = [(j, i) for (i, j) in inward]

# Neighbors (bidirectional)
neighbor = inward + outward


class Graph:
    """
    HybrIK / SMPL 24-joint skeleton graph.

    For Parkinson scale–related actions; compatible with HybrIK pose output.

    Args:
        labeling_mode: adjacency construction mode
            - 'spatial': spatial graph (default)
            - 'distance': distance-based graph (requires joint coordinates)
        scale: hop count for multi-scale graph (default 1)

    Attributes:
        A: adjacency [3, num_node, num_node] or [num_node, num_node]
           - If 3D: A[0]=self, A[1]=inward, A[2]=outward
           - If 2D: binary adjacency
        num_node: number of joints (24)
    """
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(labeling_mode)

        try:
            self.A_binary = tools.edge2mat(neighbor, num_node)

            self.A_norm = tools.normalize_adjacency_matrix(
                self.A_binary + np.eye(num_node)
            )

            self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)
        except:
            self.A_binary = self._edge2mat_simple(neighbor, num_node)
            self.A_norm = self.A_binary
            self.A_binary_K = self.A_binary

    def get_adjacency_matrix(self, labeling_mode=None):
        """
        Return adjacency matrix.

        Args:
            labeling_mode: 'spatial' or other

        Returns:
            A: [3, num_node, num_node] or [num_node, num_node]
        """
        if labeling_mode is None:
            return self.A

        if labeling_mode == 'spatial':
            try:
                A = tools.get_spatial_graph(num_node, self_link, inward, outward)
            except:
                A = self._get_spatial_graph_simple()
        else:
            raise ValueError(f"Unknown labeling_mode: {labeling_mode}")

        return A

    def _get_spatial_graph_simple(self):
        """Fallback: simple 3-layer spatial graph."""
        A = np.zeros((3, num_node, num_node))

        for i in range(num_node):
            A[0, i, i] = 1

        for i, j in inward:
            A[1, i, j] = 1

        for i, j in outward:
            A[2, i, j] = 1

        return A

    def _edge2mat_simple(self, edges, num_node):
        """Fallback: edge list to adjacency matrix."""
        A = np.zeros((num_node, num_node))
        for i, j in edges:
            A[i, j] = 1
        return A

    def get_edge_list(self):
        """Return inward edge list (for visualization)."""
        return inward

    def get_joint_names(self):
        """Return joint name list."""
        return joint_names

    def get_body_parts(self):
        """Body-part groups for clinical motion analysis."""
        body_parts = {
            'torso': [0, 3, 6, 9, 12, 15],
            'left_leg': [1, 4, 7, 10],
            'right_leg': [2, 5, 8, 11],
            'left_arm': [13, 16, 18, 20, 22],
            'right_arm': [14, 17, 19, 21, 23],
            'left_foot': [10],   # toe tapping (left)
            'right_foot': [11],  # toe tapping (right)
            'left_hand': [22],
            'right_hand': [23],
        }
        return body_parts

    def get_limb_pairs(self):
        """Left-right symmetric joint pairs for symmetry analysis."""
        limb_pairs = {
            'hips': (1, 2),
            'knees': (4, 5),
            'ankles': (7, 8),
            'feet': (10, 11),
            'collars': (13, 14),
            'shoulders': (16, 17),
            'elbows': (18, 19),
            'wrists': (20, 21),
            'hands': (22, 23),
        }
        return limb_pairs


# ========== Verify skeleton connectivity ==========
def verify_skeleton():
    """Check skeleton connectivity."""
    connected_joints = set([0])  # root
    for child, parent in inward:
        connected_joints.add(child)
        connected_joints.add(parent)

    all_joints = set(range(num_node))
    disconnected = all_joints - connected_joints

    if disconnected:
        print(f"[WARN] Disconnected joints: {disconnected}")
    else:
        print("[OK] All joints are connected")

    children = {}
    for child, parent in inward:
        if parent not in children:
            children[parent] = []
        children[parent].append(child)

    visited = set()
    def dfs(node):
        if node in visited:
            return False
        visited.add(node)
        if node in children:
            for child in children[node]:
                if not dfs(child):
                    return False
        return True

    if dfs(0):
        print("[OK] No cycles detected")
    else:
        print("[WARN] Cycle detected in skeleton")

    return len(disconnected) == 0


# ========== Self-test ==========
if __name__ == '__main__':
    print("="*80)
    print("HybrIK / SMPL 24-Joint Skeleton Graph")
    print("="*80)

    graph = Graph(labeling_mode='spatial')

    print(f"\n[OK] Number of nodes: {graph.num_node}")
    print(f"[OK] Adjacency matrix shape: {graph.A.shape}")
    print(f"[OK] Number of edges: {len(graph.inward)}")

    print("\n" + "="*80)
    print("Skeleton Verification")
    print("="*80)
    verify_skeleton()

    print("\n" + "="*80)
    print("Skeleton Connections (Inward)")
    print("="*80)
    for i, (child, parent) in enumerate(inward, 1):
        print(f"{i:2d}. {joint_names[child]:12s} ({child:2d}) -> {joint_names[parent]:12s} ({parent:2d})")

    print("\n" + "="*80)
    print("Body Parts Grouping")
    print("="*80)
    body_parts = graph.get_body_parts()
    for part_name, joints in body_parts.items():
        joint_list = [f"{joint_names[j]}({j})" for j in joints]
        print(f"{part_name:15s}: {', '.join(joint_list)}")

    print("\n" + "="*80)
    print("Symmetric Limb Pairs")
    print("="*80)
    limb_pairs = graph.get_limb_pairs()
    for limb_name, (left, right) in limb_pairs.items():
        print(f"{limb_name:10s}: L={joint_names[left]:12s}({left:2d}) <-> R={joint_names[right]:12s}({right:2d})")

    print("\n" + "="*80)
    print("Adjacency Matrix Statistics")
    print("="*80)
    if graph.A.ndim == 3:
        print(f"A[0] (self-link) non-zero: {np.count_nonzero(graph.A[0])}")
        print(f"A[1] (inward) non-zero: {np.count_nonzero(graph.A[1])}")
        print(f"A[2] (outward) non-zero: {np.count_nonzero(graph.A[2])}")

        print("\nA[1] (inward) - First 10x10 block:")
        print(graph.A[1, :10, :10].astype(int))
    else:
        print(f"Binary adjacency matrix non-zero: {np.count_nonzero(graph.A)}")

    print("\n" + "="*80)
    print("Graph initialization complete!")
    print("="*80)
