# graph/hand.py
"""
Hand 21-joint skeleton graph definition (HAMER/MediaPipe format).

Joint indices (0-based):
    0: Wrist (root)

    Thumb:
    1: Thumb_CMC    (carpometacarpal)
    2: Thumb_MCP    (metacarpophalangeal)
    3: Thumb_IP     (interphalangeal)
    4: Thumb_Tip    (tip)

    Index:
    5: Index_MCP    (MCP)
    6: Index_PIP    (proximal interphalangeal)
    7: Index_DIP    (distal interphalangeal)
    8: Index_Tip    (tip)

    Middle:
    9: Middle_MCP   (MCP)
    10: Middle_PIP  (PIP)
    11: Middle_DIP  (DIP)
    12: Middle_Tip  (tip)

    Ring:
    13: Ring_MCP    (MCP)
    14: Ring_PIP    (PIP)
    15: Ring_DIP    (DIP)
    16: Ring_Tip    (tip)

    Pinky:
    17: Pinky_MCP   (MCP)
    18: Pinky_PIP   (PIP)
    19: Pinky_DIP   (DIP)
    20: Pinky_Tip   (tip)

Skeleton layout:
                    Wrist (0)
                   /  |  |  |  \
                  /   |  |  |   \
            Thumb(1) Index(5) Middle(9) Ring(13) Pinky(17)
              |        |         |          |         |
              2        6        10         14        18
              |        |         |          |         |
              3        7        11         15        19
              |        |         |          |         |
              4        8        12         16        20
           (Tip)    (Tip)     (Tip)      (Tip)     (Tip)
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


# ========== 21 joints ==========
num_node = 21

# Joint names
joint_names = [
    'Wrist',        # 0  - wrist (root)

    # Thumb (1-4)
    'Thumb_CMC',    # 1  - thumb CMC
    'Thumb_MCP',    # 2  - thumb MCP
    'Thumb_IP',     # 3  - thumb IP
    'Thumb_Tip',    # 4  - thumb tip

    # Index (5-8)
    'Index_MCP',    # 5  - index MCP
    'Index_PIP',    # 6  - index PIP
    'Index_DIP',    # 7  - index DIP
    'Index_Tip',    # 8  - index tip

    # Middle (9-12)
    'Middle_MCP',   # 9  - middle MCP
    'Middle_PIP',   # 10 - middle PIP
    'Middle_DIP',   # 11 - middle DIP
    'Middle_Tip',   # 12 - middle tip

    # Ring (13-16)
    'Ring_MCP',     # 13 - ring MCP
    'Ring_PIP',     # 14 - ring PIP
    'Ring_DIP',     # 15 - ring DIP
    'Ring_Tip',     # 16 - ring tip

    # Pinky (17-20)
    'Pinky_MCP',    # 17 - pinky MCP
    'Pinky_PIP',    # 18 - pinky PIP
    'Pinky_DIP',    # 19 - pinky DIP
    'Pinky_Tip',    # 20 - pinky tip
]

# Self-loops
self_link = [(i, i) for i in range(num_node)]

# Bone edges (child -> parent)
# Format: (child, parent)
inward = [
    # ========== Thumb (tip to wrist) ==========
    (4, 3),    # Thumb_Tip -> Thumb_IP
    (3, 2),    # Thumb_IP -> Thumb_MCP
    (2, 1),    # Thumb_MCP -> Thumb_CMC
    (1, 0),    # Thumb_CMC -> Wrist

    # ========== Index (tip to wrist) ==========
    (8, 7),    # Index_Tip -> Index_DIP
    (7, 6),    # Index_DIP -> Index_PIP
    (6, 5),    # Index_PIP -> Index_MCP
    (5, 0),    # Index_MCP -> Wrist

    # ========== Middle (tip to wrist) ==========
    (12, 11),  # Middle_Tip -> Middle_DIP
    (11, 10),  # Middle_DIP -> Middle_PIP
    (10, 9),   # Middle_PIP -> Middle_MCP
    (9, 0),    # Middle_MCP -> Wrist

    # ========== Ring (tip to wrist) ==========
    (16, 15),  # Ring_Tip -> Ring_DIP
    (15, 14),  # Ring_DIP -> Ring_PIP
    (14, 13),  # Ring_PIP -> Ring_MCP
    (13, 0),   # Ring_MCP -> Wrist

    # ========== Pinky (tip to wrist) ==========
    (20, 19),  # Pinky_Tip -> Pinky_DIP
    (19, 18),  # Pinky_DIP -> Pinky_PIP
    (18, 17),  # Pinky_PIP -> Pinky_MCP
    (17, 0),   # Pinky_MCP -> Wrist
]

# Reverse edges (parent -> child)
outward = [(j, i) for (i, j) in inward]

# Neighbors (bidirectional)
neighbor = inward + outward


class Graph:
    """
    Hand 21-joint skeleton graph (HAMER/MediaPipe format).

    Intended for finger-tapping recognition; compatible with HAMER hand pose output.

    Args:
        labeling_mode: How the adjacency is built
            - 'spatial': spatial graph (default)
            - 'distance': distance-based graph (requires joint coordinates)
        scale: hop count for multi-scale graph (default 1)

    Attributes:
        A: adjacency [3, num_node, num_node] or [num_node, num_node]
           - If 3D: A[0]=self, A[1]=inward, A[2]=outward
           - If 2D: binary adjacency
        num_node: number of joints (21)
    """
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        # Build adjacency
        self.A = self.get_adjacency_matrix(labeling_mode)

        # Optional extras when tools is available
        try:
            # Binary adjacency
            self.A_binary = tools.edge2mat(neighbor, num_node)

            # Normalized adjacency
            self.A_norm = tools.normalize_adjacency_matrix(
                self.A_binary + np.eye(num_node)
            )

            # Multi-scale graph
            self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)
        except:
            # Fallback if tools is missing
            self.A_binary = self._edge2mat_simple(neighbor, num_node)
            self.A_norm = self.A_binary
            self.A_binary_K = self.A_binary

    def get_adjacency_matrix(self, labeling_mode=None):
        """Return adjacency matrix."""
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

        # A[0]: self-links
        for i in range(num_node):
            A[0, i, i] = 1

        # A[1]: inward (child -> parent)
        for i, j in inward:
            A[1, i, j] = 1

        # A[2]: outward (parent -> child)
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

    def get_finger_groups(self):
        """Finger groups for finger-tapping analysis."""
        finger_groups = {
            'wrist': [0],
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20],

            # Finger-tapping key joints
            'thumb_tip': [4],
            'index_tip': [8],
            'tapping_pair': [4, 8],
        }
        return finger_groups

    def get_finger_tips(self):
        """Indices of all fingertip joints."""
        return {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }

    def get_mcp_joints(self):
        """Indices of MCP joints."""
        return {
            'thumb': 2,   # thumb uses MCP index 2 in this topology
            'index': 5,
            'middle': 9,
            'ring': 13,
            'pinky': 17
        }


# ========== Verify skeleton connectivity ==========
def verify_skeleton():
    """Check skeleton connectivity."""
    # Every joint should have a parent except root
    connected_joints = set([0])  # root (wrist)
    for child, parent in inward:
        connected_joints.add(child)
        connected_joints.add(parent)

    all_joints = set(range(num_node))
    disconnected = all_joints - connected_joints

    if disconnected:
        print(f"[WARN] Disconnected joints: {disconnected}")
    else:
        print("[OK] All joints are connected")

    # Cycle check
    children = {}
    for child, parent in inward:
        if parent not in children:
            children[parent] = []
        children[parent].append(child)

    visited = set()
    def dfs(node):
        if node in visited:
            return False  # cycle
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
    print("Hand 21-Joint Skeleton Graph (HAMER/MediaPipe)")
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
        print(f"{i:2d}. {joint_names[child]:15s} ({child:2d}) -> {joint_names[parent]:15s} ({parent:2d})")

    print("\n" + "="*80)
    print("Finger Groups")
    print("="*80)
    finger_groups = graph.get_finger_groups()
    for finger_name, joints in finger_groups.items():
        joint_list = [f"{joint_names[j]}({j})" for j in joints]
        print(f"{finger_name:15s}: {', '.join(joint_list)}")

    print("\n" + "="*80)
    print("Finger Tips")
    print("="*80)
    finger_tips = graph.get_finger_tips()
    for finger_name, tip_idx in finger_tips.items():
        print(f"{finger_name:10s}: {joint_names[tip_idx]:15s} ({tip_idx})")

    print("\n" + "="*80)
    print("MCP Joints (Metacarpophalangeal)")
    print("="*80)
    mcp_joints = graph.get_mcp_joints()
    for finger_name, mcp_idx in mcp_joints.items():
        print(f"{finger_name:10s}: {joint_names[mcp_idx]:15s} ({mcp_idx})")

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

    print("\n" + "="*80)
    print("Finger Tapping Analysis")
    print("="*80)

    thumb_tip = graph.get_finger_tips()['thumb']
    index_tip = graph.get_finger_tips()['index']

    print(f"\nTapping key joints:")
    print(f"   Thumb tip: {joint_names[thumb_tip]} (index={thumb_tip})")
    print(f"   Index tip: {joint_names[index_tip]} (index={index_tip})")

    print(f"\nDistance formula:")
    print(f"   distance = ||keypoints[{thumb_tip}] - keypoints[{index_tip}]||")

    print(f"\nRelevant kinematic chains:")
    print(f"   Thumb: Wrist(0) -> CMC(1) -> MCP(2) -> IP(3) -> Tip(4)")
    print(f"   Index: Wrist(0) -> MCP(5) -> PIP(6) -> DIP(7) -> Tip(8)")

    print("\n" + "="*80)
