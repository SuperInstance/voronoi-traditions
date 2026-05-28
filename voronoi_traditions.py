"""
Voronoi-Based Music Tradition Classifier
=========================================
Maps 10 musical traditions in 3D parameter space (I_vert, I_horiz, I_spectral)
and builds a Voronoi tessellation where each cell = the "territory" of a tradition.

Based on: docs/DIALS-NOT-LAWS.md — the dial model of musical tension.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial import KDTree
from itertools import combinations
import json
import os

# ─── Configuration ─────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)

# ─── Tradition Data ────────────────────────────────────────────
# I_vert, I_horiz from DIALS-NOT-LAWS.md (Appendix A.1)
# I_spectral: estimated from cluster descriptions:
#   - Gamelan traditions: high spectral (inharmonic instruments, ~3.5-4.0)
#   - Gagaku: high spectral (shō mouth organ, ma aesthetic, ~3.8)
#   - Indian/Middle Eastern: moderate spectral (microtonal inflection, ~2.0-2.5)
#   - West African: low-moderate spectral (percussive timbral variation, ~1.5)
#   - Western/Chinese: low spectral (harmonic instruments, clean spectra, ~1.0-1.2)

traditions = {
    'Carnatic':            {'I_vert': 2.767, 'I_horiz': 3.626, 'I_spectral': 2.3, 'cluster': 'Maximal'},
    'Hindustani':          {'I_vert': 2.765, 'I_horiz': 3.451, 'I_spectral': 2.2, 'cluster': 'Maximal'},
    'Turkish Makam':       {'I_vert': 2.828, 'I_horiz': 3.276, 'I_spectral': 2.0, 'cluster': 'Maximal'},
    'Arabic Maqam':        {'I_vert': 2.936, 'I_horiz': 3.101, 'I_spectral': 2.1, 'cluster': 'Maximal'},
    'West African':        {'I_vert': 2.412, 'I_horiz': 3.625, 'I_spectral': 1.5, 'cluster': 'Rhythmic'},
    'Balinese Gamelan':    {'I_vert': 2.308, 'I_horiz': 3.100, 'I_spectral': 3.5, 'cluster': 'Balanced'},
    'Javanese Gamelan':    {'I_vert': 2.308, 'I_horiz': 2.750, 'I_spectral': 3.8, 'cluster': 'Balanced'},
    'Western CP':          {'I_vert': 2.715, 'I_horiz': 2.051, 'I_spectral': 1.0, 'cluster': 'Harmonic'},
    'Chinese Traditional': {'I_vert': 2.318, 'I_horiz': 2.050, 'I_spectral': 1.2, 'cluster': 'Presence'},
    'Japanese Gagaku':     {'I_vert': 2.384, 'I_horiz': 1.700, 'I_spectral': 3.8, 'cluster': 'Presence'},
}

names = list(traditions.keys())
coords = np.array([[t['I_vert'], t['I_horiz'], t['I_spectral']] for t in traditions.values()])
clusters = [t['cluster'] for t in traditions.values()]

# Cluster colors
CLUSTER_COLORS = {
    'Maximal':   '#e74c3c',
    'Rhythmic':  '#f39c12',
    'Balanced':  '#2ecc71',
    'Harmonic':  '#3498db',
    'Presence':  '#9b59b6',
}

colors = [CLUSTER_COLORS[c] for c in clusters]


# ─── Part 1: Voronoi Diagram ──────────────────────────────────

print("=" * 70)
print("VORONOI-BASED MUSIC TRADITION CLASSIFIER")
print("=" * 70)
print()

# For bounded Voronoi, add mirror points
def bounded_voronoi_2d(points_2d, bounds):
    """Create a bounded Voronoi diagram by adding mirror points."""
    xmin, xmax, ymin, ymax = bounds
    mirrored = []
    for p in points_2d:
        mirrored.append(p)
        mirrored.append([2*xmin - p[0], p[1]])
        mirrored.append([2*xmax - p[0], p[1]])
        mirrored.append([p[0], 2*ymin - p[1]])
        mirrored.append([p[0], 2*ymax - p[1]])
    return np.array(mirrored)

# 2D Voronoi for visualization
pts_2d = coords[:, :2]  # I_vert, I_horiz
x_bounds = (pts_2d[:, 0].min() - 0.5, pts_2d[:, 0].max() + 0.5)
y_bounds = (pts_2d[:, 1].min() - 0.5, pts_2d[:, 1].max() + 0.5)

mirrored = bounded_voronoi_2d(pts_2d, (*x_bounds, *y_bounds))
vor_2d = Voronoi(mirrored)

# 3D Voronoi
vor_3d = Voronoi(coords)

print("1. VORONOI TESSELLATION")
print("-" * 40)
print(f"   Traditions: {len(names)}")
print(f"   Dimensions: 3 (I_vert, I_horiz, I_spectral)")
print(f"   3D Voronoi vertices: {len(vor_3d.vertices)}")
print(f"   3D Voronoi ridges: {len(vor_3d.ridge_vertices)}")
print()


# ─── Part 2: Classifier ───────────────────────────────────────

class TraditionClassifier:
    """Classify any point in dial space by nearest tradition (Voronoi cell)."""

    def __init__(self, names, coords, clusters, cluster_colors):
        self.names = names
        self.coords = coords
        self.clusters = clusters
        self.cluster_colors = cluster_colors
        self.tree = KDTree(coords)
        self.vor = Voronoi(coords)

    def classify(self, point):
        """Classify a point. Returns (tradition_name, distance, cluster)."""
        point = np.asarray(point)
        dist, idx = self.tree.query(point)
        return self.names[idx], dist, self.clusters[idx]

    def is_frontier(self, point, threshold=0.5):
        """Is this point in unexplored territory?"""
        _, dist, _ = self.classify(point)
        return dist > threshold

    def frontier_volume(self, n_samples=100000, threshold=0.5):
        """Monte Carlo estimate of frontier fraction."""
        mins = self.coords.min(axis=0) - 1.0
        maxs = self.coords.max(axis=0) + 1.0
        samples = np.random.uniform(mins, maxs, (n_samples, 3))
        dists, _ = self.tree.query(samples)
        return np.mean(dists > threshold)

    def voronoi_cell_volume(self, n_samples=100000):
        """Monte Carlo estimate of each tradition's Voronoi cell volume."""
        mins = self.coords.min(axis=0) - 1.0
        maxs = self.coords.max(axis=0) + 1.0
        samples = np.random.uniform(mins, maxs, (n_samples, 3))
        _, indices = self.tree.query(samples)
        volumes = {}
        total_vol = np.prod(maxs - mins)
        for i, name in enumerate(self.names):
            volumes[name] = np.sum(indices == i) / n_samples * total_vol
        return volumes

    def neighbors(self):
        """Find which traditions share Voronoi boundaries (can hybridize)."""
        neighbor_pairs = set()
        for ridge_points in self.vor.ridge_points:
            i, j = sorted(ridge_points)
            neighbor_pairs.add((self.names[i], self.names[j]))
        return neighbor_pairs

    def optimal_path(self):
        """Greedy nearest-neighbor path through all traditions."""
        visited = [0]
        unvisited = set(range(1, len(self.names)))
        total_dist = 0
        while unvisited:
            current = visited[-1]
            dists = np.linalg.norm(self.coords[list(unvisited)] - self.coords[current], axis=1)
            nearest_idx = list(unvisited)[np.argmin(dists)]
            total_dist += dists.min()
            visited.append(nearest_idx)
            unvisited.remove(nearest_idx)
        return [self.names[i] for i in visited], total_dist


classifier = TraditionClassifier(names, coords, clusters, CLUSTER_COLORS)

# ─── Part 3: Analysis ──────────────────────────────────────────

print("2. VORONOI CELL VOLUMES (territory size)")
print("-" * 40)
volumes = classifier.voronoi_cell_volume(n_samples=200000)
# Sort by volume descending
for name, vol in sorted(volumes.items(), key=lambda x: -x[1]):
    bar = "█" * int(vol * 3)
    print(f"   {name:25s}  vol={vol:5.2f}  {bar}")
print()

print("3. FRONTIER ANALYSIS")
print("-" * 40)
for threshold in [0.3, 0.5, 0.7, 1.0]:
    fv = classifier.frontier_volume(n_samples=200000, threshold=threshold)
    print(f"   threshold={threshold:.1f}  →  frontier volume = {fv*100:.1f}%")
print()

print("4. NEIGHBOR ANALYSIS (Voronoi boundaries = hybridization zones)")
print("-" * 40)
neighbor_pairs = classifier.neighbors()
for a, b in sorted(neighbor_pairs):
    ca = traditions[a]['cluster']
    cb = traditions[b]['cluster']
    same = "✓ SAME" if ca == cb else "✗ CROSS"
    print(f"   {a:25s} ↔ {b:25s}  [{ca:10s} ↔ {cb:10s}]  {same}")
print()

print("5. CENTROID PATH (greedy nearest-neighbor tour)")
print("-" * 40)
path, total_dist = classifier.optimal_path()
print(f"   Total path length: {total_dist:.3f}")
print(f"   Path: {' → '.join(path)}")
print()

# ─── Part 4: Predictions ──────────────────────────────────────

print("6. PREDICTIONS VALIDATION")
print("-" * 40)

# Prediction 1: Carnatic and Hindustani should share a boundary
carnatic_hindustani = ('Carnatic', 'Hindustani') in neighbor_pairs or ('Hindustani', 'Carnatic') in neighbor_pairs
print(f"   Carnatic ↔ Hindustani share boundary: {carnatic_hindustani}  {'✓ PASS' if carnatic_hindustani else '✗ FAIL'}")

# Prediction 2: Western and Gagaku should NOT share a boundary
western_gagaku = ('Western CP', 'Japanese Gagaku') in neighbor_pairs or ('Japanese Gagaku', 'Western CP') in neighbor_pairs
print(f"   Western ↔ Gagaku share boundary:     {western_gagaku}  {'✓ PASS' if not western_gagaku else '✗ FAIL'}")

# Prediction 3: Frontier volume ~82%
fv_05 = classifier.frontier_volume(n_samples=300000, threshold=0.5)
print(f"   Frontier volume (threshold=0.5):      {fv_05*100:.1f}%  {'✓ PASS' if 75 < fv_05*100 < 90 else '~ PARTIAL'}")
print()

# ─── Part 5: Visualizations ────────────────────────────────────

# --- Figure 1: 2D Voronoi diagram with cluster coloring ---
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

from scipy.spatial import Voronoi as Vor
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from shapely.ops import unary_union

# Use scipy Voronoi directly on the 10 points for 2D
vor_plot = Vor(pts_2d)

# Bound the Voronoi regions
bound_box = shapely_box(x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1])

for i, region_idx in enumerate(vor_plot.point_region):
    region = vor_plot.regions[region_idx]
    if -1 in region or len(region) == 0:
        continue
    vertices = vor_plot.vertices[region]
    poly = ShapelyPolygon(vertices)
    if not poly.is_valid:
        poly = poly.buffer(0)
    clipped = poly.intersection(bound_box)
    if clipped.is_empty:
        continue
    if hasattr(clipped, 'exterior'):
        coords_poly = np.array(clipped.exterior.coords)
        patch = MplPolygon(coords_poly, alpha=0.15, facecolor=colors[i],
                           edgecolor='black', linewidth=1.5)
        ax.add_patch(patch)

# Plot tradition points
for i, name in enumerate(names):
    ax.scatter(pts_2d[i, 0], pts_2d[i, 1], c=colors[i], s=200,
               edgecolors='black', linewidth=1.5, zorder=5)
    offset_x, offset_y = 0.03, 0.04
    if name == 'West African':
        offset_y = 0.08
    elif name == 'Carnatic':
        offset_y = -0.12
    elif name == 'Hindustani':
        offset_x = 0.06
        offset_y = 0.02
    elif name == 'Arabic Maqam':
        offset_y = -0.12
    elif name == 'Balinese Gamelan':
        offset_y = 0.08
    elif name == 'Javanese Gamelan':
        offset_y = -0.10
    elif name == 'Chinese Traditional':
        offset_y = -0.10
    ax.annotate(name, (pts_2d[i, 0], pts_2d[i, 1]),
                xytext=(pts_2d[i, 0] + offset_x, pts_2d[i, 1] + offset_y),
                fontsize=9, fontweight='bold', zorder=6)

ax.set_xlabel('I_vert (Pitch/Tuning Information)', fontsize=13)
ax.set_ylabel('I_horiz (Rhythmic Information)', fontsize=13)
ax.set_title('Voronoi Tessellation of Musical Traditions in Dial Space\n(I_vert × I_horiz projection)', fontsize=15)
ax.set_xlim(x_bounds)
ax.set_ylim(y_bounds)

# Legend for clusters
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=12, label=cluster, markeredgecolor='black')
                   for cluster, c in CLUSTER_COLORS.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, title='Cluster')

ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'voronoi_2d_traditions.png'), dpi=150, bbox_inches='tight')
print("   Saved: voronoi_2d_traditions.png")
plt.close(fig)

# --- Figure 2: Frontier map (color the space by nearest tradition) ---
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

resolution = 300
x_grid = np.linspace(x_bounds[0], x_bounds[1], resolution)
y_grid = np.linspace(y_bounds[0], y_bounds[1], resolution)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# Use 2D coords for the frontier map
tree_2d = KDTree(pts_2d)
dists_grid, indices_grid = tree_2d.query(grid_points)
dists_grid = dists_grid.reshape(xx.shape)
indices_grid = indices_grid.reshape(xx.shape)

# Create color map
color_map = np.zeros((*xx.shape, 4))
for i in range(len(names)):
    mask = indices_grid == i
    c = np.array(plt.matplotlib.colors.to_rgba(colors[i]))
    color_map[mask] = c

# Frontier regions get alpha based on distance
alpha_map = np.clip(1.0 - (dists_grid - 0.2) / 1.2, 0.1, 0.8)
color_map[..., 3] = alpha_map

ax.imshow(color_map, extent=[*x_bounds, *y_bounds], origin='lower', aspect='auto')

# Draw frontier contour
frontier_contour = ax.contour(xx, yy, dists_grid, levels=[0.5], colors='white',
                               linewidths=2, linestyles='dashed')
ax.clabel(frontier_contour, fmt='d=0.5 (frontier)', fontsize=10)

for i, name in enumerate(names):
    ax.scatter(pts_2d[i, 0], pts_2d[i, 1], c='white', s=150,
               edgecolors='black', linewidth=2, zorder=5)
    ax.annotate(name, (pts_2d[i, 0], pts_2d[i, 1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
                zorder=6)

ax.set_xlabel('I_vert (Pitch/Tuning Information)', fontsize=13)
ax.set_ylabel('I_horiz (Rhythmic Information)', fontsize=13)
ax.set_title('Frontier Map: Tradition Territories & Unexplored Regions\n(dashed line = frontier boundary, d > 0.5)', fontsize=15)
ax.grid(True, alpha=0.2, color='white')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'frontier_map.png'), dpi=150, bbox_inches='tight')
print("   Saved: frontier_map.png")
plt.close(fig)

# --- Figure 3: 3D scatter with Voronoi vertices ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot traditions
for i, name in enumerate(names):
    ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
               c=colors[i], s=200, edgecolors='black', linewidth=1.5,
               label=f"{name} ({clusters[i]})", zorder=5)
    ax.text(coords[i, 0]+0.02, coords[i, 1]+0.02, coords[i, 2]+0.05,
            name, fontsize=7, zorder=6)

# Plot Voronoi vertices (finite only)
finite_verts = vor_3d.vertices[~np.isinf(vor_3d.vertices).any(axis=1)]
if len(finite_verts) > 0:
    ax.scatter(finite_verts[:, 0], finite_verts[:, 1], finite_verts[:, 2],
               c='gray', s=20, alpha=0.3, zorder=2)

# Plot ridges between neighboring traditions
for ridge_points, ridge_verts in zip(vor_3d.ridge_points, vor_3d.ridge_vertices):
    if -1 in ridge_verts:
        continue
    verts = vor_3d.vertices[ridge_verts]
    if len(verts) == 2 and not np.isinf(verts).any():
        ax.plot(verts[:, 0], verts[:, 1], verts[:, 2],
                'k-', alpha=0.15, linewidth=0.8)

ax.set_xlabel('I_vert (Pitch)', fontsize=11)
ax.set_ylabel('I_horiz (Rhythm)', fontsize=11)
ax.set_zlabel('I_spectral (Timbre)', fontsize=11)
ax.set_title('3D Voronoi Diagram of Musical Traditions\n(I_vert × I_horiz × I_spectral)', fontsize=14)

# Simplified legend (clusters only)
legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                         markersize=10, label=cluster, markeredgecolor='black')
                  for cluster, c in CLUSTER_COLORS.items()]
ax.legend(handles=legend_handles, loc='upper left', fontsize=9, title='Cluster')

ax.view_init(elev=25, azim=135)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'voronoi_3d_traditions.png'), dpi=150, bbox_inches='tight')
print("   Saved: voronoi_3d_traditions.png")
plt.close(fig)

# --- Figure 4: Neighbor graph / hybridization map ---
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Draw connections between neighbors
for a, b in neighbor_pairs:
    i, j = names.index(a), names.index(b)
    ca, cb = clusters[i], clusters[j]
    style = '-' if ca == cb else '--'
    lw = 2 if ca == cb else 1
    ax.plot([pts_2d[i, 0], pts_2d[j, 0]], [pts_2d[i, 1], pts_2d[j, 1]],
            style, color='gray', linewidth=lw, alpha=0.5, zorder=2)

for i, name in enumerate(names):
    ax.scatter(pts_2d[i, 0], pts_2d[i, 1], c=colors[i], s=300,
               edgecolors='black', linewidth=2, zorder=5)
    ax.annotate(name, (pts_2d[i, 0], pts_2d[i, 1]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, fontweight='bold', zorder=6)

ax.set_xlabel('I_vert (Pitch/Tuning Information)', fontsize=13)
ax.set_ylabel('I_horiz (Rhythmic Information)', fontsize=13)
ax.set_title('Hybridization Map: Voronoi Boundary Connections\n(solid = same-cluster, dashed = cross-cluster)', fontsize=15)
ax.grid(True, alpha=0.3)

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=12, label=cluster, markeredgecolor='black')
                   for cluster, c in CLUSTER_COLORS.items()]
legend_elements.extend([
    Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='Same-cluster boundary'),
    Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Cross-cluster boundary'),
])
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'hybridization_map.png'), dpi=150, bbox_inches='tight')
print("   Saved: hybridization_map.png")
plt.close(fig)

# --- Figure 5: Distance heatmap ---
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(coords))
im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f'{dist_matrix[i, j]:.2f}', ha='center', va='center',
                fontsize=7, color='white' if dist_matrix[i, j] > 1.2 else 'black')
ax.set_title('Euclidean Distance Between Traditions in 3D Dial Space\n(I_vert × I_horiz × I_spectral)', fontsize=14)
fig.colorbar(im, label='Euclidean Distance')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'distance_heatmap.png'), dpi=150, bbox_inches='tight')
print("   Saved: distance_heatmap.png")
plt.close(fig)


# ─── Summary JSON ──────────────────────────────────────────────

results = {
    'traditions': {name: traditions[name] for name in names},
    'voronoi_cell_volumes': {k: round(v, 3) for k, v in sorted(volumes.items(), key=lambda x: -x[1])},
    'frontier_volumes': {},
    'neighbor_pairs': sorted([list(p) for p in neighbor_pairs]),
    'optimal_path': path,
    'optimal_path_length': round(total_dist, 3),
    'predictions': {
        'carnatic_hindustani_boundary': carnatic_hindustani,
        'western_gagaku_no_boundary': not western_gagaku,
        'frontier_volume_pct': round(fv_05 * 100, 1),
    },
    'distance_matrix': {names[i]: {names[j]: round(dist_matrix[i, j], 3) for j in range(len(names))}
                        for i in range(len(names))},
}

for threshold in [0.3, 0.5, 0.7, 1.0]:
    results['frontier_volumes'][str(threshold)] = round(
        classifier.frontier_volume(n_samples=200000, threshold=threshold) * 100, 1)

with open(os.path.join(OUTPUT_DIR, 'voronoi_results.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("   Saved: voronoi_results.json")

print()
print("=" * 70)
print("DONE. All plots saved to:", OUTPUT_DIR)
print("=" * 70)
