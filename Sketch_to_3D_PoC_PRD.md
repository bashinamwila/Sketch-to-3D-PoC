# Sketch-to-3D Architectural Model — Proof of Concept
## Product Requirements Document

---

| Field | Value |
|---|---|
| **Version** | 1.0 — Proof of Concept |
| **Status** | Draft — Internal Review |
| **Date** | February 2026 |
| **Classification** | Confidential |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Objectives & Success Criteria](#3-objectives--success-criteria)
4. [System Architecture](#4-system-architecture)
   - [Stage 1 — Image Preprocessing](#41-stage-1--image-preprocessing)
   - [Stage 2 — Straight Line Extraction](#42-stage-2--straight-line-extraction)
   - [Stage 3 — Curve Extraction & Statistical Best-Fit](#43-stage-3--curve-extraction--statistical-best-fit)
   - [Stage 4 — Topology Reconstruction](#44-stage-4--topology-reconstruction)
   - [Stage 5 — CadQuery 3D Model Generation](#45-stage-5--cadquery-3d-model-generation)
   - [Stage 6 — Camera-Aligned Snapshot](#46-stage-6--camera-aligned-snapshot)
5. [Data Flow & Module Interfaces](#5-data-flow--module-interfaces)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Technology Stack](#8-technology-stack)
9. [Test Plan](#9-test-plan)
10. [Risk Register](#10-risk-register)
11. [Development Milestones](#11-development-milestones)
12. [Out of Scope](#12-out-of-scope-for-poc)
13. [Open Questions](#13-open-questions)
14. [Appendix — Algorithm Reference](#14-appendix--algorithm-quick-reference)

---

## 1. Executive Summary

This document defines the requirements, architecture, and acceptance criteria for a Proof of Concept (PoC) system that automatically converts hand-drawn architectural sketches into accurate CadQuery 3D models. The system relies exclusively on classical computer vision, statistical best-fit algorithms, and computational geometry — with no dependence on large language models or neural networks for geometry inference.

The PoC validates a proprietary pipeline derived from a manually-executed workflow that achieved **95% geometric accuracy** when converting hand-drawn sketches to photorealistic renders via an intermediate 3D model. The PoC's primary goal is to automate every stage of that pipeline and prove it scales beyond single-user, manual operation.

> **Core Value Proposition**
>
> Replace manual CAD model construction with a deterministic, automated computer vision pipeline that extracts geometric primitives directly from hand-drawn architectural sketches, fits statistically optimal curves and lines to noisy sketch data, reconstructs a topologically correct 3D building model in CadQuery, and produces camera-aligned renders ready for downstream photorealistic generation — at scale.

---

## 2. Background & Motivation

### 2.1 The Proven Manual Workflow

The founding insight of this project is a manually-executed pipeline that demonstrated 95% geometric accuracy in sketch-to-photorealistic-render conversion:

1. A hand-drawn architectural sketch is used as the source of truth.
2. The sketch is interpreted and reconstructed as a CadQuery parametric 3D model.
3. A snapshot of the 3D model is taken at the same perspective angle as the sketch.
4. The snapshot is processed into a clean CAD-style line drawing.
5. The line drawing is fed into a ControlNet-conditioned Stable Diffusion pipeline to produce a photorealistic render.

This workflow produces substantially higher geometric fidelity than direct sketch-to-render approaches because the intermediate 3D model enforces structural correctness — correct proportions, accurate perspective, and coherent depth — which no 2D image-to-image method can reliably infer.

### 2.2 The Scaling Problem

The manual workflow does not scale. Each model requires an experienced CadQuery user spending 30–90 minutes per sketch. The PoC must replace that manual step with an automated pipeline capable of processing a sketch in **under 60 seconds** with no human CAD authorship required.

### 2.3 Why Classical CV Over Neural Approaches

Neural sketch-to-3D models (Shap-E, Zero-1-to-3, One-2-3-45) are insufficiently reliable for architectural precision. They hallucinate geometry, cannot guarantee structural coherence, and produce non-parametric outputs that cannot be post-edited. The classical pipeline chosen for this PoC is deterministic, auditable, and produces CadQuery parametric models that remain editable after generation.

---

## 3. Objectives & Success Criteria

### 3.1 PoC Objectives

- Automate extraction of straight-line geometry from hand-drawn architectural sketches using classical edge detection and Hough transforms.
- Automate extraction and best-fit of curved geometry (arcs, ellipses, B-splines) from residual curve pixels using statistical fitting algorithms.
- Reconstruct a topologically correct 2D floor plan and elevation profile from extracted geometry using polygon detection.
- Generate a parametric CadQuery 3D model from the reconstructed topology.
- Render a camera-aligned snapshot of the 3D model at a perspective matching the original sketch.
- Validate that the resulting model meets the geometric accuracy threshold required for downstream ControlNet photorealistic rendering.

### 3.2 Quantitative Success Criteria

| Metric | Target (PoC) | Measurement Method |
|---|---|---|
| Wall position accuracy | ≤ 5% deviation from sketch | IoU of projected footprint vs sketch outline |
| Straight line detection recall | ≥ 90% of sketch lines detected | Manual review on 20-sketch test set |
| Curve fit RMS error | ≤ 3px at 1000px image width | RANSAC residuals + splprep residuals |
| 3D model generation success rate | ≥ 80% of test sketches | CadQuery execution without exception |
| Camera angle match score | ≥ 85% vanishing point alignment | Homography overlap metric |
| End-to-end processing time | ≤ 60 seconds per sketch | Wall clock on reference hardware |
| ControlNet render fidelity | ≥ 90% structural match (manual review) | Overlay comparison with original sketch |

---

## 4. System Architecture

The pipeline is a sequential, modular system of six stages. Each stage produces a well-defined output consumed by the next. Stages are implemented as independent Python modules to allow isolated testing, replacement, and future parallelisation.

```
Stage 1: Image Preprocessing
        ↓
Stage 2: Straight Line Extraction
        ↓
Stage 3: Curve Extraction & Best-Fit
        ↓
Stage 4: Topology Reconstruction
        ↓
Stage 5: CadQuery 3D Model Generation
        ↓
Stage 6: Camera-Aligned Snapshot
        ↓
   [Lineart PNG + Depth Map]
   → ControlNet / Stable Diffusion (out of scope for PoC)
```

---

### 4.1 Stage 1 — Image Preprocessing

#### 4.1.1 Purpose

Transform a raw sketch photograph or scan into a clean, normalised binary edge image suitable for geometric extraction. This stage is the foundation of all downstream accuracy.

#### 4.1.2 Input

- Raw sketch image: JPEG, PNG, or TIFF, minimum 800px on the short edge, maximum 4000px on the long edge.
- Optional user metadata: known scale reference (e.g. `front wall = 12 metres`), sketch orientation hint.

#### 4.1.3 Processing Steps

1. Convert to greyscale (`cv2.cvtColor`, `COLOR_BGR2GRAY`).
2. **Perspective correction:** detect the four outer corners of the sketch paper using `cv2.findContours` on a thresholded version, compute the homography with `cv2.getPerspectiveTransform`, and warp to a flat rectangular image.
3. **Adaptive histogram equalisation** (`cv2.createCLAHE`, `clipLimit=2.0`) to normalise uneven pencil pressure and lighting.
4. **Noise reduction:** `cv2.GaussianBlur` with kernel `(5,5)`, `sigma=0`. Bilateral filter as fallback for sketches with heavy texture.
5. **Edge detection:** `cv2.Canny` with lower threshold 40, upper threshold 120. Thresholds are auto-tuned per image using Otsu's method on the greyscale histogram.
6. **Morphological closing** (`cv2.morphologyEx`, `MORPH_CLOSE`, 3×3 kernel) to bridge small gaps in hand-drawn lines.
7. Resize to a canonical processing width of **1200px**, preserving aspect ratio.

#### 4.1.4 Output

- Binary edge image: `numpy` ndarray, `uint8`, single channel.
- Scale factor: `float`, pixels-per-metre (if reference provided) or `None`.
- Orientation metadata: `dict` with detected vanishing point candidates.

---

### 4.2 Stage 2 — Straight Line Extraction

#### 4.2.1 Purpose

Extract all straight line segments from the edge image, clean them statistically, cluster duplicates, snap to architectural angles, and compute precise intersection vertices. These vertices and lines form the primary geometric skeleton of the building.

#### 4.2.2 Probabilistic Hough Line Detection

`cv2.HoughLinesP` is applied to the edge image with the following parameters, tuned for architectural sketch characteristics:

| Parameter | Value | Rationale |
|---|---|---|
| `rho` | 1 px | Sub-pixel precision on normalised image |
| `theta` | π/360 rad (0.5°) | Detect lines at half-degree precision |
| `threshold` | 60 | Minimum votes; reduces false positives from hatching |
| `minLineLength` | 40 px | Rejects texture and shading strokes |
| `maxLineGap` | 12 px | Bridges typical hand-drawn gaps in wall lines |

#### 4.2.3 RANSAC Best-Fit Refinement

Raw HoughLinesP detections are noisy. Each detected line segment is extended and nearby edge pixels are collected within a 4px corridor. `sklearn.linear_model.RANSACRegressor` is applied to those pixels to compute the statistically optimal line fit, discarding outlier pixels from hatching or smearing. RANSAC `residual_threshold` is set to `2.0px`. Lines with fewer than 15 inlier pixels after RANSAC are discarded as spurious.

#### 4.2.4 Line Clustering with DBSCAN

Multiple detected segments often represent the same wall. Lines are encoded as `(angle, perpendicular_distance_from_centre)` feature vectors and clustered with `sklearn.cluster.DBSCAN` (`eps=0.08` in normalised space, `min_samples=2`). Each cluster is merged into a single canonical line by averaging the RANSAC-fitted parameters of its members, weighted by inlier count.

#### 4.2.5 Architectural Angle Snapping

Fitted line angles are snapped to the nearest architectural canonical angle from the set `{0°, 30°, 45°, 60°, 90°, 120°, 135°, 150°}` if the deviation is within **5°**. This eliminates accumulated angular error from hand drawing. Lines whose angle is not within 5° of any canonical angle are retained as-is (e.g. irregular roof pitches).

```python
def snap_angle(angle, tolerance=5):
    architectural_angles = [0, 30, 45, 60, 90, 120, 135, 150, 180]
    nearest = min(architectural_angles, key=lambda a: abs(angle - a))
    if abs(angle - nearest) < tolerance:
        return nearest
    return angle
```

#### 4.2.6 Vanishing Point Detection & Perspective Metadata

All near-horizontal lines are extended and their pairwise intersections computed. The two dominant vanishing point clusters (identified by DBSCAN on intersection coordinates) define the drawing's perspective projection. These vanishing points are stored as metadata and used in Stage 6 for camera alignment.

#### 4.2.7 Intersection Vertex Computation

All pairs of non-parallel canonical lines are tested for intersection using the standard cross-product formula. Intersections within the image bounds are collected as candidate vertices. Vertices within 8px of each other are merged by averaging.

```python
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:   # parallel
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    return (x, y)
```

#### 4.2.8 Output

- Canonical line list: `list` of `(x1, y1, x2, y2)` tuples in image pixel coordinates.
- Vertex set: `numpy` ndarray of shape `(N, 2)`.
- Vanishing points: list of two `(x, y)` tuples.
- Line mask: binary image of all detected canonical lines (consumed by Stage 3).

---

### 4.3 Stage 3 — Curve Extraction & Statistical Best-Fit

#### 4.3.1 Purpose

Extract all curved geometry from the sketch — arcs, ellipses, circular elements, and organic B-spline curves — that was not captured by straight line detection. This stage handles the architectural elements most resistant to linear methods: curved walls, arched openings, bay windows, domes, and parametric roof profiles.

#### 4.3.2 Residual Curve Pixel Extraction

The line mask from Stage 2 is morphologically dilated (kernel 5×5) to account for line width, then subtracted from the Canny edge image. The result is a residual edge image containing only pixels not explained by any detected straight line. Residual clusters smaller than 50 pixels are discarded as noise.

```python
def extract_curve_pixels(edge_image, line_mask, dilation_kernel=5):
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated_mask = cv2.dilate(line_mask, kernel)
    return cv2.subtract(edge_image, dilated_mask)
```

#### 4.3.3 Stroke Clustering

Residual edge pixels are clustered into individual curve strokes using `DBSCAN` with `eps=8px`, `min_samples=10`. This groups spatially proximate pixels into strokes without requiring the number of curves to be specified in advance. DBSCAN noise points (`label = -1`) are discarded.

#### 4.3.4 Point Ordering Along Stroke

Each DBSCAN cluster is an unordered point set. Points are ordered sequentially along the curve by a greedy nearest-neighbour traversal seeded from the leftmost point, using `scipy.spatial.KDTree` for efficient nearest-neighbour queries.

```python
def order_curve_points(points):
    ordered = [points[np.argmin(points[:, 1])]]
    remaining = set(range(len(points)))
    remaining.remove(np.argmin(points[:, 1]))
    tree = KDTree(points)
    while remaining:
        last = ordered[-1]
        _, idxs = tree.query(last, k=len(points))
        for idx in idxs:
            if idx in remaining:
                ordered.append(points[idx])
                remaining.remove(idx)
                break
    return np.array(ordered)
```

#### 4.3.5 Curve Type Classification

Before fitting, each ordered stroke is classified using three geometric metrics:

| Metric | Formula | Classification Rule |
|---|---|---|
| Compactness | 4π·Area / Perimeter² | > 0.70 → Ellipse / Circle |
| Arc-chord ratio | Arc length / Chord length | < 1.5 → Arc; ≥ 1.5 → Spline |
| Aspect ratio | max(x,y range) / min(x,y range) | > 4.0 → Spline regardless |

```python
def classify_curve(ordered_points):
    pts = ordered_points.astype(float)
    hull = ConvexHull(pts)
    compactness = (4 * np.pi * hull.volume) / (hull.area ** 2)
    chord = np.linalg.norm(pts[-1] - pts[0])
    arc = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    arc_chord_ratio = arc / (chord + 1e-5)
    x_range = pts[:,1].max() - pts[:,1].min()
    y_range = pts[:,0].max() - pts[:,0].min()
    aspect = max(x_range, y_range) / (min(x_range, y_range) + 1e-5)

    if compactness > 0.70:    return 'ellipse'
    if aspect > 4.0:          return 'spline'
    if arc_chord_ratio < 1.5: return 'arc'
    return 'spline'
```

#### 4.3.6 Curvature Breakpoint Detection for Compound Curves

Many architectural curves are compound: a curved arch meeting straight wall segments. Curvature breakpoints are detected by computing the cross-product of successive direction vectors along the ordered stroke using a sliding window of 5 points. Locations where absolute curvature exceeds `mean + 2·std` are marked as breakpoints. The stroke is split at breakpoints, and each sub-segment is classified and fitted independently.

```python
def find_curvature_breakpoints(ordered_points, window=5):
    pts = ordered_points.astype(float)
    curvatures = []
    for i in range(window, len(pts) - window):
        v1 = pts[i] - pts[i - window]
        v2 = pts[i + window] - pts[i]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        curvatures.append(abs(cross / norm))
    threshold = np.mean(curvatures) + 2 * np.std(curvatures)
    return np.where(np.array(curvatures) > threshold)[0] + window
```

#### 4.3.7 Arc and Ellipse Fitting

Strokes classified as arcs or ellipses are fitted using `cv2.fitEllipse`, which applies the Bookstein algebraic ellipse fitting method (minimum 5 points). Returns centre, semi-axes lengths, and rotation angle. `scipy.optimize.least_squares` with the Taubin circle formulation is used as a fallback for noisy arcs.

```python
def fit_arc(ordered_points):
    pts = ordered_points[:, ::-1].astype(np.float32)   # (x, y)
    if len(pts) >= 5:
        ellipse = cv2.fitEllipse(pts)
        center, axes, angle = ellipse
        return {'type': 'ellipse', 'center': center, 'axes': axes, 'angle': angle}

def fit_circle_taubin(points):
    x, y = points[:,1].astype(float), points[:,0].astype(float)
    def residuals(p):
        cx, cy, r = p
        return np.sqrt((x-cx)**2 + (y-cy)**2) - r
    cx0, cy0 = x.mean(), y.mean()
    r0 = np.sqrt((x-cx0)**2 + (y-cy0)**2).mean()
    result = least_squares(residuals, [cx0, cy0, r0])
    cx, cy, r = result.x
    return {'type': 'circle', 'center': (cx, cy), 'radius': r}
```

#### 4.3.8 B-Spline Fitting

Strokes classified as splines are fitted using `scipy.interpolate.splprep`, which fits a parametric B-spline of degree 3. The smoothing parameter `s` is set **adaptively**:

```
s = N_points × σ_line
```

where `σ_line` is the mean RANSAC residual from Stage 2 — a proxy for the sketch's overall line roughness. Higher roughness → more smoothing, creating an adaptive response to sketch quality.

```python
def fit_bspline(ordered_points, sigma_line):
    x = ordered_points[:, 1].astype(float)
    y = ordered_points[:, 0].astype(float)
    s = len(x) * sigma_line          # adaptive smoothing
    tck, u = splprep([x, y], s=s, k=3)
    u_fine = np.linspace(0, 1, 500)
    x_smooth, y_smooth = splev(u_fine, tck)
    ctrl_pts = np.array(splev(np.linspace(0, 1, 8), tck)).T
    return {
        'type': 'bspline',
        'tck': tck,
        'control_points': ctrl_pts,
        'smooth_curve': list(zip(x_smooth, y_smooth))
    }
```

#### 4.3.9 Output

- Fitted curve list: `list` of dicts containing `type`, fitted parameters, and pixel-coordinate control points.
- Compound segment map: `dict` mapping original stroke index to list of fitted sub-segments.

---

### 4.4 Stage 4 — Topology Reconstruction

#### 4.4.1 Purpose

Convert the unstructured set of lines and curves into a topologically coherent architectural description: an identified building footprint, classified walls, located openings, and a roof profile.

#### 4.4.2 Floor Plan Polygon Detection

All canonical straight lines from Stage 2 are converted to Shapely `LineString` objects and unioned using `shapely.ops.unary_union`. `shapely.ops.polygonize` extracts all enclosed polygonal regions. Regions smaller than 1% of image area are discarded. The largest polygon is designated the **building exterior footprint**. Interior polygons represent rooms or structural divisions.

```python
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString

def detect_floor_plan(line_list):
    lines = [LineString([(x1,y1),(x2,y2)]) for x1,y1,x2,y2 in line_list]
    merged = unary_union(lines)
    polygons = list(polygonize(merged))
    polygons = [p for p in polygons if p.area > 0.01 * total_image_area]
    footprint = max(polygons, key=lambda p: p.area)
    rooms = [p for p in polygons if p != footprint]
    return footprint, rooms
```

#### 4.4.3 Elevation vs. Plan View Detection

The pipeline determines whether the sketch is a **floor plan** (top-down) or **elevation** (front/side) since 3D reconstruction differs fundamentally between them.

- If > 60% of lines are horizontal/near-horizontal and no strong vanishing point convergence is detected → **floor plan**
- Otherwise → **elevation**
- When both are present on one sheet → user is prompted to identify each region.

#### 4.4.4 Coordinate Normalisation

All pixel coordinates are converted to a real-world metric system. If a scale reference was provided, the pixels-per-metre ratio is applied directly. Otherwise, the building footprint is normalised so its longest dimension = **12 metres** (a common residential building width), adjustable via a scale slider. The coordinate system origin is placed at the lower-left vertex. CadQuery's right-hand coordinate system is applied (y-axis inverted from image coordinates).

```python
def px_to_world(px_point, origin_px, scale):
    x = (px_point[0] - origin_px[0]) * scale
    y = (px_point[1] - origin_px[1]) * scale
    return (x, -y)   # CadQuery right-hand coords
```

#### 4.4.5 Architectural Element Classification

| Element | Detection Heuristic |
|---|---|
| Exterior walls | Segments forming the outer boundary polygon |
| Interior walls | Segments forming inner polygon boundaries |
| Doors | Gaps of 0.6–2.5 m in exterior wall segments |
| Windows | Gaps of 0.4–1.8 m at mid-wall height in elevation |
| Roof lines | Lines above the wall envelope with upward slope |
| Curved elements | Arcs assigned to nearest enclosing wall polygon |

#### 4.4.6 Output

- `topology` dict: `building_footprint` (Shapely Polygon), `rooms`, `walls`, `roof_profile`, `curved_elements`
- World-coordinate vertex array in metres
- `view_type`: `'plan'`, `'elevation_front'`, `'elevation_side'`, or `'combined'`

---

### 4.5 Stage 5 — CadQuery 3D Model Generation

#### 4.5.1 Purpose

Assemble the topology description into a fully parametric CadQuery solid model. Every geometric decision is driven by the extracted data. No default dimensions are hardcoded except where the sketch provides insufficient information, in which case architectural standards are applied.

#### 4.5.2 Default Architectural Parameters

| Parameter | Default | Source of Override |
|---|---|---|
| Storey height | 3.0 m | Detected floor-to-ceiling lines in elevation |
| Wall thickness (exterior) | 0.25 m | Detected double-line walls in plan |
| Wall thickness (interior) | 0.15 m | Detected double-line walls in plan |
| Door height | 2.1 m | Gap height in elevation lines |
| Window sill height | 0.9 m | Gap centroid height in elevation |
| Roof pitch | 30° | Detected roof angle in elevation |
| Eave overhang | 0.5 m | Roof line extension beyond wall line |

#### 4.5.3 Construction Sequence

1. Create the base workplane on XY (ground plane).
2. Extrude the building footprint polygon to wall height using `polyline().close().extrude()`. Curved walls use `spline()` or `ellipseArc()` before closing the wire.
3. Shell the extruded solid to wall thickness using `.shell(-wall_thickness)`.
4. Cut door and window openings: each opening is a box solid subtracted from the wall solid using `.cut()`.
5. Construct the roof: gable → triangular prism; hip → pyramid from four roof planes; curved → loft between eave wire and spline ridge profile.
6. Add detected structural details: columns as cylinder extrusions, arched openings as `ellipseArc` cuts, bay windows as extruded sub-polygons.
7. Union all component solids.
8. Export: STEP (universal compatibility), STL (rendering), BREP (CadQuery re-import).

```python
import cadquery as cq

def build_model(topology, wall_height=3.0, wall_thickness=0.25):
    footprint_pts = topology['building_footprint_coords']

    building = (
        cq.Workplane("XY")
        .polyline(footprint_pts)
        .close()
        .extrude(wall_height)
    )

    for opening in topology['openings']:
        void = (
            cq.Workplane("XY")
            .box(opening['width'], wall_thickness * 2, opening['height'])
            .translate((opening['x'], opening['y'], opening['sill_z']))
        )
        building = building.cut(void)

    for curve in topology['curved_elements']:
        if curve['type'] == 'bspline':
            ctrl_pts = [cq.Vector(p[0], p[1], 0) for p in curve['control_points']]
            curved_wall = cq.Workplane("XY").spline(ctrl_pts).extrude(wall_height)
            building = building.union(curved_wall)

    return building
```

#### 4.5.4 Error Handling and Fallbacks

- If `polyline().close()` fails due to non-planar vertices, flatten to median Z-plane before retry.
- If `shell()` produces a self-intersecting solid (thin walls), fall back to individual extruded wall rectangles.
- If a curve fails spline fitting with fewer than 4 control points, replace with a polyline approximation.
- If the construction raises an unrecoverable OCC exception, log the failure, write a diagnostic JSON, and surface a user-facing error.

#### 4.5.5 Output

- `building.step` — full solid in ISO 10303-21 format
- `building.stl` — triangulated mesh for rendering
- `building.brep` — CadQuery-native format
- `metadata.json` — all extracted parameters, defaults applied, and override flags

---

### 4.6 Stage 6 — Camera-Aligned Snapshot

#### 4.6.1 Purpose

Render a snapshot of the 3D model from precisely the same perspective angle as the original sketch, derived mathematically from the vanishing points extracted in Stage 2.

#### 4.6.2 Vanishing Point to Camera Transform

The two dominant vanishing points VP1 and VP2 define the horizontal plane of the drawing's perspective. Camera parameters are derived as follows:

1. **Principal point:** estimated as the image centre.
2. **Focal length:** `f = d(VP1, centre) × d(VP2, centre) / d(VP1, VP2)` using the two-point perspective relationship.
3. **Camera azimuth:** computed from the angle of VP1 relative to the model's X-axis.
4. **Camera elevation:** derived from the horizon line height relative to the building's vertical extent.
5. **Camera distance:** set so the building footprint fills approximately 70% of the render frame.

#### 4.6.3 Rendering

The CadQuery STEP file is imported into a headless Blender instance (`bpy`, run via subprocess) with the camera positioned using the computed transform. A neutral three-point lighting setup (key, fill, rim) produces a clean architectural render saved as PNG at **1200×900px**.

PythonOCC's built-in offscreen renderer is implemented as a fallback with zero external dependency, at the cost of lower render quality.

#### 4.6.4 Linework Extraction (Replacing the Manual GPT-4V Step)

The render is post-processed to produce a ControlNet-compatible CAD-style line drawing:

```python
def extract_lineart(render_path):
    img = cv2.imread(render_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 80, 160)
    thinned = cv2.ximgproc.thinning(edges)   # single-pixel-wide lines
    lineart = cv2.bitwise_not(thinned)        # white lines on black
    return lineart
```

#### 4.6.5 Output

- `render.png` — perspective render, 1200×900px
- `lineart.png` — white lines on black background, ControlNet Lineart-ready
- `depth.png` — 16-bit greyscale depth map for dual ControlNet conditioning

---

## 5. Data Flow & Module Interfaces

Each stage is a Python class with a standardised interface:

```python
class PipelineStage:
    def process(self, input: StageInput) -> StageOutput: ...
    def validate(self, output: StageOutput) -> ValidationReport: ...
    def diagnostics(self, output: StageOutput) -> DiagnosticsBundle: ...
```

All intermediate results are serialisable to JSON or numpy format for inspection and replay.

| Stage | Input | Output | Key Library |
|---|---|---|---|
| 1. Preprocessing | Raw sketch image | Binary edge image + metadata | `opencv-python` |
| 2. Line Extraction | Edge image | Lines, vertices, vanishing pts | `opencv`, `scikit-learn` |
| 3. Curve Extraction | Edge image + line mask | Fitted curves (arc / spline) | `scipy`, `scikit-learn` |
| 4. Topology Recon. | Lines + curves | Floor plan + elevation topology | `shapely` |
| 5. CadQuery Build | Topology dict | STEP + STL + metadata JSON | `cadquery` |
| 6. Snapshot Render | STEP + vanishing pts | Lineart + depth map PNGs | `blender (bpy)` / OCC |

---

## 6. Functional Requirements

### 6.1 Image Input

- **FR-01:** The system SHALL accept sketch images in JPEG, PNG, and TIFF formats.
- **FR-02:** The system SHALL accept images with a minimum resolution of 800px on the short edge.
- **FR-03:** The system SHALL apply automatic perspective correction to images taken at an angle from the sketch paper.
- **FR-04:** The system SHALL accept an optional real-world scale reference as a floating-point value in metres.
- **FR-05:** The system SHALL handle pencil, ink, and charcoal sketch types without parameter changes.

### 6.2 Geometric Extraction

- **FR-06:** The system SHALL detect all straight line segments longer than 40px in the canonical 1200px-wide image.
- **FR-07:** The system SHALL cluster near-duplicate line detections into single canonical lines before further processing.
- **FR-08:** The system SHALL apply RANSAC fitting to all detected lines to produce statistically optimal line parameters.
- **FR-09:** The system SHALL snap fitted line angles to architectural canonical angles (0°, 30°, 45°, 60°, 90°) within a 5° tolerance.
- **FR-10:** The system SHALL compute all pairwise intersections of non-parallel canonical lines and produce a deduplicated vertex set.
- **FR-11:** The system SHALL classify residual curve pixels into arc, ellipse, or B-spline categories.
- **FR-12:** The system SHALL fit arcs and ellipses using algebraic fitting with a residual threshold of 3px.
- **FR-13:** The system SHALL fit B-splines using `scipy splprep` with adaptive smoothing derived from sketch line quality.
- **FR-14:** The system SHALL detect compound curve breakpoints and fit each sub-segment independently.

### 6.3 Topology

- **FR-15:** The system SHALL identify all enclosed polygons in the line network using Shapely `polygonize`.
- **FR-16:** The system SHALL classify the largest enclosed polygon as the building exterior footprint.
- **FR-17:** The system SHALL detect the sketch view type (plan, elevation, or combined) using line direction statistics.
- **FR-18:** The system SHALL classify individual wall segments as exterior or interior based on polygon membership.
- **FR-19:** The system SHALL detect door and window openings as gaps in wall segments within defined dimensional ranges.

### 6.4 3D Model

- **FR-20:** The system SHALL generate a valid CadQuery solid from any successfully reconstructed topology.
- **FR-21:** The system SHALL produce the building model as a watertight solid exportable to STEP format.
- **FR-22:** The system SHALL construct curved walls using CadQuery `spline` or `ellipseArc` primitives, not polygonal approximations, when curve data is available.
- **FR-23:** The system SHALL cut door and window openings from wall solids at the correct position and dimensions.
- **FR-24:** The system SHALL construct the roof geometry from the detected roof profile, defaulting to a 30° gable if no roof lines are detected.
- **FR-25:** The system SHALL export the final model in STEP, STL, and BREP formats.

### 6.5 Rendering

- **FR-26:** The system SHALL compute camera azimuth, elevation, and focal length from the sketch's detected vanishing points.
- **FR-27:** The system SHALL render the 3D model from the computed camera position at 1200×900px minimum resolution.
- **FR-28:** The system SHALL produce a ControlNet-compatible lineart image from the render via Canny edge detection.
- **FR-29:** The system SHALL produce a 16-bit depth map from the render for dual ControlNet conditioning.

---

## 7. Non-Functional Requirements

| Category | Requirement | Acceptance Criterion |
|---|---|---|
| Performance | End-to-end processing ≤ 60s | Measured on CPU: Intel i7-12th gen or equivalent |
| Reliability | ≥ 80% sketch success rate on test set | 20-sketch diverse test set, no cherry-picking |
| Accuracy | Wall position deviation ≤ 5% | IoU of projected footprint vs hand-measured sketch |
| Determinism | Identical output for identical input | No random seeds; full reproducibility required |
| Auditability | All intermediate outputs logged | JSON diagnostics bundle saved per run |
| Portability | Runs on Linux, macOS, Windows | Docker container provided; tested on all three |
| Testability | Unit tests for each stage | `pytest` coverage ≥ 70% for each stage module |
| Editability | Output STEP files open in FreeCAD | Manual verification on 5 representative outputs |

---

## 8. Technology Stack

| Library / Tool | Version | Role in Pipeline |
|---|---|---|
| Python | 3.11+ | Primary implementation language |
| `opencv-python` | 4.9+ | Preprocessing, Canny, HoughLinesP, fitEllipse |
| `numpy` | 1.26+ | Array operations, intersection geometry, angle snapping |
| `scipy` | 1.12+ | splprep/splev B-spline fitting, least_squares circle fit, KDTree |
| `scikit-learn` | 1.4+ | RANSACRegressor line fitting, DBSCAN clustering |
| `shapely` | 2.0+ | Polygonize floor plan, topology reconstruction, geometry ops |
| `cadquery` | 2.4+ | 3D solid construction, STEP/STL/BREP export |
| `cadquery-ocp` | 7.7+ | OpenCASCADE kernel underlying CadQuery |
| `blender (bpy)` | 4.0+ | Headless rendering, depth pass, camera alignment |
| `pytest` | 8.0+ | Unit and integration testing |
| `loguru` | 0.7+ | Structured logging and diagnostics |

---

## 9. Test Plan

### 9.1 Unit Tests (Per Stage)

| Stage | Required Unit Tests |
|---|---|
| 1. Preprocessing | Greyscale conversion, perspective warp accuracy on synthetic grid, Canny auto-threshold, resize to canonical width |
| 2. Line Extraction | HoughLinesP on synthetic line images with known ground truth, RANSAC outlier rejection, DBSCAN cluster merging of parallel lines, angle snapping to all 8 canonical angles, intersection computation on non-parallel pairs |
| 3. Curve Extraction | Residual pixel extraction leaves only curve pixels, DBSCAN stroke isolation on multi-curve images, ellipse fit accuracy on synthetic ellipse images, B-spline fit error within 3px on generated spline images, compound curve breakpoint detection |
| 4. Topology | Polygonize finds correct polygon count on synthetic floor plans, largest polygon identified as footprint, door/window gap detection at boundary dimensions, view type classification on plan vs elevation test images |
| 5. CadQuery | Simple box building generates watertight STEP, curved wall generates valid spline wire, door cut produces correct void, gable roof loft does not self-intersect, STEP/STL/BREP export files are non-empty and parseable |
| 6. Rendering | Camera transform from two known vanishing points, render output is 1200×900px PNG, lineart image has white lines on black background, depth map is 16-bit greyscale |

### 9.2 Integration Tests

- End-to-end pipeline test on **5 synthetic sketches** with known ground-truth geometry (computer-generated line drawings of simple buildings with measured dimensions).
- End-to-end pipeline test on **5 hand-drawn test sketches** of varying roughness, with manual geometric measurement of the output STEP model.
- Regression test: re-run all 10 test sketches after any stage modification to detect accuracy regressions.

### 9.3 Evaluation Test Set

The PoC acceptance evaluation uses a test set of **20 hand-drawn architectural sketches**:

- 5 simple rectilinear buildings with no curves (baseline)
- 5 buildings with arched openings or curved bay windows
- 5 buildings with complex roof profiles (hip, mansard, shed)
- 5 buildings combining floor plan and elevation views on one sheet

Sketches are sourced from **three different human draughtspeople** with varying levels of drawing precision, to ensure the pipeline is not tuned to a single drawing style.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HoughLinesP misses lines in rough or light-pressure sketches | Medium | High | Adaptive Canny thresholds via Otsu; bilateral filter pre-processing; user re-scan guidance |
| Polygonize fails on unclosed line networks (sketch gaps) | Medium | High | Morphological gap-closing in preprocessing; fallback to convex hull of vertex set |
| CadQuery OCC kernel exception on complex geometry | Medium | Medium | Per-operation try/except with fallback to simpler geometry; wall-by-wall construction mode |
| Vanishing point detection fails on flat-on drawings (no perspective) | Low | Medium | Detect zero-perspective case and use orthographic camera projection instead |
| B-spline smoothing over-smooths fine architectural detail | Medium | Low | Adaptive smoothing from sketch roughness metric; user smoothing slider in final product |
| Blender headless rendering unavailable in target environment | Low | Medium | PythonOCC offscreen renderer as primary fallback; pre-tested Docker image provided |
| Processing time exceeds 60s on large/complex sketches | Medium | Medium | Profile hotspots; vectorise intersection computation; multiprocessing for DBSCAN |

---

## 11. Development Milestones

| Week | Milestone | Deliverable | Exit Criterion |
|---|---|---|---|
| 1–2 | Stage 1 & 2: Preprocessing and line extraction | `preprocessing.py` + `line_extractor.py` + unit tests | 90% line detection recall on synthetic test set |
| 3–4 | Stage 3: Curve extraction and best-fit | `curve_extractor.py` + unit tests | Arc fit error ≤ 3px; spline fit on 5 freehand curves |
| 5 | Stage 4: Topology reconstruction | `topology.py` + unit tests | Correct polygon detection on 3 synthetic floor plans |
| 6–7 | Stage 5: CadQuery model generation | `model_builder.py` + unit tests | Watertight STEP from 3 topology inputs |
| 8 | Stage 6: Camera-aligned snapshot | `renderer.py` + unit tests | Vanishing point alignment ≥ 85% on 3 sketches |
| 9 | Integration & pipeline assembly | `pipeline.py` + integration tests | End-to-end run on 5 synthetic sketches |
| 10 | Evaluation on 20-sketch test set | Evaluation report + diagnostics | All PoC success criteria met or documented gaps |

---

## 12. Out of Scope for PoC

- **Photorealistic image generation** (ControlNet/Stable Diffusion stage) — the PoC ends at the lineart and depth map outputs. Integration with the downstream generative pipeline is a separate workstream.
- **User interface** — the PoC is a command-line Python pipeline. A web or desktop UI is a post-PoC deliverable.
- **Multi-storey building support** — the PoC targets single-storey structures. Multi-storey extension requires cross-elevation topology logic not in scope.
- **Interior rendering** — the PoC models building exteriors only.
- **Real-time or streaming processing** — batch processing only in the PoC.
- **LLM-assisted geometry inference** — the PoC intentionally excludes all neural network geometry inference to validate the pure classical CV approach first.

---

## 13. Open Questions

| ID | Question |
|---|---|
| OQ-01 | Should the system attempt to detect sketch scale from standard architectural symbols (e.g. a drawn human figure or a car) when no explicit reference is provided? |
| OQ-02 | For combined plan + elevation sketches, should the system require both views to produce the 3D model, or attempt single-view reconstruction when only one is present? |
| OQ-03 | Is PythonOCC offscreen rendering of sufficient quality for the ControlNet stage, or is the Blender dependency mandatory for downstream photorealistic accuracy? |
| OQ-04 | What is the acceptable false positive rate for door/window opening detection — should questionable openings be flagged for human review rather than silently defaulted? |
| OQ-05 | Should the pipeline expose a parameter-editing step between Stage 4 and Stage 5 — a lightweight UI showing extracted dimensions before CadQuery runs — to allow user correction? |

---

## 14. Appendix — Algorithm Quick Reference

### A. Line Extraction Summary

| Algorithm | Library / Function | Purpose |
|---|---|---|
| Gaussian Blur | `cv2.GaussianBlur` | Noise reduction before edge detection |
| Canny Edge Detection | `cv2.Canny` (auto-threshold via Otsu) | Produce binary edge image |
| Probabilistic Hough | `cv2.HoughLinesP` | Detect line segments from edge image |
| RANSAC Line Fitting | `sklearn.linear_model.RANSACRegressor` | Statistically optimal line through noisy pixels |
| DBSCAN Clustering | `sklearn.cluster.DBSCAN` | Merge duplicate / parallel line detections |
| Angle Snapping | `numpy` (custom) | Quantise angles to architectural grid |
| Line Intersection | `numpy` (cross-product formula) | Compute candidate vertices |
| Vanishing Points | DBSCAN on intersection cluster | Extract perspective metadata |

### B. Curve Extraction Summary

| Algorithm | Library / Function | Purpose |
|---|---|---|
| Residual Extraction | `cv2.subtract` on edge vs line mask | Isolate curve pixels from straight lines |
| Stroke Clustering | `sklearn.cluster.DBSCAN` (eps=8px) | Group curve pixels into individual strokes |
| Point Ordering | `scipy.spatial.KDTree` (greedy NN) | Order stroke pixels sequentially |
| Curve Classification | `scipy.spatial.ConvexHull` + custom metrics | Classify as arc, ellipse, or spline |
| Curvature Breakpoints | `numpy` cross-product (sliding window=5) | Detect compound curve segment boundaries |
| Arc / Ellipse Fitting | `cv2.fitEllipse` (Bookstein method) | Algebraic ellipse / arc fitting |
| Circle Fitting | `scipy.optimize.least_squares` (Taubin) | Robust circle fit as fallback |
| B-Spline Fitting | `scipy.interpolate.splprep` / `splev` | Parametric spline with adaptive smoothing |

### C. CadQuery Primitive Mapping

| Extracted Element | CadQuery Primitive | Construction Method |
|---|---|---|
| Straight wall segment | Line + extrude | `Workplane.polyline().close().extrude()` |
| Curved wall (B-spline) | Spline wire + extrude | `Workplane.spline(control_pts).extrude()` |
| Curved wall (arc) | Arc wire + extrude | `Workplane.ellipseArc(rx, ry, a1, a2).extrude()` |
| Circular room / tower | Circle + extrude | `Workplane.circle(r).extrude()` |
| Door opening | Box solid — subtract | `Workplane.box().translate().cut(wall)` |
| Window opening | Box solid — subtract | `Workplane.box().translate(sill_h).cut(wall)` |
| Gable roof | Triangular prism | `Workplane.polygon(3_pts).extrude(ridge_h)` |
| Hip roof | Lofted solid | `Workplane.shell → loft([eave_wire, ridge_pt])` |
| Curved roof | Spline loft | `Workplane.loft([eave_wire, ridge_spline])` |
| Column / pillar | Cylinder | `Workplane.circle(r).extrude(h)` |

---

*End of Document — Sketch-to-3D Architectural Model PoC PRD v1.0*  
*© 2026 — Confidential — Internal Use Only*
