import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree, ConvexHull
from scipy.interpolate import splprep, splev
from scipy.optimize import least_squares
from loguru import logger

class CurveExtractor:
    def __init__(self, eps=8, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples

    def extract_curves(self, edge_image, lines):
        """
        Full curve extraction pipeline for Stage 3.
        """
        logger.info("Extracting curves from residual edge image.")
        # 1. Residual Curve Pixel Extraction
        residual = self.get_residual_pixels(edge_image, lines)
        
        # 2. Stroke Clustering
        coords = np.column_stack(np.where(residual > 0))
        if len(coords) < self.min_samples:
            return []

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coords)
        labels = db.labels_

        curves = []
        for label in set(labels):
            if label == -1: continue
            stroke_pts = coords[labels == label]
            
            # 3. Point Ordering
            ordered_pts = self.order_points(stroke_pts)
            
            # 4. Classification & 5/6. Fitting
            curve_data = self.fit_curve(ordered_pts)
            if curve_data:
                curves.append(curve_data)

        logger.info(f"Extracted {len(curves)} curves.")
        return curves

    def get_residual_pixels(self, edge_image, lines):
        """
        Subtract detected lines from edge image to get residual pixels.
        """
        line_mask = np.zeros_like(edge_image)
        for x1, y1, x2, y2 in lines:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 7) # Increased dilation
        
        residual = cv2.subtract(edge_image, line_mask)
        return residual

    def order_points(self, points):
        """
        Order points sequentially using greedy nearest neighbour.
        Optimized version using KDTree more efficiently.
        """
        if len(points) <= 1: return points
        
        # Start from point with minimum y (or x)
        start_idx = np.argmin(points[:, 0])
        ordered = []
        remaining_mask = np.ones(len(points), dtype=bool)
        
        current_idx = start_idx
        tree = KDTree(points)
        
        for _ in range(len(points)):
            ordered.append(points[current_idx])
            remaining_mask[current_idx] = False
            
            # Find nearest among remaining
            # Query more points to find one that is still in remaining_mask
            dists, idxs = tree.query(points[current_idx], k=min(20, len(points)))
            
            next_idx = -1
            for idx in idxs:
                if remaining_mask[idx]:
                    next_idx = idx
                    break
            
            if next_idx == -1:
                # Fallback: search all points if local neighborhood is empty
                # This only happens if there's a large gap
                remaining_indices = np.where(remaining_mask)[0]
                if len(remaining_indices) == 0:
                    break
                dists, idxs = tree.query(points[current_idx], k=len(points))
                for idx in idxs:
                    if remaining_mask[idx]:
                        next_idx = idx
                        break
            
            if next_idx == -1: break
            current_idx = next_idx
        
        return np.array(ordered)

    def fit_curve(self, points):
        """
        Classify and fit the stroke.
        """
        if len(points) < 5: return None
        
        # Simple classification: Compactness for circle/ellipse
        pts = points.astype(float)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            peri = hull.area
            compactness = (4 * np.pi * area) / (peri ** 2) if peri > 0 else 0
        except:
            compactness = 0

        if compactness > 0.7:
            return self.fit_ellipse(pts)
        else:
            return self.fit_bspline(pts)

    def fit_ellipse(self, pts):
        # cv2.fitEllipse expects (x, y)
        pts_xy = pts[:, ::-1].astype(np.float32)
        if len(pts_xy) >= 5:
            ellipse = cv2.fitEllipse(pts_xy)
            center, axes, angle = ellipse
            return {'type': 'ellipse', 'center': center, 'axes': axes, 'angle': angle}
        return None

    def fit_bspline(self, pts):
        # splprep expects list of arrays for each dimension
        x = pts[:, 1]
        y = pts[:, 0]
        try:
            tck, u = splprep([x, y], s=len(x)*2, k=min(3, len(x)-1))
            u_fine = np.linspace(0, 1, 100)
            x_smooth, y_smooth = splev(u_fine, tck)
            return {
                'type': 'bspline',
                'points': list(zip(x_smooth, y_smooth)),
                'tck': tck
            }
        except:
            return {'type': 'polyline', 'points': list(zip(x, y))}
