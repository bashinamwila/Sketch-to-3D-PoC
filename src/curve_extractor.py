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
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 5) # Dilated line mask
        
        residual = cv2.subtract(edge_image, line_mask)
        return residual

    def order_points(self, points):
        """
        Order points sequentially using greedy nearest neighbour.
        """
        if len(points) == 0: return points
        
        # Start from point with minimum y (or x)
        start_idx = np.argmin(points[:, 0])
        ordered = [points[start_idx]]
        remaining = list(range(len(points)))
        remaining.remove(start_idx)
        
        tree = KDTree(points)
        
        while remaining:
            last = ordered[-1]
            dist, idxs = tree.query(last, k=min(10, len(points)))
            found = False
            for idx in idxs:
                if idx in remaining:
                    ordered.append(points[idx])
                    remaining.remove(idx)
                    found = True
                    break
            if not found:
                # Jump to nearest remaining point if gap encountered
                dist, idx = tree.query(last, k=len(points))
                for i in idx:
                    if i in remaining:
                        ordered.append(points[i])
                        remaining.remove(i)
                        found = True
                        break
                if not found: break
        
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
