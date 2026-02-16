import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from loguru import logger

class LineExtractor:
    def __init__(self, rho=1, theta=np.pi/180, threshold=50, min_line_length=50, max_line_gap=10):
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def extract_lines(self, edge_image):
        """
        Full line extraction pipeline for Stage 2.
        """
        logger.info("Extracting lines from edge image.")
        # 1. Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edge_image, self.rho, self.theta, self.threshold, 
                                minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        
        if lines is None:
            logger.warning("No lines detected.")
            return [], [], []

        # Convert to list of (x1, y1, x2, y2)
        raw_lines = [l[0] for l in lines]

        # 2. RANSAC Refinement & 3. Clustering
        # For simplicity in this PoC, we'll cluster first then refine or vice versa.
        # Let's do a basic clustering based on angle and distance.
        clustered_lines = self.cluster_lines(raw_lines)

        # 4. Architectural Angle Snapping
        snapped_lines = [self.snap_line(l) for l in clustered_lines]

        # 5. Vanishing Point Detection & 6. Intersections
        # (Simplified versions for now)
        vertices = self.compute_intersections(snapped_lines, edge_image.shape)

        # 6. Directional Classification for Perspective
        directions = self.classify_directions(snapped_lines)
        
        # 7. Vanishing Point Estimation from Categories
        directions['vp_left'] = self.estimate_vp(directions['left_vp'])
        directions['vp_right'] = self.estimate_vp(directions['right_vp'])

        # 8. INTELLIGENCE: Filter Diagonals
        # If a diagonal line points toward a VP, it's a perspective line, not a roof line
        directions['diagonal'] = self.filter_roof_lines(directions['diagonal'], directions)

        logger.info(f"Extracted {len(snapped_lines)} lines. VPs: L={directions['vp_left']}, R={directions['vp_right']}")
        return snapped_lines, vertices, directions

    def filter_roof_lines(self, diagonals, directions):
        """
        Ensure diagonals are truly roof slopes and not perspective lines.
        """
        roof_lines = []
        vps = [v for v in [directions.get('vp_left'), directions.get('vp_right')] if v]
        
        for l in diagonals:
            is_perspective = False
            for vp in vps:
                # Check if the line points toward the VP
                # Line: (x1, y1) to (x2, y2). Vector v = (x2-x1, y2-y1)
                # Vector to VP: v_vp = (vp_x - x1, vp_y - y1)
                v = np.array([l[2]-l[0], l[3]-l[1]])
                v_vp = np.array([vp[0]-l[0], vp[1]-l[1]])
                
                norm_v = v / (np.linalg.norm(v) + 1e-6)
                norm_v_vp = v_vp / (np.linalg.norm(v_vp) + 1e-6)
                
                # If vectors are highly aligned (dot product near 1 or -1)
                if abs(np.dot(norm_v, norm_v_vp)) > 0.95:
                    is_perspective = True
                    break
            
            if not is_perspective:
                roof_lines.append(l)
        return roof_lines

    def estimate_vp(self, lines):
        """
        Estimate a vanishing point from a set of lines.
        """
        if len(lines) < 2: return None
        intersections = []
        # Sample some intersections to find a centroid
        for i in range(min(10, len(lines))):
            for j in range(i + 1, min(10, len(lines))):
                pt = self.line_intersection(lines[i], lines[j])
                if pt: intersections.append(pt)
        
        if not intersections: return None
        return np.mean(intersections, axis=0).astype(int).tolist()

    def classify_directions(self, lines):
        """
        Group lines into Vertical, Left-VP, Right-VP, and Diagonal (Roof) buckets based on slope.
        """
        categories = {'vertical': [], 'left_vp': [], 'right_vp': [], 'diagonal': []}
        for l in lines:
            x1, y1, x2, y2 = l
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
            
            # Broaden vertical threshold for hand sketches (70 to 90 degrees)
            if 70 < angle <= 90:
                categories['vertical'].append(l)
            elif angle < 20: # Near horizontal perspective lines
                slope = dy / (dx + 1e-6)
                if slope > 0:
                    categories['right_vp'].append(l)
                else:
                    categories['left_vp'].append(l)
            else:
                categories['diagonal'].append(l)
        return categories

    def cluster_lines(self, lines):
        """
        Clustering lines using DBSCAN on (angle, perpendicular_distance).
        """
        if not lines:
            return []

        features = []
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1) % np.pi
            # Distance from origin to line
            dist = abs((y2 - y1) * x1 - (x2 - x1) * y1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            features.append([angle, dist / 1000.0]) # normalise distance

        db = DBSCAN(eps=0.05, min_samples=1).fit(features)
        labels = db.labels_

        new_lines = []
        for label in set(labels):
            if label == -1: continue
            indices = np.where(labels == label)[0]
            cluster = [lines[i] for i in indices]
            # Average the lines (simplified)
            avg_line = np.mean(cluster, axis=0).astype(int)
            new_lines.append(tuple(avg_line))
        
        return new_lines

    def snap_line(self, line):
        """
        Snap line angle to nearest architectural angle.
        """
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        
        architectural_angles = [0, 30, 45, 60, 90, 120, 135, 150, 180]
        nearest = min(architectural_angles, key=lambda a: abs(angle - a))
        
        if abs(angle - nearest) < 5:
            # Rotate line around its center to the snapped angle
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rad = np.radians(nearest)
            nx1 = int(cx - (length / 2) * np.cos(rad))
            ny1 = int(cy - (length / 2) * np.sin(rad))
            nx2 = int(cx + (length / 2) * np.cos(rad))
            ny2 = int(cy + (length / 2) * np.sin(rad))
            return (nx1, ny1, nx2, ny2)
        
        return line

    def compute_intersections(self, lines, img_shape):
        """
        Compute all pairwise intersections within image bounds.
        """
        vertices = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self.line_intersection(lines[i], lines[j])
                if pt:
                    x, y = pt
                    if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                        vertices.append((int(x), int(y)))
        
        # Deduplicate vertices
        if not vertices:
            return []
        
        db = DBSCAN(eps=10, min_samples=1).fit(vertices)
        unique_vertices = []
        for label in set(db.labels_):
            indices = np.where(db.labels_ == label)[0]
            avg_v = np.mean([vertices[i] for i in indices], axis=0).astype(int)
            unique_vertices.append(tuple(avg_v))
            
        return unique_vertices

    def line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        return (x, y)
