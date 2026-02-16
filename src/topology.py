from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Polygon
from loguru import logger
import numpy as np

class TopologyReconstructor:
    def __init__(self, pixels_per_metre=100):
        self.pixels_per_metre = pixels_per_metre

    def reconstruct(self, lines, vertices, curves, img_shape):
        """
        Full topology reconstruction for Stage 4.
        """
        logger.info("Reconstructing topology from lines and curves.")
        
        # 1. Snap line endpoints to nearest vertices to close gaps
        # Also extend lines slightly to ensure they overlap/intersect
        snapped_lines = self.snap_endpoints_to_vertices(lines, vertices, threshold=30)
        extended_lines = self.extend_lines(snapped_lines, extension=5)

        # 2. Floor Plan Polygon Detection
        shapely_lines = [LineString([(l[0], l[1]), (l[2], l[3])]) for l in extended_lines]
        merged = unary_union(shapely_lines)
        polygons = list(polygonize(merged))
        
        if not polygons:
            logger.warning("No polygons detected by polygonize. Falling back to convex hull of vertices.")
            # Fallback: create a polygon from the convex hull of all line endpoints
            all_pts = []
            for l in extended_lines:
                all_pts.extend([(l[0], l[1]), (l[2], l[3])])
            if len(all_pts) < 3:
                return {}
            footprint = Polygon(all_pts).convex_hull
            interior_rooms = []
        else:
            # 3. Identify Footprint (Largest Polygon)
            footprint = max(polygons, key=lambda p: p.area)
            interior_rooms = [p for p in polygons if p != footprint]

        # 4. Coordinate Normalisation
        minx, miny, maxx, maxy = footprint.bounds
        origin = (minx, maxy) 

        normalized_footprint = self.normalize_polygon(footprint, origin)
        normalized_rooms = [self.normalize_polygon(p, origin) for p in interior_rooms]

        topology = {
            'footprint': normalized_footprint,
            'rooms': normalized_rooms,
            'curves': curves,
            'origin_px': origin,
            'scale': self.pixels_per_metre
        }

        logger.info(f"Detected footprint with area {normalized_footprint.area:.2f} m2.")
        return topology

    def extend_lines(self, lines, extension=5):
        """
        Extend line segments slightly at both ends.
        """
        extended = []
        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length == 0: continue
            
            ux = dx / length
            uy = dy / length
            
            nx1 = x1 - ux * extension
            ny1 = y1 - uy * extension
            nx2 = x2 + ux * extension
            ny2 = y2 + uy * extension
            extended.append((nx1, ny1, nx2, ny2))
        return extended

    def snap_endpoints_to_vertices(self, lines, vertices, threshold=20):
        """
        Adjust line endpoints to match nearby vertices.
        """
        if not vertices:
            return lines
            
        v_array = np.array(vertices)
        new_lines = []
        
        for x1, y1, x2, y2 in lines:
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            
            # Find nearest vertex for p1
            dists1 = np.linalg.norm(v_array - p1, axis=1)
            if np.min(dists1) < threshold:
                p1 = v_array[np.argmin(dists1)]
                
            # Find nearest vertex for p2
            dists2 = np.linalg.norm(v_array - p2, axis=1)
            if np.min(dists2) < threshold:
                p2 = v_array[np.argmin(dists2)]
                
            new_lines.append((p1[0], p1[1], p2[0], p2[1]))
            
        return new_lines

    def normalize_polygon(self, poly, origin):
        """
        Convert image coordinates to world coordinates (metres).
        CadQuery uses right-hand system: x right, y up.
        """
        ox, oy = origin
        coords = []
        for x, y in poly.exterior.coords:
            nx = (x - ox) / self.pixels_per_metre
            ny = (oy - y) / self.pixels_per_metre # Flip Y
            coords.append((nx, ny))
        
        return Polygon(coords)
