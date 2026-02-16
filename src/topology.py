from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Polygon
from loguru import logger
import numpy as np

class TopologyReconstructor:
    def __init__(self, pixels_per_metre=100):
        self.pixels_per_metre = pixels_per_metre

    def reconstruct(self, lines, vertices, curves, img_shape, directions=None):
        """
        Full topology reconstruction for Stage 4.
        """
        logger.info("Reconstructing topology from lines and curves.")
        
        # Check if we should use Perspective Mode
        if directions and len(directions.get('vertical', [])) > 2:
            return self.reconstruct_perspective(directions)

        # Fallback to Plan Mode
        return self.reconstruct_plan(lines, vertices, curves)

    def reconstruct_perspective(self, directions):
        """
        Estimate 3D box dimensions from perspective lines.
        """
        # 1. Height from verticals
        heights = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in directions['vertical']]
        avg_h_px = np.mean(heights) if heights else 300
        
        # 2. Width/Depth from perspective lines
        widths = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in directions['left_vp']]
        depths = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in directions['right_vp']]
        
        w_m = (np.max(widths) if widths else 500) / self.pixels_per_metre
        d_m = (np.max(depths) if depths else 500) / self.pixels_per_metre
        h_m = avg_h_px / self.pixels_per_metre

        # Create a simple box footprint for the builder
        footprint = Polygon([(0,0), (w_m, 0), (w_m, d_m), (0, d_m)])
        
        topology = {
            'view_type': 'perspective',
            'footprint': footprint,
            'height': h_m,
            'width': w_m,
            'depth': d_m,
            'scale': self.pixels_per_metre
        }
        logger.info(f"Perspective Mode: Estimated {w_m:.1f}m x {d_m:.1f}m x {h_m:.1f}m box.")
        return topology

    def reconstruct_plan(self, lines, vertices, curves):
        # 1. Snap line endpoints to nearest vertices to close gaps
        snapped_lines = self.snap_endpoints_to_vertices(lines, vertices, threshold=30)
        extended_lines = self.extend_lines(snapped_lines, extension=5)

        # 2. Floor Plan Polygon Detection
        shapely_lines = [LineString([(l[0], l[1]), (l[2], l[3])]) for l in extended_lines]
        merged = unary_union(shapely_lines)
        polygons = list(polygonize(merged))
        
        if not polygons:
            logger.warning("No polygons detected by polygonize. Falling back to convex hull of vertices.")
            all_pts = []
            for l in extended_lines:
                all_pts.extend([(l[0], l[1]), (l[2], l[3])])
            if len(all_pts) < 3:
                return {}
            footprint = Polygon(all_pts).convex_hull
            interior_rooms = []
        else:
            footprint = max(polygons, key=lambda p: p.area)
            interior_rooms = [p for p in polygons if p != footprint]

        minx, miny, maxx, maxy = footprint.bounds
        origin = (minx, maxy) 

        normalized_footprint = self.normalize_polygon(footprint, origin)
        normalized_rooms = [self.normalize_polygon(p, origin) for p in interior_rooms]

        return {
            'view_type': 'plan',
            'footprint': normalized_footprint,
            'rooms': normalized_rooms,
            'curves': curves,
            'origin_px': origin,
            'scale': self.pixels_per_metre
        }

    def snap_endpoints_to_vertices(self, lines, vertices, threshold=20):
        if not vertices: return lines
        v_array = np.array(vertices)
        new_lines = []
        for x1, y1, x2, y2 in lines:
            p1, p2 = np.array([x1, y1]), np.array([x2, y2])
            d1 = np.linalg.norm(v_array - p1, axis=1)
            if np.min(d1) < threshold: p1 = v_array[np.argmin(d1)]
            d2 = np.linalg.norm(v_array - p2, axis=1)
            if np.min(d2) < threshold: p2 = v_array[np.argmin(d2)]
            new_lines.append((p1[0], p1[1], p2[0], p2[1]))
        return new_lines

    def extend_lines(self, lines, extension=5):
        extended = []
        for x1, y1, x2, y2 in lines:
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length == 0: continue
            ux, uy = dx / length, dy / length
            extended.append((x1 - ux * extension, y1 - uy * extension, x2 + ux * extension, y2 + uy * extension))
        return extended

    def normalize_polygon(self, poly, origin):
        ox, oy = origin
        coords = [( (x - ox) / self.pixels_per_metre, (oy - y) / self.pixels_per_metre ) for x, y in poly.exterior.coords]
        return Polygon(coords)
