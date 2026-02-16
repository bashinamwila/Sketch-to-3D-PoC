from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Polygon
from loguru import logger
import numpy as np

class TopologyReconstructor:
    def __init__(self, pixels_per_metre=100):
        self.pixels_per_metre = pixels_per_metre

    def reconstruct(self, lines, curves, img_shape):
        """
        Full topology reconstruction for Stage 4.
        """
        logger.info("Reconstructing topology from lines and curves.")
        
        # 1. Floor Plan Polygon Detection
        shapely_lines = [LineString([(l[0], l[1]), (l[2], l[3])]) for l in lines]
        merged = unary_union(shapely_lines)
        polygons = list(polygonize(merged))
        
        if not polygons:
            logger.warning("No polygons detected in floor plan.")
            # Fallback: create a single polygon if lines form a rough enclosure?
            # For PoC, just return empty.
            return {}

        # 2. Identify Footprint (Largest Polygon)
        footprint = max(polygons, key=lambda p: p.area)
        interior_rooms = [p for p in polygons if p != footprint]

        # 3. Coordinate Normalisation (Convert to Metres)
        # We'll use the bottom-left of the footprint as origin (0,0)
        minx, miny, maxx, maxy = footprint.bounds
        origin = (minx, maxy) # In image coords, y increases downwards

        normalized_footprint = self.normalize_polygon(footprint, origin)
        normalized_rooms = [self.normalize_polygon(p, origin) for p in interior_rooms]

        topology = {
            'footprint': normalized_footprint,
            'rooms': normalized_rooms,
            'curves': curves, # Curves need normalization too
            'origin_px': origin,
            'scale': self.pixels_per_metre
        }

        logger.info(f"Detected footprint with area {normalized_footprint.area:.2f} m2 and {len(normalized_rooms)} rooms.")
        return topology

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
