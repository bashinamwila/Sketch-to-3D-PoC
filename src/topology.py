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
            return self.reconstruct_perspective(directions, vertices, img_shape)

        # Fallback to Plan Mode
        return self.reconstruct_plan(lines, vertices, curves)

    def reconstruct_perspective(self, directions, vertices, img_shape):
        """
        Estimate 3D box dimensions from perspective lines using the 12m rule.
        """
        # 1. Height from verticals (pixels)
        heights = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in directions['vertical']]
        avg_h_px = np.mean(heights) if heights else 300
        
        # 2. Width/Depth from perspective lines (pixels)
        # Use total span of segments in each direction
        def get_span(lines):
            if not lines: return 0
            all_pts = []
            for l in lines: all_pts.extend([(l[0], l[1]), (l[2], l[3])])
            all_pts = np.array(all_pts)
            return np.max(np.linalg.norm(all_pts[:, None] - all_pts, axis=2))

        w_px = get_span(directions['left_vp'])
        d_px = get_span(directions['right_vp'])
        
        # 3. Apply 12m Rule (PRD 4.4.4)
        # Longest horizontal dimension = 12.0m
        max_dim_px = max(w_px, d_px)
        if max_dim_px == 0: max_dim_px = 500
        
        px_per_m = max_dim_px / 12.0
        
        w_m = w_px / px_per_m
        d_m = d_px / px_per_m
        h_m = avg_h_px / px_per_m

        # Cap height to realistic ratio if it's too tall, or boost if too short
        max_dim_m = max(w_m, d_m)
        
        # 4. Height Sanity Check
        # If height is < 2.5m for a building > 5m wide, likely under-estimated due to perspective
        if max_dim_m > 5.0 and h_m < 2.5:
            logger.warning(f"Calculated height {h_m:.1f}m too low. Boosting to standard 3.0m storey.")
            h_m = 3.0
        
        # 5. Complexity Check (Internal Verticals) -> L-Shape Generation
        verticals = directions['vertical']
        w_px_min = min(l[0] for l in verticals) if verticals else 0
        w_px_max = max(l[0] for l in verticals) if verticals else 1
        width_px = w_px_max - w_px_min
        
        # Filter for true internal verticals (not just noisy corners)
        internal_verticals = [l for l in verticals 
                              if (l[0] > w_px_min + width_px * 0.25) and 
                                 (l[0] < w_px_max - width_px * 0.25)]
        
        footprint = None
        sub_footprints = [] # Store rectangles for composite roof generation
        
        if len(internal_verticals) > 0:
            logger.warning(f"Detected {len(internal_verticals)} internal vertical lines. Generating L-shaped footprint.")
            
            # Density Check for Orientation (Left vs Right Wing)
            # Count lines in left half vs right half of the building bounds
            mid_x = w_px_min + width_px / 2
            left_count = sum(1 for l in verticals if l[0] < mid_x)
            right_count = sum(1 for l in verticals if l[0] > mid_x)
            
            # If right side has significantly more lines, wing is likely on Right
            wing_on_right = right_count > left_count
            
            # Calculate Split Position
            avg_split_px = np.mean([l[0] for l in internal_verticals])
            split_ratio = (avg_split_px - w_px_min) / width_px
            split_ratio = max(0.3, min(0.7, split_ratio))
            split_m = w_m * split_ratio

            # INTELLIGENCE LAYER: Optimize wing depth based on vertex support
            wing_depth_ratio = self.find_optimal_wing_depth(vertices, split_m, d_m, wing_on_right)
            wing_depth_m = d_m * wing_depth_ratio
            
            rect_main = None
            rect_wing = None

            if wing_on_right:
                logger.info(f"Orientation: Wing on RIGHT. Optimized Depth Ratio: {wing_depth_ratio:.2f}")
                rect_main = Polygon([(0, 0), (split_m, 0), (split_m, d_m), (0, d_m)])
                rect_wing = Polygon([(split_m, 0), (w_m, 0), (w_m, wing_depth_m), (split_m, wing_depth_m)])
            else:
                logger.info(f"Orientation: Wing on LEFT. Optimized Depth Ratio: {wing_depth_ratio:.2f}")
                rect_wing = Polygon([(0, 0), (split_m, 0), (split_m, wing_depth_m), (0, wing_depth_m)])
                rect_main = Polygon([(split_m, 0), (w_m, 0), (w_m, d_m), (split_m, d_m)])
            
            sub_footprints = [
                {'poly': rect_main, 'roof_type': 'hip', 'label': 'main'},
                {'poly': rect_wing, 'roof_type': 'gable', 'label': 'wing', 'gable_face': 'front'}
            ]
            footprint = unary_union([rect_main, rect_wing])
        else:
            # Simple Box
            rect = Polygon([(0,0), (w_m, 0), (w_m, d_m), (0, d_m)])
            footprint = rect
            sub_footprints = [{'poly': rect, 'roof_type': 'hip', 'label': 'main'}]
        
        # 6. Roof Pitch Inference
        diagonals = directions.get('diagonal', [])
        # ... (keep existing pitch logic)
        pitch = 0.4
        if diagonals:
            slopes = [abs((l[3]-l[1])/(l[2]-l[0]+1e-6)) for l in diagonals]
            pitch = max(0.2, min(1.0, np.mean(slopes)))

        # 7. INTELLIGENCE: Opening Detection (Windows/Doors)
        openings = self.detect_openings(img_shape, directions)

        topology = {
            'view_type': 'perspective',
            'footprint': footprint,
            'sub_footprints': sub_footprints,
            'height': h_m,
            'width': w_m,
            'depth': d_m,
            'pitch': pitch,
            'openings': openings,
            'scale': px_per_m
        }
        logger.info(f"Perspective Mode: {w_m:.1f}x{d_m:.1f}m. Detected {len(openings)} openings.")
        return topology

    def detect_openings(self, img_shape, directions):
        """
        Aggressive Feature Detection: Find all potential windows/doors.
        """
        openings = []
        verticals = directions.get('vertical', [])
        if not verticals: return []
        
        verticals.sort(key=lambda x: x[0])
        
        # Cross-pair verticals to find rectangles
        for i in range(len(verticals)):
            for j in range(i + 1, min(i + 10, len(verticals))):
                l1, l2 = verticals[i], verticals[j]
                
                dx = abs(l1[0] - l2[0])
                if 10 < dx < 150: # Wider range
                    h1 = abs(l1[3] - l1[1])
                    h2 = abs(l2[3] - l2[1])
                    # Relaxed alignment tolerance
                    if abs(h1 - h2) < 50:
                        w_m = dx / self.pixels_per_metre
                        h_m = max(h1, h2) / self.pixels_per_metre
                        
                        center_x = (l1[0] + l2[0]) / 2
                        w_px_min = min(l[0] for l in verticals)
                        w_px_max = max(l[0] for l in verticals)
                        rel_x = (center_x - w_px_min) / (w_px_max - w_px_min + 1e-6)
                        
                        # Use Y-position to guess if it's a door (near bottom) or window
                        # Normalized Y (0 at top, 1 at bottom of verticals)
                        y_bottom = max(l1[1], l1[3], l2[1], l2[3])
                        y_px_max = max(l[1] for l in verticals + [l1,l2]) # error here potentially
                        # Actually just use simple height threshold
                        is_door = h_m > 1.9
                        
                        openings.append({
                            'w': w_m, 
                            'h': h_m, 
                            'rel_x': rel_x, 
                            'is_door': is_door,
                            'z_level': 0.0 if is_door else 0.9
                        })
        
        # Deduplicate with 10% overlap tolerance
        unique_openings = []
        for op in sorted(openings, key=lambda x: x['w'], reverse=True):
            if not any(abs(op['rel_x'] - u['rel_x']) < 0.1 for u in unique_openings):
                unique_openings.append(op)
                
        return unique_openings
        logger.info(f"Perspective Mode (12m rule): Estimated {w_m:.1f}m x {d_m:.1f}m x {h_m:.1f}m box.")
        return topology

    def find_optimal_wing_depth(self, vertices, split_m, total_d_m, wing_on_right):
        """
        Intelligence Layer: Search for the wing depth that most vertices 'agree' with.
        """
        if not vertices:
            return 0.6 # Default fallback
            
        best_ratio = 0.6
        max_support = -1
        
        # Candidate ratios from 30% to 90%
        for ratio in np.linspace(0.3, 0.9, 7):
            candidate_depth = total_d_m * ratio
            
            # Define the 'probed' boundary line in world coordinates
            # This is the line where the wing ends
            support = 0
            for vx, vy in vertices:
                # Convert vertex back to normalized m (simplified)
                # We check if vertices align with the proposed depth line
                # Note: This is a simplified heuristic for the PoC
                norm_vy = (vy / self.pixels_per_metre) # very rough
                if abs(norm_vy - candidate_depth) < 0.5: # 0.5m tolerance
                    support += 1
            
            if support > max_support:
                max_support = support
                best_ratio = ratio
                
        return best_ratio
        logger.info(f"Perspective Mode (12m rule): Estimated {w_m:.1f}m x {d_m:.1f}m x {h_m:.1f}m box.")
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
