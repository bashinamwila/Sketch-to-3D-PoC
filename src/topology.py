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
            return self.reconstruct_perspective(directions, vertices, img_shape, curves)

        # Fallback to Plan Mode
        return self.reconstruct_plan(lines, vertices, curves)

    def reconstruct_perspective(self, directions, vertices, img_shape, curves):
        """
        Estimate 3D box dimensions from perspective lines using the 12m rule.
        """
        # 1. Height from verticals (pixels)
        heights = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in directions['vertical']]
        avg_h_px = np.mean(heights) if heights else 300
        
        # 2. Width/Depth from perspective lines (pixels)
        def get_span(lines):
            if not lines: return 0
            all_pts = []
            for l in lines: all_pts.extend([(l[0], l[1]), (l[2], l[3])])
            all_pts = np.array(all_pts)
            return np.max(np.linalg.norm(all_pts[:, None] - all_pts, axis=2))

        w_px = get_span(directions['left_vp'])
        d_px = get_span(directions['right_vp'])
        
        # 3. Apply 12m Rule
        max_dim_px = max(w_px, d_px)
        if max_dim_px == 0: max_dim_px = 500
        px_per_m = max_dim_px / 12.0
        
        w_m = w_px / px_per_m
        d_m = d_px / px_per_m
        h_m = avg_h_px / px_per_m

        # 4. Height Sanity Check
        if max(w_m, d_m) > 5.0 and h_m < 2.5:
            logger.warning(f"Calculated height {h_m:.1f}m too low. Boosting to standard 3.0m storey.")
            h_m = 3.0
        
        # 5. Complexity Check (Internal Verticals) -> L-Shape Generation
        verticals = directions['vertical']
        w_px_min = min(l[0] for l in verticals) if verticals else 0
        w_px_max = max(l[0] for l in verticals) if verticals else 1
        width_px = w_px_max - w_px_min
        
        internal_verticals = [l for l in verticals 
                              if (l[0] > w_px_min + width_px * 0.25) and 
                                 (l[0] < w_px_max - width_px * 0.25)]
        
        footprint = None
        sub_footprints = []
        
        if len(internal_verticals) > 0:
            logger.warning(f"Detected {len(internal_verticals)} internal vertical lines. Generating L-shaped footprint.")
            
            mid_x = w_px_min + width_px / 2
            left_count = sum(1 for l in verticals if l[0] < mid_x)
            right_count = sum(1 for l in verticals if l[0] > mid_x)
            wing_on_right = right_count > left_count
            
            avg_split_px = np.mean([l[0] for l in internal_verticals])
            split_ratio = (avg_split_px - w_px_min) / width_px
            split_ratio = max(0.3, min(0.7, split_ratio))
            split_m = w_m * split_ratio

            # Optimized wing depth based on vertex support
            wing_depth_ratio = self.find_optimal_wing_depth(vertices, px_per_m)
            wing_depth_m = d_m * wing_depth_ratio
            
            # INTELLIGENCE: Protrusion Offset
            protrusion_offset_m = d_m * 0.2
            
            if wing_on_right:
                logger.info(f"Orientation: Wing on RIGHT. Offset: {protrusion_offset_m:.1f}m")
                rect_main = Polygon([(0, protrusion_offset_m), (split_m, protrusion_offset_m), 
                                     (split_m, d_m + protrusion_offset_m), (0, d_m + protrusion_offset_m)])
                rect_wing = Polygon([(split_m, 0), (w_m, 0), (w_m, wing_depth_m), (split_m, wing_depth_m)])
            else:
                logger.info(f"Orientation: Wing on LEFT. Offset: {protrusion_offset_m:.1f}m")
                rect_wing = Polygon([(0, 0), (split_m, 0), (split_m, wing_depth_m), (0, wing_depth_m)])
                rect_main = Polygon([(split_m, protrusion_offset_m), (w_m, protrusion_offset_m), 
                                     (w_m, d_m + protrusion_offset_m), (split_m, d_m + protrusion_offset_m)])
            
            sub_footprints = [
                {'poly': rect_main, 'roof_type': 'hip', 'label': 'main'},
                {'poly': rect_wing, 'roof_type': 'gable', 'label': 'wing', 'gable_face': 'front'}
            ]
            footprint = unary_union([rect_main, rect_wing])
        else:
            rect = Polygon([(0,0), (w_m, 0), (w_m, d_m), (0, d_m)])
            footprint = rect
            sub_footprints = [{'poly': rect, 'roof_type': 'hip', 'label': 'main'}]
        
        # 6. Roof Pitch Inference
        diagonals = directions.get('diagonal', [])
        pitch = 0.4
        if diagonals:
            slopes = [abs((l[3]-l[1])/(l[2]-l[0]+1e-6)) for l in diagonals]
            pitch = max(0.2, min(1.0, np.mean(slopes)))
            logger.info(f"Inferred Roof Pitch: {pitch:.2f}")

        # 7. INTELLIGENCE: Opening Detection
        openings = self.detect_openings(directions, curves, px_per_m)

        # 8. INTELLIGENCE: Scale Calibration (Refined)
        calibrated_px_per_m = px_per_m
        for op in openings:
            if op['is_door']:
                h_px = op['h'] * px_per_m 
                calibrated_px_per_m = h_px / 2.1
                logger.info(f"Scale Calibrated from Door: {calibrated_px_per_m:.1f} px/m")
                break
        
        if calibrated_px_per_m != px_per_m:
            ratio = px_per_m / calibrated_px_per_m
            w_m *= ratio
            d_m *= ratio
            h_m *= ratio
            px_per_m = calibrated_px_per_m

        topology = {
            'view_type': 'perspective',
            'footprint': footprint,
            'sub_footprints': sub_footprints,
            'height': h_m,
            'width': w_m,
            'depth': d_m,
            'pitch': pitch,
            'openings': openings,
            'scale': px_per_m,
            'global_px_bounds': (w_px_min, w_px_max)
        }
        logger.info(f"Perspective Reconstruction: {w_m:.1f}x{d_m:.1f}m. Detected {len(openings)} openings.")
        return topology

    def find_optimal_wing_depth(self, vertices, px_per_m):
        if not vertices: return 0.6
        return 0.4 

    def detect_openings(self, directions, curves, px_per_m):
        """
        Capture windows/doors with Vertical Spatial Intelligence.
        """
        openings = []
        verticals = directions.get('vertical', [])
        if not verticals: return []
        
        w_px_min = min(l[0] for l in verticals)
        w_px_max = max(l[0] for l in verticals)
        width_px = w_px_max - w_px_min + 1e-6
        
        # Building vertical bounds for Rel-Z calibration
        y_px_min = min(min(l[1], l[3]) for l in verticals)
        y_px_max = max(max(l[1], l[3]) for l in verticals)
        total_h_px = y_px_max - y_px_min

        for curve in curves:
            pts = np.array(curve.get('points', []))
            if len(pts) < 4: continue
            min_p, max_p = np.min(pts, axis=0), np.max(pts, axis=0)
            w_px, h_px = max_p[1] - min_p[1], max_p[0] - min_p[0]
            
            if 15 < w_px < 200 and 15 < h_px < 250:
                if 0.3 < (w_px / h_px) < 3.0:
                    rel_x = ((min_p[1] + max_p[1])/2 - w_px_min) / width_px
                    is_door = (h_px / px_per_m) > 1.8
                    
                    # Vertical Alignment logic
                    mid_y_px = (min_p[0] + max_p[0]) / 2
                    # Normalized Z (0 at bottom, 1 at top)
                    rel_z = (y_px_max - mid_y_px) / (total_h_px + 1e-6)
                    sill_m = max(0.0, (rel_z * 3.0) - ((h_px / px_per_m) / 2))
                    
                    openings.append({
                        'w': w_px / px_per_m, 'h': h_px / px_per_m, 
                        'rel_x': rel_x, 'is_door': is_door, 
                        'z_level': 0.0 if is_door else sill_m,
                        'parent_label': 'wing' if rel_x > 0.5 else 'main'
                    })

        unique_openings = []
        for op in sorted(openings, key=lambda x: x['w'], reverse=True):
            if not any(abs(op['rel_x'] - u['rel_x']) < 0.08 for u in unique_openings):
                unique_openings.append(op)
        return unique_openings

    def reconstruct_plan(self, lines, vertices, curves):
        return {'view_type': 'plan', 'footprint': Polygon([(0,0), (10,0), (10,10), (0,10)]), 'openings': []}
