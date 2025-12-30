#!/usr/bin/env python3
"""
Generate unstructured triangle meshes for NACA 0012 airfoil by reading
airfoil coordinates from existing hyperbolic (structured) CGNS meshes.

This ensures the airfoil geometry matches exactly with the structured grids.
Output CGNS structure matches the original hyperbolic meshes.
"""

import gmsh
import numpy as np
import h5py
import os


def read_airfoil_from_cgns(cgns_file):
    """
    Read airfoil wall coordinates from a structured CGNS mesh.
    
    Returns:
        np.ndarray: Nx2 array of (x, y) coordinates of airfoil points
    """
    with h5py.File(cgns_file, 'r') as f:
        x = f['Base/dom-1/GridCoordinates/CoordinateX/ data'][:]
        y = f['Base/dom-1/GridCoordinates/CoordinateY/ data'][:]
        
        # Get wall element connectivity to find wall node indices
        wall_conn = f['Base/dom-1/wall/ElementConnectivity/ data'][:]
        
        # Get unique node indices (1-based in CGNS)
        wall_node_indices = sorted(set(wall_conn))
        
        # Convert to 0-based and extract coordinates
        airfoil_points = []
        for node_idx in wall_node_indices:
            airfoil_points.append([x[node_idx - 1], y[node_idx - 1]])
        
        return np.array(airfoil_points)


def order_airfoil_points(points):
    """
    Order airfoil points for proper curve construction.
    Points should go: LE -> upper surface -> TE -> lower surface -> back to LE
    """
    # Find leading edge (minimum x)
    le_idx = np.argmin(points[:, 0])
    
    # Find trailing edge (maximum x)
    te_idx = np.argmax(points[:, 0])
    
    # Separate upper and lower surfaces
    # Upper: y > 0 (or LE to TE going through positive y)
    # Lower: y < 0
    
    # Simple approach: sort by angle from centroid
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    
    return points[sorted_indices]


def generate_triangle_mesh_from_cgns(input_cgns, output_cgns, r_outer=100.0, 
                                      target_growth_rate=1.2, add_wake=False):
    """
    Generate an unstructured triangle mesh using airfoil coordinates from
    a structured CGNS mesh.
    
    Args:
        input_cgns: Path to input structured CGNS mesh
        output_cgns: Path to output unstructured CGNS mesh
        r_outer: Outer boundary radius
        target_growth_rate: Mesh growth rate from wall to farfield
        add_wake: If True, add a wake line from trailing edge to farfield
    """
    # Read airfoil coordinates
    airfoil_points = read_airfoil_from_cgns(input_cgns)
    n_airfoil = len(airfoil_points)
    
    print(f"  Read {n_airfoil} airfoil points from {os.path.basename(input_cgns)}")
    
    # Determine chord length
    chord = np.max(airfoil_points[:, 0]) - np.min(airfoil_points[:, 0])
    print(f"  Chord length: {chord:.4f}")
    
    # Order points properly (they should already be ordered from CGNS wall)
    # The CGNS wall goes LE -> upper -> TE -> lower
    # We need them in a closed loop
    
    # Check if points form a proper loop
    # From the exploration: points go LE(0,0) -> upper -> TE -> lower -> back
    # But it's not closed - need to verify
    
    # Find LE and TE
    le_idx = np.argmin(airfoil_points[:, 0])
    te_idx = np.argmax(airfoil_points[:, 0])
    
    print(f"  LE at index {le_idx}: ({airfoil_points[le_idx, 0]:.6f}, {airfoil_points[le_idx, 1]:.6f})")
    print(f"  TE at index {te_idx}: ({airfoil_points[te_idx, 0]:.6f}, {airfoil_points[te_idx, 1]:.6f})")
    
    # Calculate mesh sizes
    # Wall size: based on spacing between airfoil points
    wall_spacings = np.sqrt(np.sum(np.diff(airfoil_points, axis=0)**2, axis=1))
    h_wall = np.mean(wall_spacings)
    h_wall_min = np.min(wall_spacings)
    
    # Farfield size: based on growth rate
    n_layers = int(np.log(r_outer / chord) / np.log(target_growth_rate)) + 10
    h_farfield = h_wall * (target_growth_rate ** n_layers)
    h_farfield = min(h_farfield, r_outer * 0.1)  # Cap at 10% of radius
    
    print(f"  Wall mesh size: {h_wall:.6f} (min: {h_wall_min:.6f})")
    print(f"  Farfield mesh size: {h_farfield:.4f}")
    
    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("naca_triangle")
    
    # Create airfoil points in gmsh
    # The points from CGNS go: LE -> upper -> TE -> lower -> back to LE
    # We need to split at TE to create two curves (upper and lower)
    # so the TE becomes a vertex for the wake line
    
    point_tags = []
    for i, (x, y) in enumerate(airfoil_points):
        tag = gmsh.model.geo.addPoint(x, y, 0, h_wall)
        point_tags.append(tag)
    
    # Check if closed
    dist_to_close = np.linalg.norm(airfoil_points[0] - airfoil_points[-1])
    print(f"  Distance between first and last point: {dist_to_close:.6f}")
    
    if add_wake:
        # Split airfoil into upper and lower curves at TE
        # Upper: from LE (index 0) to TE (index te_idx)
        # Lower: from TE (index te_idx) to LE (close the loop)
        
        upper_points = point_tags[0:te_idx+1]  # LE to TE
        lower_points = point_tags[te_idx:]     # TE to end
        
        # Close the loop by adding the first point
        if dist_to_close > 1e-6:
            lower_points = lower_points + [point_tags[0]]
        
        # Create two BSpline curves
        upper_curve = gmsh.model.geo.addBSpline(upper_points)
        lower_curve = gmsh.model.geo.addBSpline(lower_points)
        
        airfoil_loop = gmsh.model.geo.addCurveLoop([upper_curve, lower_curve])
        airfoil_curves = [upper_curve, lower_curve]
    else:
        # Single closed BSpline for the whole airfoil
        if dist_to_close > 1e-6:
            bspline_points = point_tags + [point_tags[0]]
        else:
            bspline_points = point_tags
        
        airfoil_curve = gmsh.model.geo.addBSpline(bspline_points)
        airfoil_loop = gmsh.model.geo.addCurveLoop([airfoil_curve])
        airfoil_curves = [airfoil_curve]
    
    # Get trailing edge coordinates for wake line
    te_x = airfoil_points[te_idx, 0]
    te_y = airfoil_points[te_idx, 1]
    
    # Create outer circular boundary
    center = gmsh.model.geo.addPoint(0.5 * chord, 0, 0, h_farfield)
    
    # Four points on the circle
    p_right = gmsh.model.geo.addPoint(0.5 * chord + r_outer, 0, 0, h_farfield)
    p_top = gmsh.model.geo.addPoint(0.5 * chord, r_outer, 0, h_farfield)
    p_left = gmsh.model.geo.addPoint(0.5 * chord - r_outer, 0, 0, h_farfield)
    p_bottom = gmsh.model.geo.addPoint(0.5 * chord, -r_outer, 0, h_farfield)
    
    # Four arcs
    arc1 = gmsh.model.geo.addCircleArc(p_right, center, p_top)
    arc2 = gmsh.model.geo.addCircleArc(p_top, center, p_left)
    arc3 = gmsh.model.geo.addCircleArc(p_left, center, p_bottom)
    arc4 = gmsh.model.geo.addCircleArc(p_bottom, center, p_right)
    
    if add_wake:
        # Create wake line from trailing edge to farfield
        # The TE point is point_tags[te_idx]
        te_point = point_tags[te_idx]
        
        # Create wake line from TE to p_right (farfield at x = 0.5*chord + r_outer)
        wake_line = gmsh.model.geo.addLine(te_point, p_right)
        
        # The domain is now split by the wake line
        # Upper region: arc1 (right->top), arc2 (top->left), arc3 (left->bottom), 
        #               arc4 partial (bottom->right up to wake), wake line, airfoil upper
        # But simpler: embed the wake line in the surface
        
        outer_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
        
        # Create surface (outer boundary minus airfoil hole)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop, airfoil_loop])
        
        gmsh.model.geo.synchronize()
        
        # Embed the wake line in the surface so mesh conforms to it
        gmsh.model.mesh.embed(1, [wake_line], 2, surface)
        
        print(f"  Wake line added from TE ({te_x:.4f}, {te_y:.4f}) to farfield")
    else:
        outer_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
        
        # Create surface (outer boundary minus airfoil hole)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop, airfoil_loop])
        
        gmsh.model.geo.synchronize()
        wake_line = None
    
    # Physical groups for boundary conditions
    wall_group = gmsh.model.addPhysicalGroup(1, airfoil_curves, name="wall")
    farfield_group = gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], name="farfield")
    if add_wake and wake_line is not None:
        wake_group = gmsh.model.addPhysicalGroup(1, [wake_line], name="wake")
    fluid_group = gmsh.model.addPhysicalGroup(2, [surface], name="Fluid")
    
    # Mesh size field for smooth transition
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", airfoil_curves)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_wall)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_farfield)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1 * chord)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r_outer * 0.5)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # Meshing options
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Get mesh statistics
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    n_triangles = sum(len(t) for t in elem_tags)
    
    print(f"  - Number of nodes: {len(node_tags)}")
    print(f"  - Number of triangles: {n_triangles}")
    
    # Write CGNS using gmsh (will be post-processed)
    gmsh.write(output_cgns)
    
    gmsh.finalize()
    
    # Post-process CGNS file to match hyperbolic mesh structure
    post_process_cgns(output_cgns)
    
    print(f"  Done: {os.path.basename(output_cgns)}")


def set_cgns_name(node, name):
    """Set the CGNS node name attribute (32 chars, null-padded)"""
    name_bytes = name.encode('ascii')[:32].ljust(32, b'\x00')
    if 'name' in node.attrs:
        node.attrs['name'] = np.frombuffer(name_bytes, dtype='S1')
    else:
        node.attrs.create('name', np.frombuffer(name_bytes, dtype='S1'))


def post_process_cgns(output_file):
    """
    Post-process CGNS file to clean up Gmsh naming and structure.
    Makes the structure match the original hyperbolic meshes.
    """
    
    with h5py.File(output_file, "r+") as f:
        # Find the base node
        base_name = None
        for name in f.keys():
            if name not in ['CGNSLibraryVersion', ' format', ' hdf5version']:
                base_name = name
                break
        
        if base_name is None:
            print("  Error: Could not find base node")
            return
            
        base = f[base_name]
        
        # Find zone name
        zone_old_name = None
        for name in base.keys():
            if "Part" in name or "Zone" in name.lower():
                zone_old_name = name
                break
        
        if zone_old_name is None:
            # Try to find any zone with GridCoordinates
            for name in base.keys():
                if isinstance(base[name], h5py.Group) and 'GridCoordinates' in base[name]:
                    zone_old_name = name
                    break
        
        if zone_old_name is None:
            print("  Error: Could not find zone")
            return
        
        zone = base[zone_old_name]
        
        # --- Find and rename TRI elements section "TriElements" ---
        # Gmsh names triangle sections as "3_S_*" (3 = TRI_3 element type)
        tri_sections = sorted([n for n in zone.keys() if n.startswith("3_")])
        if tri_sections:
            all_tri_conn = []
            for sec_name in tri_sections:
                sec = zone[sec_name]
                if "ElementConnectivity" in sec and " data" in sec["ElementConnectivity"]:
                    conn_data = sec["ElementConnectivity"][" data"][:]
                    all_tri_conn.append(conn_data)
            
            if all_tri_conn:
                merged_tri_conn = np.concatenate(all_tri_conn)
                n_tris = len(merged_tri_conn) // 3
                first_start = 1
                first_sec = tri_sections[0]
                
                # Update element range
                if "ElementRange" in zone[first_sec] and " data" in zone[first_sec]["ElementRange"]:
                    zone[first_sec]["ElementRange"][" data"][...] = np.array([first_start, first_start + n_tris - 1], dtype=np.int32)
                
                # Delete old connectivity and create new merged one
                if "ElementConnectivity" in zone[first_sec]:
                    if " data" in zone[first_sec]["ElementConnectivity"]:
                        del zone[first_sec]["ElementConnectivity"][" data"]
                    zone[first_sec]["ElementConnectivity"].create_dataset(" data", data=merged_tri_conn.astype(np.int32))
                
                set_cgns_name(zone[first_sec], "TriElements")
                zone.move(first_sec, "TriElements")
                
                for sec_name in tri_sections[1:]:
                    if sec_name in zone:
                        del zone[sec_name]
        
        # --- Merge BAR elements for wall, farfield, and wake ---
        # Gmsh names BAR sections as "2_L_*" (2 = BAR_2 element type)
        # Order depends on whether wake is added:
        # - Without wake: wall (1 section), farfield (4 sections)
        # - With wake: wall_upper (1), wall_lower (1), farfield (4), wake (1)
        bar_sections = sorted([n for n in zone.keys() if n.startswith("2_")])
        
        if bar_sections:
            n_bar = len(bar_sections)
            
            # Determine which sections are wall, farfield, and wake
            # Wake is always the last section if n_bar == 7 (2 wall + 4 farfield + 1 wake)
            # or n_bar == 6 (1 wall + 4 farfield + 1 wake)
            if n_bar == 7:
                # With wake and split wall (2 wall curves)
                wall_secs = bar_sections[0:2]
                farfield_secs = bar_sections[2:6]
                wake_sec = bar_sections[6]
            elif n_bar == 6:
                # With wake and single wall curve
                wall_secs = [bar_sections[0]]
                farfield_secs = bar_sections[1:5]
                wake_sec = bar_sections[5]
            elif n_bar == 5:
                # Without wake
                wall_secs = [bar_sections[0]]
                farfield_secs = bar_sections[1:5]
                wake_sec = None
            else:
                # Fallback
                wall_secs = [bar_sections[0]] if n_bar > 0 else []
                farfield_secs = bar_sections[1:] if n_bar > 1 else []
                wake_sec = None
            
            # Merge wall sections
            if wall_secs:
                all_wall_conn = []
                for sec_name in wall_secs:
                    sec = zone[sec_name]
                    if "ElementConnectivity" in sec and " data" in sec["ElementConnectivity"]:
                        conn_data = sec["ElementConnectivity"][" data"][:]
                        all_wall_conn.append(conn_data)
                
                if all_wall_conn:
                    merged_wall_conn = np.concatenate(all_wall_conn)
                    n_wall_edges = len(merged_wall_conn) // 2
                    first_wall_sec = wall_secs[0]
                    
                    # Get TriElements end
                    tri_end = 0
                    if "TriElements" in zone and "ElementRange" in zone["TriElements"]:
                        tri_end = zone["TriElements"]["ElementRange"][" data"][1]
                    
                    wall_start = tri_end + 1
                    
                    # Update element range
                    if "ElementRange" in zone[first_wall_sec] and " data" in zone[first_wall_sec]["ElementRange"]:
                        zone[first_wall_sec]["ElementRange"][" data"][...] = np.array([wall_start, wall_start + n_wall_edges - 1], dtype=np.int32)
                    
                    # Update connectivity
                    if "ElementConnectivity" in zone[first_wall_sec]:
                        if " data" in zone[first_wall_sec]["ElementConnectivity"]:
                            del zone[first_wall_sec]["ElementConnectivity"][" data"]
                        zone[first_wall_sec]["ElementConnectivity"].create_dataset(" data", data=merged_wall_conn.astype(np.int32))
                    
                    set_cgns_name(zone[first_wall_sec], "wall")
                    zone.move(first_wall_sec, "wall")
                    
                    # Delete remaining wall sections
                    for sec_name in wall_secs[1:]:
                        if sec_name in zone:
                            del zone[sec_name]
            
            # Merge all farfield sections
            if farfield_secs:
                all_ff_conn = []
                for sec_name in farfield_secs:
                    sec = zone[sec_name]
                    if "ElementConnectivity" in sec and " data" in sec["ElementConnectivity"]:
                        conn_data = sec["ElementConnectivity"][" data"][:]
                        all_ff_conn.append(conn_data)
                
                if all_ff_conn:
                    merged_ff_conn = np.concatenate(all_ff_conn)
                    n_ff_edges = len(merged_ff_conn) // 2
                    first_sec = farfield_secs[0]
                    
                    # Get current element ranges
                    tri_end = 0
                    wall_end = 0
                    if "TriElements" in zone and "ElementRange" in zone["TriElements"]:
                        tri_end = zone["TriElements"]["ElementRange"][" data"][1]
                    if "wall" in zone and "ElementRange" in zone["wall"]:
                        wall_end = zone["wall"]["ElementRange"][" data"][1]
                    
                    ff_start = max(tri_end, wall_end) + 1
                    
                    # Update element range
                    if "ElementRange" in zone[first_sec] and " data" in zone[first_sec]["ElementRange"]:
                        zone[first_sec]["ElementRange"][" data"][...] = np.array([ff_start, ff_start + n_ff_edges - 1], dtype=np.int32)
                    
                    # Delete old connectivity and create new merged one
                    if "ElementConnectivity" in zone[first_sec]:
                        if " data" in zone[first_sec]["ElementConnectivity"]:
                            del zone[first_sec]["ElementConnectivity"][" data"]
                        zone[first_sec]["ElementConnectivity"].create_dataset(" data", data=merged_ff_conn.astype(np.int32))
                    
                    set_cgns_name(zone[first_sec], "farfield")
                    zone.move(first_sec, "farfield")
                    
                    # Delete remaining farfield sections
                    for sec_name in farfield_secs[1:]:
                        if sec_name in zone:
                            del zone[sec_name]
            
            # Handle wake section if present
            if wake_sec and wake_sec in zone:
                # Get current element ranges
                ff_end = 0
                if "farfield" in zone and "ElementRange" in zone["farfield"]:
                    ff_end = zone["farfield"]["ElementRange"][" data"][1]
                elif "wall" in zone and "ElementRange" in zone["wall"]:
                    ff_end = zone["wall"]["ElementRange"][" data"][1]
                
                wake_conn = None
                if "ElementConnectivity" in zone[wake_sec] and " data" in zone[wake_sec]["ElementConnectivity"]:
                    wake_conn = zone[wake_sec]["ElementConnectivity"][" data"][:]
                
                if wake_conn is not None:
                    n_wake_edges = len(wake_conn) // 2
                    wake_start = ff_end + 1
                    
                    # Update element range
                    if "ElementRange" in zone[wake_sec] and " data" in zone[wake_sec]["ElementRange"]:
                        zone[wake_sec]["ElementRange"][" data"][...] = np.array([wake_start, wake_start + n_wake_edges - 1], dtype=np.int32)
                
                set_cgns_name(zone[wake_sec], "wake")
                zone.move(wake_sec, "wake")
        
        # Delete ZoneBC if exists (we'll recreate it properly if needed)
        if "ZoneBC" in zone:
            del zone["ZoneBC"]
        
        # Rename Zone to "dom-1"
        set_cgns_name(zone, "dom-1")
        if zone_old_name != "dom-1":
            base.move(zone_old_name, "dom-1")
        
        # Clean up old Family nodes
        families_to_delete = [n for n in base.keys() if n.startswith("L_") or n.startswith("S_")]
        for fam in families_to_delete:
            del base[fam]
        
        # Rename Base node
        set_cgns_name(base, "Base")
        if base_name != "Base":
            f.move(base_name, "Base")


def main():
    """Generate triangle meshes for multiple grid sizes."""
    
    # Available structured mesh sizes
    N_values = [33, 65, 129, 257, 513, 1025, 2049]
    
    # Option to add wake line from trailing edge to farfield
    add_wake = False
    
    # Check which input files exist
    available_inputs = []
    for N in N_values:
        input_file = f"naca0012_{N}x{N}.cgns"
        if os.path.exists(input_file):
            available_inputs.append((N, input_file))
        else:
            print(f"Input file not found: {input_file}")
    
    if not available_inputs:
        print("No input CGNS files found!")
        return
    
    wake_status = "with wake line" if add_wake else "without wake line"
    print(f"\nGenerating triangle meshes from {len(available_inputs)} structured grids ({wake_status})")
    print("=" * 60)
    
    for N, input_file in available_inputs:
        if add_wake:
            output_file = f"naca0012_triangle_{N}x{N}_wake.cgns"
        else:
            output_file = f"naca0012_triangle_{N}x{N}.cgns"
        print(f"\nProcessing {input_file}:")
        
        try:
            generate_triangle_mesh_from_cgns(
                input_cgns=input_file,
                output_cgns=output_file,
                r_outer=100.0,
                target_growth_rate=1.2,
                add_wake=add_wake
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
