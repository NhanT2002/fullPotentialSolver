import gmsh
import sys
import math
import h5py
import numpy as np

# --- Parameters ---
chord = 1.0           # NACA airfoil chord length
r_outer = 100.0       # Outer boundary radius (in chord lengths)
cx, cy = 0.5, 0.0     # Airfoil center (midchord)

# NACA 0012 parameters
naca_digits = "0012"

# Mesh size parameters (will be computed automatically)
mesh_size_inner = 0.02   # Mesh size at airfoil surface
mesh_size_outer = 10.0   # Mesh size at far-field boundary

# Number of points along the airfoil
N_airfoil = 64

# List of N values for mesh convergence studies
N_values = [16, 32, 64, 128, 256, 512, 1024]


def naca_0012_thickness(x, chord=1.0, closed_te=True):
    """
    Compute NACA 0012 thickness distribution.
    
    Parameters:
    -----------
    x : float
        x-coordinate (0 to chord)
    chord : float
        Chord length
    closed_te : bool
        If True, modify coefficients for closed trailing edge
        
    Returns:
    --------
    y_t : float
        Half-thickness at x
    """
    t = 0.12  # Maximum thickness (12% for NACA 0012)
    xn = x / chord  # Normalized x
    
    if closed_te:
        # Modified coefficients for closed trailing edge
        a0 = 0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = 0.2843
        a4 = -0.1036  # Modified from -0.1015 for closed TE
    else:
        # Standard NACA coefficients
        a0 = 0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = 0.2843
        a4 = -0.1015
    
    y_t = (t / 0.2) * chord * (a0 * math.sqrt(xn) + a1 * xn + a2 * xn**2 + 
                                a3 * xn**3 + a4 * xn**4)
    return y_t


def generate_naca_points(n_points, chord=1.0, closed_te=True, le_te_clustering=2.0):
    """
    Generate points along NACA 0012 airfoil surface.
    Uses enhanced cosine spacing for better resolution at leading/trailing edges.
    
    Parameters:
    -----------
    n_points : int
        Number of points on each surface (upper + lower)
    chord : float
        Chord length
    closed_te : bool
        If True, close trailing edge
    le_te_clustering : float
        Clustering factor for LE and TE (1.0 = standard cosine, higher = more clustering)
        Recommended range: 1.5 to 5.0
        
    Returns:
    --------
    points : list of (x, y) tuples
        Points going counterclockwise starting from trailing edge upper surface
    """
    # Use tanh-based clustering which maintains smooth distribution
    # while concentrating points at both ends
    
    # Create uniform parameter t in [0, 1]
    t = np.linspace(0, 1, n_points)
    
    if le_te_clustering > 1.0:
        # Tanh-based stretching: concentrates points at both ends
        # while maintaining smooth transition in the middle
        # Higher 'a' = more clustering at ends
        a = le_te_clustering
        
        # Map t from [0,1] to [-1,1]
        s = 2 * t - 1  # s in [-1, 1]
        
        # Apply tanh stretching
        s_stretched = np.tanh(a * s) / np.tanh(a)
        
        # Map back to [0,1]
        t_clustered = (s_stretched + 1) / 2
        
        # Apply cosine distribution for x-coordinates
        beta = np.pi * t_clustered
    else:
        beta = np.pi * t
    
    # x = 0.5 * (1 + cos(beta)) maps beta=0 -> x=1 (TE), beta=pi -> x=0 (LE)
    x_upper = chord * 0.5 * (1 + np.cos(beta))  # TE to LE
    
    # Upper surface (from TE to LE)
    upper_points = []
    for x in x_upper:
        y = naca_0012_thickness(x, chord, closed_te)
        upper_points.append((x, y))
    
    # Lower surface (from LE to TE, skip LE point as it's shared)
    lower_points = []
    for x in reversed(x_upper[:-1]):  # Skip LE (x=0) which is already in upper
        y = -naca_0012_thickness(x, chord, closed_te)
        lower_points.append((x, y))
    
    # Combine: TE upper -> LE -> TE lower (skip TE as it will be closed by loop)
    # upper_points: TE(x=1) to LE(x=0) on upper surface
    # lower_points: LE+eps to TE-eps on lower surface
    points = upper_points + lower_points
    
    return points


def compute_mesh_sizes(n_airfoil, chord=1.0, r_outer=100.0, target_growth_rate=1.15):
    """
    Automatically compute mesh_inner and mesh_outer sizes for a good quality mesh.
    
    Parameters:
    -----------
    n_airfoil : int
        Number of points along the airfoil surface
    chord : float
        Airfoil chord length
    r_outer : float
        Far-field radius
    target_growth_rate : float
        Desired geometric growth rate (typically 1.1 to 1.3)
        
    Returns:
    --------
    mesh_inner : float
        Mesh size at airfoil surface
    mesh_outer : float
        Mesh size at far-field boundary
    """
    # Approximate arc length of NACA 0012 is roughly 2.05 * chord
    arc_length = 2.05 * chord
    mesh_inner = arc_length / n_airfoil
    
    # For geometric growth from inner to outer
    g = target_growth_rate
    radial_length = r_outer - chord * 0.5
    
    # Number of layers with geometric growth
    n_layers = math.log(1 + radial_length * (g - 1) / mesh_inner) / math.log(g)
    
    # Outer mesh size
    mesh_outer = mesh_inner * (g ** n_layers)
    
    # Clamp to reasonable values
    mesh_outer = min(mesh_outer, r_outer * 0.5)
    mesh_outer = max(mesh_outer, mesh_inner * 2)
    
    return mesh_inner, mesh_outer


def set_cgns_name(node, name):
    """Set the CGNS node name attribute (32 chars, null-padded)"""
    name_bytes = name.encode('ascii')[:32].ljust(32, b'\x00')
    if 'name' in node.attrs:
        node.attrs['name'] = np.frombuffer(name_bytes, dtype='S1')
    else:
        node.attrs.create('name', np.frombuffer(name_bytes, dtype='S1'))


def generate_naca_triangle_mesh(output_file, n_airfoil, mesh_inner, mesh_outer,
                                 chord=1.0, r_outer=100.0, le_te_clustering=2.0):
    """
    Generate an unstructured triangle mesh around a NACA 0012 airfoil.
    
    Parameters:
    -----------
    output_file : str
        Output CGNS filename
    n_airfoil : int
        Number of points along the airfoil surface
    mesh_inner : float
        Mesh size at airfoil surface
    mesh_outer : float
        Mesh size at far-field boundary
    chord : float
        Airfoil chord length
    r_outer : float
        Far-field radius
    le_te_clustering : float
        Clustering factor for leading/trailing edge (1.0=standard, 2.0=enhanced)
    """
    
    print(f"\n{'='*60}")
    print(f"Generating unstructured triangle mesh around NACA 0012")
    print(f"  Chord length: {chord}")
    print(f"  Far-field radius: {r_outer}")
    print(f"  Airfoil points: {n_airfoil}")
    print(f"  Mesh size (inner): {mesh_inner:.6f}")
    print(f"  Mesh size (outer): {mesh_outer:.4f}")
    print(f"  LE/TE clustering: {le_te_clustering}")
    print(f"{'='*60}")
    
    gmsh.initialize()
    gmsh.model.add("NACA0012Triangle")
    
    # --- Generate NACA 0012 airfoil points ---
    airfoil_points = generate_naca_points(n_airfoil, chord, closed_te=True, 
                                          le_te_clustering=le_te_clustering)
    
    # --- Create geometry using built-in kernel ---
    # Add points for airfoil
    point_tags = []
    for i, (x, y) in enumerate(airfoil_points):
        tag = gmsh.model.geo.addPoint(x, y, 0, mesh_inner)
        point_tags.append(tag)
    
    # Create a single closed BSpline through all points
    # Add first point at the end to close the loop
    airfoil_bspline = gmsh.model.geo.addBSpline(point_tags + [point_tags[0]])
    
    airfoil_loop = gmsh.model.geo.addCurveLoop([airfoil_bspline])
    airfoil_curve_tags = [airfoil_bspline]
    
    # Create outer circle (far-field) using arc segments
    center_x = chord * 0.5  # Midchord
    center_y = 0.0
    # Create 4 points on the circle and connect with arcs
    p_center = gmsh.model.geo.addPoint(center_x, center_y, 0)
    p_right = gmsh.model.geo.addPoint(center_x + r_outer, center_y, 0, mesh_outer)
    p_top = gmsh.model.geo.addPoint(center_x, center_y + r_outer, 0, mesh_outer)
    p_left = gmsh.model.geo.addPoint(center_x - r_outer, center_y, 0, mesh_outer)
    p_bottom = gmsh.model.geo.addPoint(center_x, center_y - r_outer, 0, mesh_outer)
    
    arc1 = gmsh.model.geo.addCircleArc(p_right, p_center, p_top)
    arc2 = gmsh.model.geo.addCircleArc(p_top, p_center, p_left)
    arc3 = gmsh.model.geo.addCircleArc(p_left, p_center, p_bottom)
    arc4 = gmsh.model.geo.addCircleArc(p_bottom, p_center, p_right)
    
    outer_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
    outer_curve_tags = [arc1, arc2, arc3, arc4]
    
    # Create domain (outer - inner)
    domain = gmsh.model.geo.addPlaneSurface([outer_loop, airfoil_loop])
    
    gmsh.model.geo.synchronize()
    
    # --- Get curve tags after synchronization ---
    print(f"  Airfoil curve tags: {airfoil_curve_tags}")
    print(f"  Outer curve tags: {outer_curve_tags}")
    
    # --- Mesh size control ---
    # Create distance field from airfoil
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", airfoil_curve_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)
    
    # Create threshold field for mesh size transition
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_inner)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_outer)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r_outer * 0.8)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # --- Physical Groups for boundary conditions ---
    gmsh.model.addPhysicalGroup(1, airfoil_curve_tags, tag=1, name="wall")
    gmsh.model.addPhysicalGroup(1, outer_curve_tags, tag=2, name="farfield")
    gmsh.model.addPhysicalGroup(2, [domain], tag=3, name="Fluid")
    
    # --- Mesh options ---
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
    gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Keep triangles
    gmsh.option.setNumber("Mesh.Smoothing", 10)  # Smoothing passes
    gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear elements
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    
    # Quality optimization
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.QualityType", 2)  # Gamma quality measure
    
    # --- Generate mesh ---
    gmsh.model.mesh.generate(2)
    
    # Optional: optimize mesh quality
    gmsh.model.mesh.optimize("Laplace2D")
    
    # --- Mesh Statistics ---
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
    
    n_triangles = sum(len(t) for t in elem_tags)
    print(f"\n  Mesh statistics:")
    print(f"  - Number of nodes: {len(node_tags)}")
    print(f"  - Number of triangles: {n_triangles}")
    
    # --- Export ---
    gmsh.write(output_file)
    print(f"  - Output file: {output_file}")
    
    gmsh.finalize()
    
    # --- Post-process CGNS file ---
    post_process_cgns(output_file)
    
    print(f"  Done: {output_file}")
    return output_file


def post_process_cgns(output_file):
    """Post-process CGNS file to clean up Gmsh naming and structure."""
    
    print("  Post-processing CGNS file...")
    
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
            # Try to find any zone
            for name in base.keys():
                if isinstance(base[name], h5py.Group):
                    zone_old_name = name
                    break
        
        if zone_old_name is None:
            print("  Error: Could not find zone")
            return
        
        zone = base[zone_old_name]
        
        # --- Find and rename TRI elements section "TriElements" ---
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
        
        # --- Merge BAR elements for wall and farfield ---
        bar_sections = sorted([n for n in zone.keys() if n.startswith("2_")])
        
        if bar_sections:
            # Get element range info to determine tri count
            tri_count = 0
            if "TriElements" in zone and "ElementRange" in zone["TriElements"]:
                tri_range = zone["TriElements"]["ElementRange"][" data"][:]
                tri_count = tri_range[1]
            
            # First bar section is wall (airfoil), rest are farfield arcs
            wall_sec = bar_sections[0] if len(bar_sections) > 0 else None
            farfield_secs = bar_sections[1:] if len(bar_sections) > 1 else []
            
            # Merge farfield sections
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
                    
                    # Get wall edge count for element range calculation
                    n_wall_edges = 0
                    if wall_sec and wall_sec in zone:
                        wall_conn = zone[wall_sec]["ElementConnectivity"][" data"][:]
                        n_wall_edges = len(wall_conn) // 2
                    
                    # Update first farfield section with merged data
                    first_ff_sec = farfield_secs[0]
                    ff_start = tri_count + n_wall_edges + 1
                    
                    if "ElementRange" in zone[first_ff_sec] and " data" in zone[first_ff_sec]["ElementRange"]:
                        zone[first_ff_sec]["ElementRange"][" data"][...] = np.array([ff_start, ff_start + n_ff_edges - 1], dtype=np.int32)
                    
                    if "ElementConnectivity" in zone[first_ff_sec]:
                        if " data" in zone[first_ff_sec]["ElementConnectivity"]:
                            del zone[first_ff_sec]["ElementConnectivity"][" data"]
                        zone[first_ff_sec]["ElementConnectivity"].create_dataset(" data", data=merged_ff_conn.astype(np.int32))
                    
                    set_cgns_name(zone[first_ff_sec], "farfield")
                    zone.move(first_ff_sec, "farfield")
                    
                    # Delete remaining farfield sections
                    for sec_name in farfield_secs[1:]:
                        if sec_name in zone:
                            del zone[sec_name]
            
            # Rename wall section
            if wall_sec and wall_sec in zone:
                # Update wall element range
                wall_conn = zone[wall_sec]["ElementConnectivity"][" data"][:]
                n_wall_edges = len(wall_conn) // 2
                wall_start = tri_count + 1
                zone[wall_sec]["ElementRange"][" data"][...] = np.array([wall_start, wall_start + n_wall_edges - 1], dtype=np.int32)
                
                set_cgns_name(zone[wall_sec], "wall")
                zone.move(wall_sec, "wall")
        
        # Delete ZoneBC if exists
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


# --- Main execution ---
if __name__ == "__main__":

    # List of mesh densities to generate
    N_values = [32, 64, 128, 256, 512, 1024]
    
    # LE/TE clustering factor (1.0 = standard cosine, 2.0 = enhanced, 3.0 = strong)
    le_te_clustering = 2.0

    print(f"\n{'='*70}")
    print(f"Generating NACA 0012 unstructured triangle meshes")
    print(f"  LE/TE clustering factor: {le_te_clustering}")
    print(f"{'='*70}")
    print(f"{'N':>6} | {'mesh_inner':>12} | {'mesh_outer':>12} | {'est. layers':>12}")
    print(f"{'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    
    for n in N_values:
        # Automatically compute mesh sizes for good growth rate
        mesh_inner, mesh_outer = compute_mesh_sizes(
            n_airfoil=n, 
            chord=chord, 
            r_outer=r_outer, 
            target_growth_rate=1.15
        )
        
        # Estimate number of radial layers
        g = 1.15
        radial_length = r_outer - chord * 0.5
        n_layers = math.log(1 + radial_length * (g - 1) / mesh_inner) / math.log(g)
        
        print(f"{n:>6} | {mesh_inner:>12.6f} | {mesh_outer:>12.4f} | {n_layers:>12.1f}")
        
        generate_naca_triangle_mesh(
            output_file=f"naca0012_triangle_{n}.cgns",
            n_airfoil=n,
            mesh_inner=mesh_inner,
            mesh_outer=mesh_outer,
            chord=chord,
            r_outer=r_outer,
            le_te_clustering=le_te_clustering
        )
    
    print(f"\n{'='*70}")
    print(f"All NACA 0012 triangle meshes generated successfully!")
    print(f"{'='*70}")
