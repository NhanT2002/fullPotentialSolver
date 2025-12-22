import gmsh
import sys
import math
import h5py
import numpy as np

# --- Parameters ---
r_inner = 0.5       # Cylinder radius
r_outer = 100.0     # Outer boundary radius
cx, cy = 0.0, 0.0   # Cylinder center

# Mesh size parameters
mesh_size_inner = 0.05   # Mesh size at cylinder surface
mesh_size_outer = 10.0   # Mesh size at far-field boundary
growth_rate = 1.1        # Growth rate for boundary layer

# Number of points around the cylinder
N_circum = 64

# List of N values for mesh convergence studies
N_values = [8, 16, 32, 64, 128, 256, 512]


def compute_mesh_sizes(n_circum, r_inner=0.5, r_outer=100.0, target_growth_rate=1.15):
    """
    Automatically compute mesh_inner and mesh_outer sizes for a good quality mesh.
    
    The idea is:
    - mesh_inner matches the circumferential cell size for isotropic cells at the wall
    - mesh_outer is computed to achieve approximately the target growth rate
      across the radial direction
    
    Parameters:
    -----------
    n_circum : int
        Number of points around the cylinder circumference
    r_inner : float
        Cylinder radius
    r_outer : float
        Far-field radius
    target_growth_rate : float
        Desired geometric growth rate (typically 1.1 to 1.3)
        
    Returns:
    --------
    mesh_inner : float
        Mesh size at cylinder surface
    mesh_outer : float
        Mesh size at far-field boundary
    """
    # Circumferential cell size at the cylinder
    mesh_inner = 2 * math.pi * r_inner / n_circum
    
    # For a geometric progression with constant growth rate:
    # If cell size grows as h(r) = h_inner * g^k where k is the "layer" number
    # and we want h(r) to scale approximately with distance from origin,
    # a good approximation is h_outer = h_inner * (r_outer / r_inner) * factor
    #
    # More precisely, for smooth growth:
    # - Number of radial layers ≈ (r_outer - r_inner) / avg_cell_size
    # - With geometric growth: sum = h_inner * (g^n - 1) / (g - 1) = r_outer - r_inner
    # - Solving for n: n = log(1 + (r_outer - r_inner) * (g-1) / h_inner) / log(g)
    # - Then h_outer = h_inner * g^n
    
    radial_length = r_outer - r_inner
    g = target_growth_rate
    
    # Number of layers with geometric growth
    n_layers = math.log(1 + radial_length * (g - 1) / mesh_inner) / math.log(g)
    
    # Outer mesh size
    mesh_outer = mesh_inner * (g ** n_layers)
    
    # Clamp to reasonable values
    mesh_outer = min(mesh_outer, r_outer * 0.5)  # Don't exceed half the domain
    mesh_outer = max(mesh_outer, mesh_inner * 2)  # At least 2x inner
    
    return mesh_inner, mesh_outer


def set_cgns_name(node, name):
    """Set the CGNS node name attribute (32 chars, null-padded)"""
    name_bytes = name.encode('ascii')[:32].ljust(32, b'\x00')
    if 'name' in node.attrs:
        node.attrs['name'] = np.frombuffer(name_bytes, dtype='S1')
    else:
        node.attrs.create('name', np.frombuffer(name_bytes, dtype='S1'))


def uniform_refine_mesh(input_file, output_file, n_refinements=1):
    """
    Uniformly refine a triangle mesh using 1-to-4 subdivision.
    
    Each triangle is divided into 4 smaller triangles by adding
    midpoints on each edge. This preserves mesh quality and produces
    geometrically similar triangles.
    
    Parameters:
    -----------
    input_file : str
        Input CGNS filename
    output_file : str
        Output CGNS filename
    n_refinements : int
        Number of refinement levels (each level multiplies elements by 4)
    """
    import shutil
    
    # Copy input to output first
    shutil.copy(input_file, output_file)
    
    for ref_level in range(n_refinements):
        print(f"  Refinement level {ref_level + 1}/{n_refinements}...")
        
        with h5py.File(output_file, "r+") as f:
            # Navigate to zone
            base = f["Base"]
            zone = base["dom-1"]
            
            # Read coordinates
            coords_grp = zone["GridCoordinates"]
            coord_x = coords_grp["CoordinateX"][" data"][:]
            coord_y = coords_grp["CoordinateY"][" data"][:]
            coord_z = coords_grp["CoordinateZ"][" data"][:] if "CoordinateZ" in coords_grp else np.zeros_like(coord_x)
            
            n_nodes_orig = len(coord_x)
            
            # Read triangle connectivity
            tri_conn = zone["TriElements"]["ElementConnectivity"][" data"][:]
            n_tris_orig = len(tri_conn) // 3
            triangles = tri_conn.reshape(n_tris_orig, 3)
            
            # Read boundary elements
            wall_conn = zone["wall"]["ElementConnectivity"][" data"][:]
            wall_range = zone["wall"]["ElementRange"][" data"][:]
            n_wall_edges = len(wall_conn) // 2
            wall_edges = wall_conn.reshape(n_wall_edges, 2)
            
            farfield_conn = zone["farfield"]["ElementConnectivity"][" data"][:]
            farfield_range = zone["farfield"]["ElementRange"][" data"][:]
            n_ff_edges = len(farfield_conn) // 2
            farfield_edges = farfield_conn.reshape(n_ff_edges, 2)
            
            # --- Create edge-to-midpoint mapping ---
            # Use frozenset of node indices as edge key
            edge_to_midpoint = {}
            new_nodes_x = []
            new_nodes_y = []
            new_nodes_z = []
            next_node_id = n_nodes_orig + 1  # CGNS uses 1-based indexing
            
            def get_or_create_midpoint(n1, n2):
                """Get existing midpoint or create new one for edge (n1, n2)"""
                nonlocal next_node_id
                edge_key = frozenset([n1, n2])
                
                if edge_key not in edge_to_midpoint:
                    # Create new midpoint (convert to 0-based for array access)
                    mid_x = 0.5 * (coord_x[n1 - 1] + coord_x[n2 - 1])
                    mid_y = 0.5 * (coord_y[n1 - 1] + coord_y[n2 - 1])
                    mid_z = 0.5 * (coord_z[n1 - 1] + coord_z[n2 - 1])
                    
                    new_nodes_x.append(mid_x)
                    new_nodes_y.append(mid_y)
                    new_nodes_z.append(mid_z)
                    
                    edge_to_midpoint[edge_key] = next_node_id
                    next_node_id += 1
                
                return edge_to_midpoint[edge_key]
            
            # --- Subdivide triangles ---
            new_triangles = []
            for tri in triangles:
                v0, v1, v2 = tri[0], tri[1], tri[2]
                
                # Get midpoints of each edge
                m01 = get_or_create_midpoint(v0, v1)
                m12 = get_or_create_midpoint(v1, v2)
                m20 = get_or_create_midpoint(v2, v0)
                
                # Create 4 new triangles
                # Corner triangles (preserve orientation)
                new_triangles.append([v0, m01, m20])
                new_triangles.append([m01, v1, m12])
                new_triangles.append([m20, m12, v2])
                # Center triangle
                new_triangles.append([m01, m12, m20])
            
            # --- Subdivide boundary edges ---
            new_wall_edges = []
            for edge in wall_edges:
                n1, n2 = edge[0], edge[1]
                mid = get_or_create_midpoint(n1, n2)
                new_wall_edges.append([n1, mid])
                new_wall_edges.append([mid, n2])
            
            new_ff_edges = []
            for edge in farfield_edges:
                n1, n2 = edge[0], edge[1]
                mid = get_or_create_midpoint(n1, n2)
                new_ff_edges.append([n1, mid])
                new_ff_edges.append([mid, n2])
            
            # --- Project new boundary nodes to exact geometry ---
            # For cylinder mesh, project wall nodes to inner circle and farfield to outer
            for edge_key, mid_id in edge_to_midpoint.items():
                nodes = list(edge_key)
                n1, n2 = nodes[0], nodes[1]
                
                # Check if this edge is on wall boundary
                is_wall = any((n1 in edge and n2 in edge) for edge in wall_edges)
                is_farfield = any((n1 in edge and n2 in edge) for edge in farfield_edges)
                
                if is_wall or is_farfield:
                    idx = mid_id - n_nodes_orig - 1  # Index in new_nodes arrays
                    x, y = new_nodes_x[idx], new_nodes_y[idx]
                    
                    # Project to circle
                    r_current = math.sqrt(x**2 + y**2)
                    r_target = r_inner if is_wall else r_outer
                    
                    if r_current > 1e-10:
                        scale = r_target / r_current
                        new_nodes_x[idx] = x * scale
                        new_nodes_y[idx] = y * scale
            
            # --- Update CGNS file ---
            # Update coordinates
            n_nodes_new = n_nodes_orig + len(new_nodes_x)
            all_x = np.concatenate([coord_x, np.array(new_nodes_x)])
            all_y = np.concatenate([coord_y, np.array(new_nodes_y)])
            all_z = np.concatenate([coord_z, np.array(new_nodes_z)])
            
            del coords_grp["CoordinateX"][" data"]
            coords_grp["CoordinateX"].create_dataset(" data", data=all_x.astype(np.float64))
            
            del coords_grp["CoordinateY"][" data"]
            coords_grp["CoordinateY"].create_dataset(" data", data=all_y.astype(np.float64))
            
            del coords_grp["CoordinateZ"][" data"]
            coords_grp["CoordinateZ"].create_dataset(" data", data=all_z.astype(np.float64))
            
            # Update triangles
            new_tri_conn = np.array(new_triangles, dtype=np.int32).flatten()
            n_tris_new = len(new_triangles)
            
            del zone["TriElements"]["ElementConnectivity"][" data"]
            zone["TriElements"]["ElementConnectivity"].create_dataset(" data", data=new_tri_conn)
            zone["TriElements"]["ElementRange"][" data"][...] = np.array([1, n_tris_new], dtype=np.int32)
            
            # Update wall edges
            new_wall_conn = np.array(new_wall_edges, dtype=np.int32).flatten()
            n_wall_new = len(new_wall_edges)
            wall_start = n_tris_new + 1
            
            del zone["wall"]["ElementConnectivity"][" data"]
            zone["wall"]["ElementConnectivity"].create_dataset(" data", data=new_wall_conn)
            zone["wall"]["ElementRange"][" data"][...] = np.array([wall_start, wall_start + n_wall_new - 1], dtype=np.int32)
            
            # Update farfield edges
            new_ff_conn = np.array(new_ff_edges, dtype=np.int32).flatten()
            n_ff_new = len(new_ff_edges)
            ff_start = wall_start + n_wall_new
            
            del zone["farfield"]["ElementConnectivity"][" data"]
            zone["farfield"]["ElementConnectivity"].create_dataset(" data", data=new_ff_conn)
            zone["farfield"]["ElementRange"][" data"][...] = np.array([ff_start, ff_start + n_ff_new - 1], dtype=np.int32)
            
            # Update zone data array (nodes, elements, boundary)
            zone_data = zone[" data"][:]
            zone_data[0, 0] = n_nodes_new
            zone_data[1, 0] = n_tris_new + n_wall_new + n_ff_new
            zone[" data"][...] = zone_data
        
        print(f"    Nodes: {n_nodes_orig} -> {n_nodes_new}")
        print(f"    Triangles: {n_tris_orig} -> {n_tris_new}")
    
    print(f"  Refined mesh written to: {output_file}")


def generate_unstructured_cylinder_mesh(output_file, n_circum, mesh_inner, mesh_outer, 
                                         boundary_layer=True, bl_thickness=1.0, 
                                         bl_layers=10, bl_growth=1.2):
    """
    Generate an unstructured triangle mesh around a cylinder.
    
    Parameters:
    -----------
    output_file : str
        Output CGNS filename
    n_circum : int
        Number of points around the cylinder circumference
    mesh_inner : float
        Mesh size at cylinder surface
    mesh_outer : float
        Mesh size at far-field boundary
    boundary_layer : bool
        Whether to add a boundary layer mesh
    bl_thickness : float
        Total thickness of the boundary layer
    bl_layers : int
        Number of boundary layer layers
    bl_growth : float
        Growth ratio for boundary layer
    """
    
    print(f"\n{'='*60}")
    print(f"Generating unstructured triangle mesh around cylinder")
    print(f"  Cylinder radius: {r_inner}")
    print(f"  Far-field radius: {r_outer}")
    print(f"  Circumferential points: {n_circum}")
    print(f"  Mesh size (inner): {mesh_inner}")
    print(f"  Mesh size (outer): {mesh_outer}")
    if boundary_layer:
        print(f"  Boundary layer: {bl_layers} layers, thickness={bl_thickness}, growth={bl_growth}")
    print(f"{'='*60}")
    
    gmsh.initialize()
    gmsh.model.add("CylinderTriangle")
    
    # --- Create geometry using OpenCASCADE for better mesh control ---
    # Inner circle (cylinder)
    inner_circle = gmsh.model.occ.addCircle(cx, cy, 0, r_inner)
    inner_loop = gmsh.model.occ.addCurveLoop([inner_circle])
    
    # Outer circle (far-field)
    outer_circle = gmsh.model.occ.addCircle(cx, cy, 0, r_outer)
    outer_loop = gmsh.model.occ.addCurveLoop([outer_circle])
    
    # Create annular domain (outer - inner)
    annulus = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])
    
    gmsh.model.occ.synchronize()
    
    # --- Mesh size control ---
    # Set mesh size at cylinder (inner boundary)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_outer)  # Default size
    
    # Get points on inner circle and set smaller mesh size
    inner_boundary = gmsh.model.getBoundary([(2, annulus)], oriented=False)
    for dim, tag in inner_boundary:
        points = gmsh.model.getBoundary([(dim, tag)], oriented=False)
        for pdim, ptag in points:
            gmsh.model.mesh.setSize([(pdim, ptag)], mesh_inner)
    
    # Use a distance field for smooth mesh size transition
    # Find the inner curve (smaller radius)
    curves = gmsh.model.getEntities(1)
    inner_curve_tag = None
    outer_curve_tag = None
    for dim, tag in curves:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        # Check if this is the inner or outer circle based on bounding box
        radius_approx = (bbox[3] - bbox[0]) / 2  # x-extent / 2
        if abs(radius_approx - r_inner) < 0.1:
            inner_curve_tag = tag
        elif abs(radius_approx - r_outer) < 1.0:
            outer_curve_tag = tag
    
    # Create distance field from inner boundary
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [inner_curve_tag])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    
    # Create threshold field for mesh size transition
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_inner)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_outer)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r_outer - r_inner)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # Set number of points on circumference
    gmsh.model.mesh.setTransfiniteCurve(inner_curve_tag, n_circum)
    
    # --- Boundary Layer (optional) ---
    if boundary_layer:
        # Add boundary layer field
        gmsh.model.mesh.field.add("BoundaryLayer", 3)
        gmsh.model.mesh.field.setNumbers(3, "CurvesList", [inner_curve_tag])
        gmsh.model.mesh.field.setNumber(3, "Size", mesh_inner * 0.5)  # First layer height
        gmsh.model.mesh.field.setNumber(3, "Ratio", bl_growth)
        gmsh.model.mesh.field.setNumber(3, "Thickness", bl_thickness)
        gmsh.model.mesh.field.setNumber(3, "Quads", 0)  # Use triangles in BL
        
        # Use the boundary layer as background mesh
        gmsh.option.setNumber("Mesh.BoundaryLayerFanElements", 0)
        gmsh.model.mesh.field.setAsBoundaryLayer(3)
    
    # --- Physical Groups for boundary conditions ---
    gmsh.model.addPhysicalGroup(1, [inner_curve_tag], tag=1, name="wall")
    gmsh.model.addPhysicalGroup(1, [outer_curve_tag], tag=2, name="farfield")
    gmsh.model.addPhysicalGroup(2, [annulus], tag=3, name="Fluid")
    
    # --- Mesh options ---
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
    gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Don't recombine to quads
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
        
        # --- Merge BAR elements for wall and farfield ---
        # Gmsh names BAR sections as "2_L_*" (2 = BAR_2 element type)
        bar_sections = sorted([n for n in zone.keys() if n.startswith("2_")])
        
        if bar_sections:
            # First bar section is wall (2_L_1), second is farfield (2_L_2)
            wall_sec = bar_sections[0] if len(bar_sections) > 0 else None
            farfield_sec = bar_sections[1] if len(bar_sections) > 1 else None
            
            # Rename wall section
            if wall_sec and wall_sec in zone:
                set_cgns_name(zone[wall_sec], "wall")
                zone.move(wall_sec, "wall")
            
            # Rename farfield section
            if farfield_sec and farfield_sec in zone:
                set_cgns_name(zone[farfield_sec], "farfield")
                zone.move(farfield_sec, "farfield")
            
            # Delete any remaining bar sections
            for sec_name in bar_sections[2:]:
                if sec_name in zone:
                    del zone[sec_name]
        
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

    N_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    print(f"\n{'='*70}")
    print(f"Generating unstructured triangle meshes with automatic mesh sizing")
    print(f"{'='*70}")
    print(f"{'N':>6} | {'mesh_inner':>12} | {'mesh_outer':>12} | {'est. layers':>12}")
    print(f"{'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    
    for n in N_values:
        # Automatically compute mesh sizes for good growth rate
        mesh_inner, mesh_outer = compute_mesh_sizes(
            n_circum=n, 
            r_inner=r_inner, 
            r_outer=r_outer, 
            target_growth_rate=1.15
        )
        
        # Estimate number of radial layers
        g = 1.15
        radial_length = r_outer - r_inner
        n_layers = math.log(1 + radial_length * (g - 1) / mesh_inner) / math.log(g)
        
        print(f"{n:>6} | {mesh_inner:>12.6f} | {mesh_outer:>12.4f} | {n_layers:>12.1f}")
        
        generate_unstructured_cylinder_mesh(
            output_file=f"cylinder_triangle_{n}.cgns",
            n_circum=n,
            mesh_inner=mesh_inner,
            mesh_outer=mesh_outer,
            boundary_layer=False
        )
    
    print(f"\n{'='*70}")
    print(f"All unstructured triangle meshes generated successfully!")
    print(f"{'='*70}")
    
    print(f"\n{'='*60}")
    print(f"All operations completed!")
    print(f"{'='*60}")
