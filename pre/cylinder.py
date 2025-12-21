import gmsh
import sys
import math
import h5py
import numpy as np

# --- Parameters ---
r_inner = 0.5       # Cylinder radius
r_outer = 100.0     # Outer boundary radius
cx, cy = 0.0, 0.0   # Cylinder center

# List of N values (number of cells, so N+1 points per direction)
N_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

def compute_growth_ratio(total_length, first_cell, n_cells, tol=1e-10, max_iter=100):
    """
    Compute the geometric growth ratio given:
    - total_length: total distance from inner to outer
    - first_cell: desired first cell height
    - n_cells: number of cells
    
    Uses Newton-Raphson to solve: L = h1 * (r^n - 1) / (r - 1)
    """
    if n_cells == 1:
        return 1.0
    
    # Initial guess based on approximate formula
    r = (total_length / first_cell / n_cells) ** (1.0 / (n_cells - 1))
    r = max(1.001, min(r, 5.0))  # Clamp initial guess
    
    for _ in range(max_iter):
        if abs(r - 1.0) < 1e-10:
            r = 1.001
        
        # f(r) = h1 * (r^n - 1) / (r - 1) - L = 0
        rn = r ** n_cells
        f = first_cell * (rn - 1) / (r - 1) - total_length
        
        # f'(r) = h1 * [(n*r^(n-1)*(r-1) - (r^n - 1)] / (r - 1)^2
        df = first_cell * (n_cells * (r ** (n_cells - 1)) * (r - 1) - (rn - 1)) / ((r - 1) ** 2)
        
        if abs(df) < 1e-15:
            break
            
        r_new = r - f / df
        
        if r_new <= 1.0:
            r_new = 1.0 + (r - 1.0) / 2  # Bisect toward 1
        
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
    
    return r

def set_cgns_name(node, name):
    """Set the CGNS node name attribute (32 chars, null-padded)"""
    name_bytes = name.encode('ascii')[:32].ljust(32, b'\x00')
    if 'name' in node.attrs:
        node.attrs['name'] = np.frombuffer(name_bytes, dtype='S1')
    else:
        node.attrs.create('name', np.frombuffer(name_bytes, dtype='S1'))

for N in N_values:
    N_circum = N      # Number of cells around the cylinder circumference
    N_radial = N      # Number of cells in the radial direction
    
    # Compute first cell height to match circumferential width (square cells at wall)
    # Circumferential width = (2 * pi * r_inner) / N_circum
    first_cell_height = (2 * math.pi * r_inner) / N_circum
    
    # Compute growth ratio to achieve desired first cell height
    total_radial_length = r_outer - r_inner
    growth_ratio = compute_growth_ratio(total_radial_length, first_cell_height, N_radial)
    
    # Verify first cell height
    actual_first_cell = total_radial_length * (growth_ratio - 1) / (growth_ratio ** N_radial - 1) if growth_ratio > 1.001 else total_radial_length / N_radial
    
    print(f"\n{'='*60}")
    print(f"Generating mesh: {N}x{N} cells ({N+1}x{N+1} points)")
    print(f"  First cell height: {actual_first_cell:.6f} (target: {first_cell_height:.6f})")
    print(f"  Growth ratio: {growth_ratio:.6f}")
    print(f"{'='*60}")
    
    gmsh.initialize()
    gmsh.model.add("Cylinder")

    # --- Geometry ---
    # Create cylinder (inner) boundary using 4 quarter arcs
    pc = gmsh.model.geo.addPoint(cx, cy, 0)

    # Inner circle points (at 0, 90, 180, 270 degrees)
    pi1 = gmsh.model.geo.addPoint(cx + r_inner, cy, 0)
    pi2 = gmsh.model.geo.addPoint(cx, cy + r_inner, 0)
    pi3 = gmsh.model.geo.addPoint(cx - r_inner, cy, 0)
    pi4 = gmsh.model.geo.addPoint(cx, cy - r_inner, 0)

    # Outer circle points
    po1 = gmsh.model.geo.addPoint(cx + r_outer, cy, 0)
    po2 = gmsh.model.geo.addPoint(cx, cy + r_outer, 0)
    po3 = gmsh.model.geo.addPoint(cx - r_outer, cy, 0)
    po4 = gmsh.model.geo.addPoint(cx, cy - r_outer, 0)

    # Inner circle arcs (counter-clockwise)
    c_in1 = gmsh.model.geo.addCircleArc(pi1, pc, pi2)
    c_in2 = gmsh.model.geo.addCircleArc(pi2, pc, pi3)
    c_in3 = gmsh.model.geo.addCircleArc(pi3, pc, pi4)
    c_in4 = gmsh.model.geo.addCircleArc(pi4, pc, pi1)

    # Outer circle arcs (counter-clockwise)
    c_out1 = gmsh.model.geo.addCircleArc(po1, pc, po2)
    c_out2 = gmsh.model.geo.addCircleArc(po2, pc, po3)
    c_out3 = gmsh.model.geo.addCircleArc(po3, pc, po4)
    c_out4 = gmsh.model.geo.addCircleArc(po4, pc, po1)

    # Radial lines connecting inner to outer
    l1 = gmsh.model.geo.addLine(pi1, po1)
    l2 = gmsh.model.geo.addLine(pi2, po2)
    l3 = gmsh.model.geo.addLine(pi3, po3)
    l4 = gmsh.model.geo.addLine(pi4, po4)

    # --- Create 4 quadrilateral surfaces (O-grid topology) ---
    loop1 = gmsh.model.geo.addCurveLoop([c_in1, l2, -c_out1, -l1])
    loop2 = gmsh.model.geo.addCurveLoop([c_in2, l3, -c_out2, -l2])
    loop3 = gmsh.model.geo.addCurveLoop([c_in3, l4, -c_out3, -l3])
    loop4 = gmsh.model.geo.addCurveLoop([c_in4, l1, -c_out4, -l4])

    s1 = gmsh.model.geo.addPlaneSurface([loop1])
    s2 = gmsh.model.geo.addPlaneSurface([loop2])
    s3 = gmsh.model.geo.addPlaneSurface([loop3])
    s4 = gmsh.model.geo.addPlaneSurface([loop4])

    gmsh.model.geo.synchronize()

    # --- Transfinite meshing for structured grid ---
    # Number of points on circumferential segments (quarter circle each)
    N_quarter = N_circum // 4 + 1
    N_radial_pts = N_radial + 1  # Number of points = cells + 1

    # Set transfinite curves - circumferential (inner and outer arcs)
    for c in [c_in1, c_in2, c_in3, c_in4, c_out1, c_out2, c_out3, c_out4]:
        gmsh.model.mesh.setTransfiniteCurve(c, N_quarter)

    # Set transfinite curves - radial lines with geometric progression (hyperbolic growth)
    for l in [l1, l2, l3, l4]:
        gmsh.model.mesh.setTransfiniteCurve(l, N_radial_pts, "Progression", growth_ratio)

    # Set transfinite surfaces
    for s in [s1, s2, s3, s4]:
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)  # Generate quads instead of triangles

    # --- Physical Groups ---
    gmsh.model.addPhysicalGroup(1, [c_in1, c_in2, c_in3, c_in4], name="wall")
    gmsh.model.addPhysicalGroup(1, [c_out1, c_out2, c_out3, c_out4], name="farfield")
    gmsh.model.addPhysicalGroup(2, [s1, s2, s3, s4], name="Fluid")

    # --- Generate and Export ---
    gmsh.model.mesh.generate(2)
    output_file = f"cylinder_{N}x{N}.cgns"
    gmsh.write(output_file)

    # --- Mesh Statistics ---
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
    print(f"  - Grid size: {N_circum}x{N_radial} cells")
    print(f"  - Number of nodes: {len(node_tags)}")
    print(f"  - Number of 2D elements (quads): {sum(len(t) for t in elem_tags)}")
    print(f"  - Output file: {output_file}")

    gmsh.finalize()

    # --- Post-process CGNS file using h5py ---
    print("  Post-processing CGNS file...")

    with h5py.File(output_file, "r+") as f:
        # Find the base node (Gmsh names it after the output file)
        base_name = None
        for name in f.keys():
            if name not in ['CGNSLibraryVersion', ' format', ' hdf5version']:
                base_name = name
                break
        
        if base_name is None:
            print("Error: Could not find base node")
            continue
            
        base = f[base_name]
        
        # Find zone name (Gmsh creates "Cylinder_Part0")
        zone_old_name = None
        for name in base.keys():
            if "Part" in name:
                zone_old_name = name
                break
        
        if zone_old_name is None:
            print("Error: Could not find zone")
            continue
        
        zone = base[zone_old_name]
        
        # --- Merge QUAD elements into single section "QuadElements" ---
        quad_sections = sorted([n for n in zone.keys() if n.startswith("4_")])
        if quad_sections:
            all_quad_conn = []
            for sec_name in quad_sections:
                sec = zone[sec_name]
                conn_data = sec["ElementConnectivity/ data"][:]
                all_quad_conn.append(conn_data)
            
            merged_quad_conn = np.concatenate(all_quad_conn)
            n_quads = len(merged_quad_conn) // 4
            first_start = zone[quad_sections[0]]["ElementRange/ data"][0]
            first_sec = quad_sections[0]
            
            zone[first_sec]["ElementRange/ data"][...] = np.array([first_start, first_start + n_quads - 1], dtype=np.int32)
            del zone[first_sec]["ElementConnectivity/ data"]
            zone[first_sec]["ElementConnectivity"].create_dataset(" data", data=merged_quad_conn.astype(np.int32))
            set_cgns_name(zone[first_sec], "QuadElements")
            zone.move(first_sec, "QuadElements")
            
            for sec_name in quad_sections[1:]:
                del zone[sec_name]
        
        # --- Merge BAR elements ---
        bar_sections = sorted([n for n in zone.keys() if n.startswith("2_")])
        n_bar_sections = len(bar_sections)
        
        if bar_sections:
            wall_bar_sections = bar_sections[:n_bar_sections//2]
            farfield_bar_sections = bar_sections[n_bar_sections//2:]
            
            # Merge Wall elements
            all_wall_conn = []
            wall_start = None
            for sec_name in wall_bar_sections:
                sec = zone[sec_name]
                conn_data = sec["ElementConnectivity/ data"][:]
                all_wall_conn.append(conn_data)
                if wall_start is None:
                    wall_start = sec["ElementRange/ data"][0]
            
            if all_wall_conn:
                merged_wall_conn = np.concatenate(all_wall_conn)
                n_wall_edges = len(merged_wall_conn) // 2
                first_wall = wall_bar_sections[0]
                zone[first_wall]["ElementRange/ data"][...] = np.array([wall_start, wall_start + n_wall_edges - 1], dtype=np.int32)
                del zone[first_wall]["ElementConnectivity/ data"]
                zone[first_wall]["ElementConnectivity"].create_dataset(" data", data=merged_wall_conn.astype(np.int32))
                set_cgns_name(zone[first_wall], "wall")
                zone.move(first_wall, "wall")
                
                for sec_name in wall_bar_sections[1:]:
                    del zone[sec_name]
            
            # Merge Farfield elements
            all_ff_conn = []
            ff_start = None
            for sec_name in farfield_bar_sections:
                sec = zone[sec_name]
                conn_data = sec["ElementConnectivity/ data"][:]
                all_ff_conn.append(conn_data)
                if ff_start is None:
                    ff_start = sec["ElementRange/ data"][0]
            
            if all_ff_conn:
                merged_ff_conn = np.concatenate(all_ff_conn)
                n_ff_edges = len(merged_ff_conn) // 2
                first_ff = farfield_bar_sections[0]
                zone[first_ff]["ElementRange/ data"][...] = np.array([ff_start, ff_start + n_ff_edges - 1], dtype=np.int32)
                del zone[first_ff]["ElementConnectivity/ data"]
                zone[first_ff]["ElementConnectivity"].create_dataset(" data", data=merged_ff_conn.astype(np.int32))
                set_cgns_name(zone[first_ff], "farfield")
                zone.move(first_ff, "farfield")
                
                for sec_name in farfield_bar_sections[1:]:
                    del zone[sec_name]
        
        # Delete ZoneBC
        if "ZoneBC" in zone:
            del zone["ZoneBC"]
        
        # Rename Zone to "dom-1"
        set_cgns_name(zone, "dom-1")
        base.move(zone_old_name, "dom-1")
        
        # Clean up old Family nodes
        families_to_delete = [n for n in base.keys() if n.startswith("L_") or n.startswith("S_")]
        for fam in families_to_delete:
            del base[fam]
        
        # Rename Base node
        set_cgns_name(base, "Base")
        f.move(base_name, "Base")
    
    print(f"  Done: {output_file}")

print(f"\n{'='*60}")
print(f"All meshes generated successfully!")
print(f"{'='*60}")