module readMesh
{
// This module reads a mesh from an HDF5 file and provides various utilities
use HDF5;
use HDF5.C_HDF5;
use IO;
use elementTopology;
use List;
use Map;

class MeshData {
  // === Metadata ===
  var filename_: string;
  var elementType_: string;
  var nnode_: int; // number of nodes per element
  var nfael_: int; // number of faces per element
  var nfaelInv_ : real(64); // 1/nfael_
  var nedel_: int; // number of edges per element

  // === Topology Info ===
  var lnofa_dom: domain(1) = {1..0}; // domain for lnofa
  var lpofa_dom: domain(2) = {1..0, 1..0}; // domain for lpofa
  var lpoed_dom: domain(2) = {1..0, 1..0}; // domain for lpoed

  var lnofa_: [lnofa_dom] int; // node per face
  var lpofa_: [lpofa_dom] int; // node to face connectivity
  var lpoed_: [lpoed_dom] int; // node to edge connectivity

  // === Connectivity ===
  var X_dom: domain(1, idxType=uint) = {1..0}; // empty domain, can resize
  var element2node_dom: domain(1, idxType=uint) = {1..0}; // empty domain, can resize
  var farfieldElement2node_dom: domain(1, idxType=uint) = {1..0}; // empty domain, can resize
  var wallElement2node_dom: domain(1, idxType=uint) = {1..0}; // empty domain, can resize
  var elem2node_dom: domain(2) = {1..0, 1..0}; // empty domain, can resize
  var edge2node_dom: domain(2) = {1..0, 1..0}; // empty domain, can resize
  var elem2edge_dom: domain(2) = {1..0, 1..0}; // empty domain, can resize
  var edgeWall_dom: domain(1) = {1..0}; // empty domain, can resize
  var edgeFarfield_dom: domain(1) = {1..0}; // empty domain, can resize
  var edge2elem_dom: domain(2) = {1..0, 1..0}; // empty domain, can resize
  var esup1_dom: domain(1) = {1..0}; // empty domain, can resize
  var esup2_dom: domain(1) = {1..0}; // empty domain, can resize
  var psup1_dom: domain(1) = {1..0}; // empty domain, can resize
  var psup2_dom: domain(1) = {1..0}; // empty domain, can resize
  var elemsAroundElem_dom: domain(2) = {1..0, 1..0}; // empty domain, can resize

  // === Coordinates ===
  var X_: [X_dom] real(64);
  var Y_: [X_dom] real(64);
  var Z_: [X_dom] real(64);

  var element2node_: [element2node_dom] int;
  var farfieldElement2node_: [farfieldElement2node_dom] int;
  var wallElement2node_: [wallElement2node_dom] int;
  var elem2node_: [elem2node_dom] int;
  var edge2node_: [edge2node_dom] int;
  var elem2edge_: [elem2edge_dom] int;
  var edgeWall_: [edgeWall_dom] int;
  var edgeFarfield_: [edgeFarfield_dom] int;
  var edge2elem_: [edge2elem_dom] int;
  var esup1_: [esup1_dom] int;
  var esup2_: [esup2_dom] int;
  var psup1_: [psup1_dom] int;
  var psup2_: [psup2_dom] int;
  var elemsAroundElem_: [elemsAroundElem_dom] int;
  var elemToLocalEdgeIndex_: map(int, map(int, int));

  var nelem_: int; // number of elements

  // // === Ghost Cell Mesh ===
  // var XGhost_dom: domain(1) = {1..0}; // empty domain, can resize
  // var elem2nodeGhost_dom: domain(2) = {1..0, 1..0}; // empty domain, can resize

  // var X_: [XGhost_dom] real(64);
  // var Y_: [XGhost_dom] real(64);
  // var Z_: [XGhost_dom] real(64);
  // var elem2node_: [elem2nodeGhost_dom] int;

  // === Constructor ===
  proc init(filename: string, elementType: string) {
    this.filename_ = filename;
    this.elementType_ = elementType;

    // === Get element topology (if Quad) ===
    if this.elementType_ == "QuadElements" {
      var (lnofa, lpofa, lpoed, nnode, nfael, nedel) = quadElementTopology_2d();
      this.nnode_ = nnode;
      this.nfael_ = nfael;
      this.nfaelInv_ = 1 / nfael:real(64);
      this.nedel_ = nedel;

      this.lnofa_dom = lnofa.domain;
      this.lpofa_dom = lpofa.domain;
      this.lpoed_dom = lpoed.domain;

      this.lnofa_ = lnofa;
      this.lpofa_ = lpofa;
      this.lpoed_ = lpoed;
    }
    else if this.elementType_ == "TriElements" {
      var (lnofa, lpofa, lpoed, nnode, nfael, nedel) = triElementTopology_2d();
      this.nnode_ = nnode;
      this.nfael_ = nfael;
      this.nfaelInv_ = 1 / nfael:real(64);
      this.nedel_ = nedel;

      this.lnofa_dom = lnofa.domain;
      this.lpofa_dom = lpofa.domain;
      this.lpoed_dom = lpoed.domain;

      this.lnofa_ = lnofa;
      this.lpofa_ = lpofa;
      this.lpoed_ = lpoed;
    }
    else if this.elementType_ == "Elements_NGON_n" {
      writeln("Using NGON elements");
    }

    // === Read raw mesh ===
    var (Xtmp, Ytmp, Ztmp, element2nodeTmp, farfieldTmp, wallTmp) = readMesh(filename, this.elementType_);

    // === Store coordinates and connectivity ===
    this.X_dom = Xtmp.domain;
    this.element2node_dom = element2nodeTmp.domain;
    this.farfieldElement2node_dom = farfieldTmp.domain;
    this.wallElement2node_dom = wallTmp.domain;

    this.X_ = Xtmp;
    this.Y_ = Ytmp;
    this.Z_ = Ztmp;
    this.element2node_ = element2nodeTmp;
    this.farfieldElement2node_ = farfieldTmp;
    this.wallElement2node_ = wallTmp;

    // === Compute number of elements ===
    this.nelem_ = this.element2node_.size / this.nnode_;
  }

  proc buildConnectivity() {
    var elem2node = elementToNodes(this.element2node_, this.nnode_, this.nelem_);
    var edge2node = buildEdgesFromElements(elem2node);
    var elem2edge = buildInedel(elem2node, edge2node);
    var edgeWall = boundaryFaces(this.wallElement2node_, edge2node, this.lnofa_);
    var edgeFarfield = boundaryFaces(this.farfieldElement2node_, edge2node, this.lnofa_);
    var edge2elem = leftRightFromFace(edge2node, elem2node);
    var (esup1, esup2) = elementsAroundNode(elem2node, this.X_.size, this.nnode_);
    var (psup1, psup2) = nodesAroundNode(esup1, esup2, elem2node, this.X_.size, this.nnode_);

    this.elem2node_dom = elem2node.domain;
    this.edge2node_dom = edge2node.domain;
    this.elem2edge_dom = elem2edge.domain;
    this.edgeWall_dom = edgeWall.domain;
    this.edgeFarfield_dom = edgeFarfield.domain;
    this.edge2elem_dom = edge2elem.domain;
    this.esup1_dom = esup1.domain;
    this.esup2_dom = esup2.domain;
    this.psup1_dom = psup1.domain;
    this.psup2_dom = psup2.domain;

    this.elem2node_ = elem2node;
    this.edge2node_ = edge2node;
    this.elem2edge_ = elem2edge;
    this.edgeWall_ = edgeWall;
    this.edgeFarfield_ = edgeFarfield;
    this.edge2elem_ = edge2elem;
    this.esup1_ = esup1;
    this.esup2_ = esup2;
    this.psup1_ = psup1;
    this.psup2_ = psup2;

    for elem in 1..this.nelem_ {
      var globalIndexToLocalIndex = new map(int, int);
      for edge in 1..this.nedel_ {
        globalIndexToLocalIndex[elem2edge[edge, elem]] = edge;
      }
      this.elemToLocalEdgeIndex_[elem] = globalIndexToLocalIndex;
    }
  }

  proc buildConnectivityWithGhostCells() {
    var (xWall, yWall, elem2nodeGhostWall) = updateConnectivityWithGhostCells(this.X_, this.Y_, this.elem2node_, this.edge2elem_, this.edge2node_, this.edgeWall_, this.elementType_);
    var (xFarfield, yFarfield, elem2nodeGhostFarfield) = updateConnectivityWithGhostCells(xWall, yWall, elem2nodeGhostWall, this.edge2elem_, this.edge2node_, this.edgeFarfield_, this.elementType_);
    var (esup1, esup2) = elementsAroundNode(elem2nodeGhostFarfield, xFarfield.size, this.nnode_);
    var elemAroundElem = elemsAroundElem(esup1, esup2, elem2nodeGhostFarfield, this.lnofa_, this.lpofa_, xFarfield.size, this.nfael_);

    // this.XGhost_dom = xFarfield.domain;
    // this.elem2nodeGhost_dom = elem2nodeGhostFarfield.domain;
    this.elemsAroundElem_dom = elemAroundElem.domain;

    // this.X_ = xFarfield;
    // this.Y_ = yFarfield;
    // this.elem2node_ = elem2nodeGhostFarfield;
    this.elemsAroundElem_ = elemAroundElem;
  }
}

// === Generic 1D reader (for real(64) or int) ===
proc read1DDataset(type eltType, file_id: hid_t, name: string): [] eltType {
  var dset_id = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
  if dset_id < 0 then halt("Dataset not found: ", name);

  var space_id = H5Dget_space(dset_id);
  var dims: [0..<1] uint(64);
  H5Sget_simple_extent_dims(space_id, c_ptrTo(dims), nil);

  var dom = {1..dims[0]};
  var data: [dom] eltType;
  readHDF5Dataset(file_id, name, data);

  H5Sclose(space_id);
  H5Dclose(dset_id);
  return data;
}

proc readMesh(filename: string, elementType: string) {
  // === Dataset paths ===
  const dsetX = "/Base/dom-1/GridCoordinates/CoordinateX/ data";
  const dsetY = "/Base/dom-1/GridCoordinates/CoordinateY/ data";
  const dsetZ = "/Base/dom-1/GridCoordinates/CoordinateZ/ data";

  const dsetElem     = "/Base/dom-1/" + elementType + "/ElementConnectivity/ data";
  const dsetFarfield = "/Base/dom-1/farfield/ElementConnectivity/ data";
  const dsetWall     = "/Base/dom-1/wall/ElementConnectivity/ data";

  var file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if file_id < 0 then
      halt("Could not open file: ", filename);

  // Read 1D coordinate arrays (type: real(64))
  var X = read1DDataset(real(64), file_id, dsetX);
  var Y = read1DDataset(real(64), file_id, dsetY);
  var Z = read1DDataset(real(64), file_id, dsetZ);

  // Read connectivity arrays
  var element2node = read1DDataset(int, file_id, dsetElem);
  var farfieldElement2node = read1DDataset(int, file_id, dsetFarfield);
  var wallElement2node = read1DDataset(int, file_id, dsetWall);

  H5Fclose(file_id);

  return (X, Y, Z, element2node, farfieldElement2node, wallElement2node);
}

proc elementToNodes(const element2node: [] int, const nnode: int, const nelem: int) {
  var inpoel: [1..nnode, 1..nelem] int;

  for i in 1..nelem {
    for j in 1..nnode {
      inpoel[j,i] = element2node[j + (i-1)*nnode];
    }
  }

  return inpoel;
}

proc elementsAroundNode(const inpoel: [] int, const npoin: int, const nnode:int) {
  var nelem = inpoel.dim(1).size;
  var esup2 : [1..npoin+1] int;
  var ipoi1, ipoin, istor: int;

  for i in 1..nelem {
    for j in 1..nnode {
      ipoi1 = inpoel[j,i] + 1;
      esup2[ipoi1] = esup2[ipoi1] + 1;
    }
  }

  for i in 2..npoin+1 {
    esup2[i] = esup2[i] + esup2[i-1];
  }
  
  var esup1 : [1..esup2[npoin+1]] int;
  for i in 1..nelem {
    for j in 1..nnode {
      ipoin = inpoel[j,i];
      istor = esup2[ipoin] + 1;
      esup2[ipoin] = istor;
      esup1[istor] = i;
    }
  }

  for i in 2..npoin+1 by -1 {
    esup2[i] = esup2[i-1];
  }
  esup2[1] = 0;

  return (esup1, esup2);
}

proc nodesAroundNode(const esup1: [] int, const esup2: [] int, const inpoel: [] int, const npoin: int, const nnode: int) {
  var lpoin : [1..npoin] int;
  var psup2 : [1..npoin+1] int;
  var istor, ielem, jpoin : int;

  for ipoin in 1..npoin {
    for iesup in (esup2[ipoin]+1)..esup2[ipoin+1] {
      ielem = esup1[iesup];
      for inode in 1..nnode {
        jpoin = inpoel[inode, ielem];
        if jpoin != ipoin && lpoin[jpoin] != ipoin {
          istor = istor + 1;
          lpoin[jpoin] = ipoin;
        }
      }
    }
    psup2[ipoin+1] = istor;
  }

  var psup1 : [1..psup2[npoin+1]] int;
  istor = 0;
  for ipoin in 1..npoin {
    for iesup in (esup2[ipoin]+1)..esup2[ipoin+1] {
      ielem = esup1[iesup];
      for inode in 1..nnode {
        jpoin = inpoel[inode, ielem];
        if jpoin != ipoin && lpoin[jpoin] != ipoin {
          istor = istor + 1;
          psup1[istor] = jpoin;
          lpoin[jpoin] = ipoin;
        }
      }
    }
  }

  return (psup1, psup2);
}

proc elemsAroundElem(const esup1: [] int, const esup2: [] int, const inpoel: [] int, const lnofa: [] int, const lpofa: [] int, const npoin: int, const nfael: int) {
  var nelem = inpoel.dim(1).size;
  var lpoin : [1..npoin] int;
  var esuel : [1..nfael, 1..nelem] int;
  var nnofa, ipoin, jelem, nnofj, icoun, jpoin: int;

  for ielem in 1..nelem {
    for ifael in 1..nfael {
      nnofa = lnofa[ifael];
      var lhelp = inpoel[lpofa[1..nnofa, ifael], ielem];
      lpoin[lhelp[1..nnofa]] = 1;
      ipoin = lhelp[1];

      for istor in (esup2[ipoin]+1)..esup2[ipoin+1] {
        jelem = esup1[istor];
        if jelem != ielem {
          for jfael in 1..nfael {
            nnofj = lnofa[jfael];
            if nnofj == nnofa {
              icoun = 0;
              for jnofa in 1..nnofa {
                jpoin = inpoel[lpofa[jnofa, jfael], jelem];
                icoun = icoun + lpoin[jpoin];
              }
              if icoun == nnofa {
                esuel[ifael, ielem] = jelem;
              }
            }
          }
        }
      }
      lpoin[lhelp[1..nnofa]] = 0;
    }
  }

  return esuel;
}

proc edgedToNodes(const psup1: [] int, const psup2: [] int, const npoin: int) {
  var nedge : int = 0;
  var ip, jp : int;

  for ip in 1..npoin {
    for j in psup2[ip]+1 .. psup2[ip+1] {
      jp = psup1[j];
      if ip < jp {
        nedge += 1;
      }
    }
  }

  var inpoed : [1..2, 1..nedge] int;
  nedge = 0;
  for ip in 1..npoin {
    for j in (psup2[ip]+1)..psup2[ip+1] {
      jp = psup1[j];
      if ip < jp {
        nedge = nedge + 1;
        inpoed[1, nedge] = ip;
        inpoed[2, nedge] = jp;
      }
    }
    
  }

  var inpoe1: [1..npoin+1] int;
  inpoe1 = 0;

  for iedge in 1..nedge {
    ip = inpoed[1, iedge];
    inpoe1[ip+1] += 1;
  }

  for ip in 2..npoin+1 {
    inpoe1[ip] += inpoe1[ip-1];
  }

  return (inpoed, inpoe1);
}

proc buildEdgesFromElements(const inpoel: [] int) {
  const nnode = inpoel.dim(0).size;
  const nelem = inpoel.dim(1).size;

  var edgeSet = new map(string, (int, int)); // Use string keys to deduplicate

  for e in 1..nelem {
    for i in 1..nnode {
      const i1 = inpoel[i, e];
      const i2 = inpoel[(i % nnode) + 1, e]; // wrap around for last node

      const ip = min(i1, i2);
      const jp = max(i1, i2);
      const key = try! "%i_%i".format(ip, jp);

      edgeSet.add(key, (ip, jp));
    }
  }

  const nedge = edgeSet.size;
  var inpoed: [1..2, 1..nedge] int;
  var idx = 1;

  for (ip, jp) in edgeSet.values() {
    inpoed[1, idx] = ip;
    inpoed[2, idx] = jp;
    idx += 1;
  }

  return inpoed;
}

proc buildInedel(const inpoel: [] int, const inpoed: [] int) {
  const nedel = inpoel.dim(0).size;
  const nelem = inpoel.dim(1).size;
  const nedge = inpoed.dim(1).size;

  var edgeMap = new map((int, int), int); // maps (min, max) -> edge ID

  // Step 1: Build edgeMap from inpoed
  for iedge in 1..nedge {
    const ip = min(inpoed[1, iedge], inpoed[2, iedge]);
    const jp = max(inpoed[1, iedge], inpoed[2, iedge]);
    edgeMap[(ip, jp)] = iedge;
  }

  // Step 2: Allocate inedel
  var inedel: [1..nedel, 1..nelem] int;

  // Step 3: For each element, map its edges to global edge IDs
  for e in 1..nelem {
    for k in 1..nedel {
      const i1 = inpoel[k, e];
      const i2 = inpoel[(k % nedel) + 1, e]; // wrap around
      const ip = min(i1, i2);
      const jp = max(i1, i2);
      inedel[k, e] = try! edgeMap[(ip, jp)];
    }
  }

  return inedel;
}

proc edgeOfElemLocalIndex(const inedel) {
  const nelem = inedel.dim(1).size;
  const nedel = inedel.dim(0).size;
  var elemToLocalEdgeIndex = new map(int, map(int, int));

  for elem in 1..nelem {
    var globalIndexToLocalIndex = new map(int, int);
    for edge in 1..nedel {
      globalIndexToLocalIndex[inedel[edge, elem]] = edge;
    }
    elemToLocalEdgeIndex[elem] = globalIndexToLocalIndex;
  }

  return elemToLocalEdgeIndex;
}

proc edgedOfElem(const inpoel: [] int, const inpoed: [] int, const inpoe1: [] int, const lpoed: [] int, const nedel: int, const nelem: int) {
  var inedel: [1..nedel, 1..nelem] int;
  var ipoi1, ipoi2, ipmin, ipmax : int;

  for ielem in 1..nelem {
    for iedel in 1..nedel {
      ipoi1 = inpoel[lpoed[1, iedel], ielem];
      ipoi2 = inpoel[lpoed[2, iedel], ielem];
      ipmin = min(ipoi1, ipoi2);
      ipmax = max(ipoi1, ipoi2);
      for iedge in (inpoe1[ipmin]+1)..inpoe1[ipmin+1] {
        if inpoed[2, iedge] == ipmax {
          inedel[iedel, ielem] = iedge;
        }
      }
    }
  }

  return inedel;
}

proc boundaryFaces(const boundaryElement2node: [] int, const inpoed: [] int, const lnofa: [] int) {
  
  const nface = boundaryElement2node.size / lnofa[1];
  const nedge = inpoed.dim(1).size;
  var bface: [1..nface] int;
  var ipoi1, ipoi2, ipmin, ipmax : int;

  for edge in 1..nedge {
    for boundaryEdge in 1..nface {
      ipoi1 = boundaryElement2node[2*boundaryEdge-1];
      ipoi2 = boundaryElement2node[2*boundaryEdge];
      ipmin = min(ipoi1, ipoi2);
      ipmax = max(ipoi1, ipoi2);

      if ipmin == inpoed[1, edge] && ipmax == inpoed[2, edge] {
        bface[boundaryEdge] = edge;
      }
    }
  }

  return bface;      
}

proc externalFaces(const boundaryElement2node: [] int, const inpoel: [] int, const lnofa: [] int, const lpofa: [] int, const npoin: int, const nelem: int, const nfael: int) {
  var lpoin: [1..npoin] int;
  var nface: int = 0;
  var nnofa, icoun: int;

  for node in boundaryElement2node {
    lpoin[node] = 1;
  }

  for ielem in 1..nelem {
    for ifael in 1..nfael {
      nnofa = lnofa[ifael];
      var lhelp = inpoel[lpofa[1..nnofa, ifael], ielem];
      icoun = 0;
      for inofa in 1..nnofa {
        icoun = icoun + lpoin[lhelp[inofa]];
      }
      if icoun == nnofa {
        nface = nface + 1;
      }
    }
  }

  var bface: [1..lnofa[1], 1..nface] int;

  nface = 0;
  for ielem in 1..nelem {
    for ifael in 1..nfael {
      nnofa = lnofa[ifael];
      var lhelp = inpoel[lpofa[1..nnofa, ifael], ielem];
      icoun = 0;
      for inofa in 1..nnofa {
        icoun = icoun + lpoin[lhelp[inofa]];
      }
      if icoun == nnofa {
        nface = nface + 1;
        bface[1..nnofa, nface] = lhelp[1..nnofa];
      }
    }
  }

  return bface;      
}

proc leftRightFromFace(const inpoed: [] int, const inpoel: [] int) {
  const nedge = inpoed.dim(1).size;
  const nnodePerElem = inpoel.dim(0).size;
  const nelem = inpoel.dim(1).size;

  // Map edge (ip,jp) => edge index
  var edgeMap = new map((int, int), int);

  for e in 1..nedge {
    const ip = inpoed[1, e];
    const jp = inpoed[2, e];
    edgeMap[(ip, jp)] = e;
  }

  // Prepare edge-to-element map: max 2 elems per edge
  var edge2elem: [1..2, 1..nedge] int = 0;

  for elem in 1..nelem {
    for i in 1..nnodePerElem {
      const ip = inpoel[i, elem];
      const jp = inpoel[(i % nnodePerElem) + 1, elem]; // wrap around

      const ep = min(ip, jp);
      const eq = max(ip, jp);
      const edgeKey = (ep, eq);

      if edgeMap.contains(edgeKey) {
        const eIdx = try! edgeMap[edgeKey];
        if edge2elem[1, eIdx] == 0 then
          edge2elem[1, eIdx] = elem;
        else
          edge2elem[2, eIdx] = elem;
      } else {
        writeln("Edge not found for element ", elem, " edge (", ep, ",", eq, ")");
      }
    }
  }

  return edge2elem;
}

proc generateGhostCells(const X : [] real(64), const Y: [] real(64),  const elem2node: [] int, ref edge2elem: [] int, const edge2node: [] int, const edgeBoundary: [] int, elementType: string) {
  const nedgeboundary = edgeBoundary.dim(0).size;
  const nedge = edge2elem.dim(1).size;
  const nnodePerElem = elem2node.dim(0).size;
  const nnodePerEdge = edge2elem.dim(0).size;

  var ghostIndex = 1;
  var ghostNode = X.dim(0).size + 1;
  var XGhostIndex = 1;
  var ghostElemIndex = elem2node.dim(1).size + 1;

  var ghostElem2node: [1..nnodePerElem, 1..nedgeboundary] int;
  var XGhost: [1..(nedgeboundary*nnodePerEdge)] real(64);
  var YGhost: [1..(nedgeboundary*nnodePerEdge)] real(64);
  for edge in edgeBoundary {
    const e1 = edge2elem[1, edge];
    const e2 = edge2elem[2, edge];

    var realElem : int;
    if e1 != 0 {
        realElem = e1;
        edge2elem[2, edge] = ghostElemIndex;
    }
    else {
        realElem = e2;
        edge2elem[1, edge] = ghostElemIndex;
    }

    ghostElemIndex += 1;
    

    const node1 = edge2node[1, edge];
    const node2 = edge2node[2, edge];

    const x1 = X[node1];
    const y1 = Y[node1];
    const x2 = X[node2];
    const y2 = Y[node2];

    var A = y2-y1;
    var B = -(x2-x1);
    var C = -A*x1 - B*y1;
    
    const M = sqrt(A*A + B*B);

    A = A/M;
    B = B/M;
    C = C/M;

    for point in 1..nnodePerElem {
      const p = elem2node[point, realElem];
      const pp1 = elem2node[(point % nnodePerElem)+1, realElem];
      const pp2 = elem2node[((point + 1) % nnodePerElem) + 1, realElem];
      if elementType == "QuadElements" {
        if p != node1 && p != node2 && pp1 != node1 && pp1 != node2 {
          var x = X[pp1];
          var y = Y[pp1];

          var D = A*x + B*y + C;

          var x_mirror = x - 2*A*D;
          var y_mirror = y - 2*B*D;

          if pp2 == node1 {
          ghostElem2node[1, ghostIndex] = node2;
          ghostElem2node[2, ghostIndex] = node1;
          }
          else if pp2 == node2 {
              ghostElem2node[1, ghostIndex] = node1;
              ghostElem2node[2, ghostIndex] = node2;
          }

          ghostElem2node[3, ghostIndex] = ghostNode;
          ghostElem2node[4, ghostIndex] = ghostNode+1;

          XGhost[XGhostIndex] = x_mirror;
          YGhost[XGhostIndex] = y_mirror;

          x = X[p];
          y = Y[p];

          D = A*x + B*y + C;

          x_mirror = x - 2*A*D;
          y_mirror = y - 2*B*D;

          XGhost[XGhostIndex+1] = x_mirror;
          YGhost[XGhostIndex+1] = y_mirror;

          ghostNode += 2;
          ghostIndex += 1;
          XGhostIndex += 2;
        }
      }
      else if elementType == "TriElements" {
        if p != node1 && p != node2 {
          var x = X[p];
          var y = Y[p];

          var D = A*x + B*y + C;

          var x_mirror = x - 2*A*D;
          var y_mirror = y - 2*B*D;

          if pp2 == node1 {
            ghostElem2node[1, ghostIndex] = node2;
            ghostElem2node[2, ghostIndex] = node1;
          }
          else if pp2 == node2 {
            ghostElem2node[1, ghostIndex] = node1;
            ghostElem2node[2, ghostIndex] = node2;
          }
          ghostElem2node[3, ghostIndex] = ghostNode;

          XGhost[XGhostIndex] = x_mirror;
          YGhost[XGhostIndex] = y_mirror;

          ghostNode += 1;
          ghostIndex += 1;
          XGhostIndex += 1;
        }
      }          
    }
  }

  return (ghostElem2node, XGhost, YGhost);
}

proc updateConnectivityWithGhostCells(const X : [] real(64), const Y: [] real(64),  const elem2node: [] int, ref edge2elem: [] int, const edge2node: [] int, const edgeBoundary: [] int, elementType: string) {
  var (ghostBoundaryElem2node, XGhostBoundary, YGhostBoundary) = generateGhostCells(X, Y, elem2node, edge2elem, edge2node, edgeBoundary, elementType);
  const npoin = X.dim(0).size+XGhostBoundary.dim(0).size; 
  const nelem = elem2node.dim(1).size+ghostBoundaryElem2node.dim(1).size;
  const nnodePerElem = elem2node.dim(0);
  var X_: [1..npoin] real(64);
  var Y_: [1..npoin] real(64);
  var elem2node_: [elem2node.dim(0), 1..nelem] int;

  for i in 1..X.dim(0).size {
      X_[i] = X[i];
      Y_[i] = Y[i];
  }
  for i in (X.dim(0).size+1)..npoin {
      X_[i] = XGhostBoundary[i-(X.dim(0).size)];
      Y_[i] = YGhostBoundary[i-(X.dim(0).size)];
  }

  for i in elem2node.dim(1) {
      elem2node_[nnodePerElem, i] = elem2node[nnodePerElem, i];
  }

  for i in (elem2node.dim(1).size+1)..nelem {
      elem2node_[nnodePerElem, i] = ghostBoundaryElem2node[nnodePerElem, i-elem2node.dim(1).size];
  }

  return (X_, Y_, elem2node_);

}


proc info_on_element(edge_Id: int, edge2elem: [] int, elem2node: [] int, edge2node: [] int, X: [] real(64), Y: [] real(64)) {
  writeln("edge ", edge_Id);
  writeln("points ", edge2node[edge2node.dim(0), edge_Id]);
  writeln("elem neighbour : ", edge2elem[edge2elem.dim(0), edge_Id]);
  writeln("elem ", edge2elem[edge2elem.dim(0), edge_Id][1], " points : ", elem2node[elem2node.dim(0), edge2elem[edge2elem.dim(0), edge_Id][1]]);
  for point in elem2node[elem2node.dim(0), edge2elem[edge2elem.dim(0), edge_Id][1]] {
    writeln("point ", point, " = ", X[point], ", ", Y[point]);
  }
  writeln("elem ", edge2elem[edge2elem.dim(0), edge_Id][2], " points : ", elem2node[elem2node.dim(0), edge2elem[edge2elem.dim(0), edge_Id][2]]);
  for point in elem2node[elem2node.dim(0), edge2elem[edge2elem.dim(0), edge_Id][2]] {
    writeln("point ", point, " = ", X[point], ", ", Y[point]);
  }
}

  
}
