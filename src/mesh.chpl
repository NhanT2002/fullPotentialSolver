module mesh
{
use List;
use Map;
use IO;
use HDF5;
use HDF5.C_HDF5;
use Set;

config const elem : int = 0; // Element index for debugging purposes

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

proc readSolution(filename) {
    const dsetxElem = "/Base/dom-1/FLOW_SOLUTION_CC/xElem/ data";
    const dsetyElem = "/Base/dom-1/FLOW_SOLUTION_CC/yElem/ data";
    const dsetRho = "/Base/dom-1/FLOW_SOLUTION_CC/rho/ data";
    const dsetPhi = "/Base/dom-1/FLOW_SOLUTION_CC/phi/ data";
    const dsetIt = "/Base/GlobalConvergenceHistory/IterationCounters/ data";
    const dsetTime = "/Base/GlobalConvergenceHistory/Time/ data";
    const dsetRes = "/Base/GlobalConvergenceHistory/Residual/ data";
    const dsetCl = "/Base/GlobalConvergenceHistory/Cl/ data";
    const dsetCd = "/Base/GlobalConvergenceHistory/Cd/ data";
    const dsetCm = "/Base/GlobalConvergenceHistory/Cm/ data";
    const dsetCirculation = "/Base/GlobalConvergenceHistory/Circulation/ data";
    const dsetWakeGamma = "/WakeBase/wake/WAKE_FLOW_SOLUTION_NC/gammaWake/ data";

    var file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if file_id < 0 then
        halt("Could not open file: ", filename);

    var xElem = read1DDataset(real(64), file_id, dsetxElem);
    var yElem = read1DDataset(real(64), file_id, dsetyElem);
    var rho = read1DDataset(real(64), file_id, dsetRho);
    var phi = read1DDataset(real(64), file_id, dsetPhi);
    var it = read1DDataset(int, file_id, dsetIt);
    var time = read1DDataset(real(64), file_id, dsetTime);
    var res = read1DDataset(real(64), file_id, dsetRes);
    var cl = read1DDataset(real(64), file_id, dsetCl);
    var cd = read1DDataset(real(64), file_id, dsetCd);
    var cm = read1DDataset(real(64), file_id, dsetCm);
    var circulation = read1DDataset(real(64), file_id, dsetCirculation);
    var wakeGamma = read1DDataset(real(64), file_id, dsetWakeGamma);

    H5Fclose(file_id);
    return (xElem, yElem, rho, phi, it, time, res, cl, cd, cm, circulation, wakeGamma);
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

    return (X, Y, Z, element2node, wallElement2node, farfieldElement2node);
}

// proc readMesh(filename: string, elementType: string) {
//     // === Dataset paths ===
//     const dsetX = "/Base/Zone00001_step0/GridCoordinates/CoordinateX/ data";
//     const dsetY = "/Base/Zone00001_step0/GridCoordinates/CoordinateY/ data";
//     const dsetZ = "/Base/Zone00001_step0/GridCoordinates/CoordinateZ/ data";

//     const dsetElem     = "/Base/Zone00001_step0/" + "Elements" + "/ElementConnectivity/ data";
//     const dsetFarfield = "/Base/Zone00001_step0/farfield/ElementConnectivity/ data";
//     const dsetWall     = "/Base/Zone00001_step0/wall/ElementConnectivity/ data";

//     var file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
//     if file_id < 0 then
//         halt("Could not open file: ", filename);

//     // Read 1D coordinate arrays (type: real(64))
//     var X = read1DDataset(real(64), file_id, dsetX);
//     var Y = read1DDataset(real(64), file_id, dsetY);
//     var Z = read1DDataset(real(64), file_id, dsetZ);

//     // Read connectivity arrays
//     var element2node = read1DDataset(int, file_id, dsetElem);
//     var farfieldElement2node = read1DDataset(int, file_id, dsetFarfield);
//     var wallElement2node = read1DDataset(int, file_id, dsetWall);

//     var element2nodeList = new list(int);
//     var farfieldElement2nodeList = new list(int);
//     var wallElement2nodeList = new list(int);
//     if elementType == "QuadElements" {
//         for (i, node) in zip(element2node.domain, element2node) {
//             if (i-1) % 5 != 0 {
//                 element2nodeList.pushBack(node);
//             }
//         }
//         for (i, node) in zip(farfieldElement2node.domain, farfieldElement2node) {
//             if (i-1) % 3 != 0 {
//                 farfieldElement2nodeList.pushBack(node);
//             }
//         }
//         for (i, node) in zip(wallElement2node.domain, wallElement2node) {
//             if (i-1) % 3 != 0 {
//                 wallElement2nodeList.pushBack(node);
//             }
//         }
//     }
//     var newElment2node: [1..element2nodeList.size] int;
//     var newFarfieldElement2node: [1..farfieldElement2nodeList.size] int;
//     var newWallElement2node: [1..wallElement2nodeList.size] int;
//     for (i, node) in zip(newElment2node.domain, element2nodeList) {
//         newElment2node[i] = node;
//     }
//     for (i, node) in zip(newFarfieldElement2node.domain, farfieldElement2nodeList) {
//         newFarfieldElement2node[i] = node;
//     }
//     for (i, node) in zip(newWallElement2node.domain, wallElement2nodeList) {
//         newWallElement2node[i] = node;
//     }

//     H5Fclose(file_id);

//     return (X, Y, Z, newElment2node, newWallElement2node, newFarfieldElement2node);
// }

class MeshData {
    var elementType_: string;
    // === Connectivity ===
    var X_dom: domain(1) = {1..0}; // empty domain, can resize
    var elem2node_dom: domain(1) = {1..0}; // empty domain, can resize
    var elem2nodeIndex_dom: domain(1) = {1..0}; // empty domain, can resize
    var wallElem2node_dom: domain(1) = {1..0}; // empty domain, can resize
    var wallElem2nodeIndex_dom: domain(1) = {1..0}; // empty domain, can resize
    var farfieldElem2node_dom: domain(1) = {1..0}; // empty domain, can resize
    var farfieldElem2nodeIndex_dom: domain(1) = {1..0}; // empty domain, can resize
    var edge2node_dom: domain(2) = {1..0, 1..0}; // empty domain, can resize
    var elem2edge_dom: domain(1) = {1..0}; // empty domain, can resize
    var elem2edgeIndex_dom: domain(1) = {1..0}; // empty domain, can resize
    var edgeWall_dom: domain(1) = {1..0}; // empty domain, can resize
    var edgeFarfield_dom: domain(1) = {1..0}; // empty domain, can resize
    var edge2elem_dom: domain(2) = {1..0, 1..0}; // empty domain, can resize
    var esup1_dom: domain(1) = {1..0}; // empty domain, can resize
    var esup2_dom: domain(1) = {1..0}; // empty domain, can resize
    var psup1_dom: domain(1) = {1..0}; // empty domain, can resize
    var psup2_dom: domain(1) = {1..0}; // empty domain, can resize
    var esuel_dom: domain(1) = {1..0}; // empty domain, can resize
    var esuelIndex_dom: domain(1) = {1..0}; // empty domain, can resize

    // === Coordinates ===
    var X_: [X_dom] real(64);
    var Y_: [X_dom] real(64);
    var Z_: [X_dom] real(64);

    var elem2node_: [elem2node_dom] int;
    var elem2nodeIndex_: [elem2nodeIndex_dom] int;
    var wallElem2node_: [wallElem2node_dom] int;
    var wallElem2nodeIndex_: [wallElem2nodeIndex_dom] int;
    var farfieldElem2node_: [farfieldElem2node_dom] int;
    var farfieldElem2nodeIndex_: [farfieldElem2nodeIndex_dom] int;
    var edge2node_: [edge2node_dom] int;
    var elem2edge_: [elem2edge_dom] int;
    var elem2edgeIndex_: [elem2edgeIndex_dom] int;
    var edgeWall_: [edgeWall_dom] int;
    var edgeFarfield_: [edgeFarfield_dom] int;
    var edge2elem_: [edge2elem_dom] int;
    var esup1_: [esup1_dom] int;
    var esup2_: [esup2_dom] int;
    var psup1_: [psup1_dom] int;
    var psup2_: [psup2_dom] int;
    var esuel_: [esuel_dom] int;
    var esuelIndex_: [esuelIndex_dom] int;
    var elemToLocalEdgeIndex_: map(int, map(int, int));

    var nelem_: int; // number of elements
    var nedge_: int; // number of edges
    var nelemWithGhost_: int; // number of elements including ghost cells

    proc init(X : [] real(64), Y : [] real(64), elem2node : [] int, elem2nodeIndex : [] int,
               wallElem2node : [] int, farfieldElem2node : [] int) {
        this.X_dom = {1..X.size};
        this.elem2node_dom = {1..elem2node.size};
        this.elem2nodeIndex_dom = {1..elem2nodeIndex.size};
        this.wallElem2node_dom = {1..wallElem2node.size};
        this.farfieldElem2node_dom = {1..farfieldElem2node.size};

        this.X_ = X;
        this.Y_ = Y;
        this.elem2node_ = elem2node;
        this.elem2nodeIndex_ = elem2nodeIndex;
        this.wallElem2node_ = wallElem2node;
        this.farfieldElem2node_ = farfieldElem2node;

        this.nelem_ = elem2nodeIndex.size - 1;
    }

    proc init(filename: string, elementType: string) {
        var (Xtmp, Ytmp, Ztmp, element2nodeTmp, wallTmp, farfieldTmp) = readMesh(filename, elementType);

        var nelem = 0;
        var nnode = 0;
        if elementType == "QuadElements" {
            nnode = 4;
            nelem = element2nodeTmp.size / nnode;
        }
        else if elementType == "TriElements" {
            nnode = 3;
            nelem = element2nodeTmp.size / nnode;
        }

        var elem2nodeIndexTmp: [1..nelem + 1] int;
        elem2nodeIndexTmp[1] = 0;
        for i in 1..nelem {
            elem2nodeIndexTmp[i + 1] = elem2nodeIndexTmp[i] + nnode;
        }

        this.init(Xtmp, Ytmp, element2nodeTmp, elem2nodeIndexTmp, wallTmp, farfieldTmp);
        this.elementType_ = elementType;
    }

    proc buildConnectivity() {
        const edge2node = getEdge2Node();
        const (elem2edge, elem2edgeIndex) = getElem2edge(edge2node);
        const edge2elem = getEdge2elem(edge2node);
        const (esup1, esup2) = getElementsAroundNode();
        const (psup1, psup2) = getNodesAroundNode(esup1, esup2);
        getBoundaryFaces(edge2node, edge2elem);

        this.edge2node_dom = edge2node.domain;
        this.elem2edge_dom = elem2edge.domain;
        this.elem2edgeIndex_dom = elem2edgeIndex.domain;
        this.edge2elem_dom = edge2elem.domain;
        this.esup1_dom = esup1.domain;
        this.esup2_dom = esup2.domain;
        this.psup1_dom = psup1.domain;
        this.psup2_dom = psup2.domain;

        this.edge2node_ = edge2node;
        this.elem2edge_ = elem2edge;
        this.elem2edgeIndex_ = elem2edgeIndex;
        this.edge2elem_ = edge2elem;
        this.esup1_ = esup1;
        this.esup2_ = esup2;
        this.psup1_ = psup1;
        this.psup2_ = psup2;

        buildConnectivityWithGhostCells();

        for elem in 1..this.nelem_ {
            var globalIndexToLocalIndex = new map(int, int);
            const edges = this.elem2edge_[this.elem2edgeIndex_[elem]+1..this.elem2edgeIndex_[elem + 1]];
            for (edge, globalIndex) in zip(edges, edges.domain) {
                globalIndexToLocalIndex[edge] = globalIndex;
            }
            this.elemToLocalEdgeIndex_[elem] = globalIndexToLocalIndex;
        }

        // writeln("edge2node_ = ", this.edge2node_.shape, " = ", this.edge2node_);
        // writeln("edgeWall_ = ", this.edgeWall_.shape, " = ", this.edgeWall_);
        // writeln("edgeFarfield_ = ", this.edgeFarfield_.shape, " = ", this.edgeFarfield_);
        // writeln("elem2edge_ = ", this.elem2edge_.shape, " = ", this.elem2edge_);
        // writeln("elem2edgeIndex_ = ", this.elem2edgeIndex_.shape, " = ", this.elem2edgeIndex_);
        // writeln("edge2elem_ = ", this.edge2elem_.shape, " = ", this.edge2elem_);
        // writeln("esuel_ = ", this.esuel_.shape, " = ", this.esuel_);
        // writeln("esuelIndex_ = ", this.esuelIndex_.shape, " = ", this.esuelIndex_);
        // writeln("elemToLocalEdgeIndex_ = ", this.elemToLocalEdgeIndex_.size, " = ", this.elemToLocalEdgeIndex_);
        

        if elem != 0 {
            writeln("elem ", elem, " with points ", this.elem2node_[this.elem2nodeIndex_[elem]+1..this.elem2nodeIndex_[elem + 1]]);
            for point in this.elem2node_[this.elem2nodeIndex_[elem]+1..this.elem2nodeIndex_[elem + 1]] {
                writeln("point ", point, " is connected to elements ", this.esup1_[this.esup2_[point]+1..this.esup2_[point + 1]], " and nodes ", this.psup1_[this.psup2_[point]+1..this.psup2_[point + 1]]);
            }
            writeln("elem ", elem, " has edges ", this.elem2edge_[this.elem2edgeIndex_[elem]+1..this.elem2edgeIndex_[elem + 1]]);
            for edge in this.elem2edge_[this.elem2edgeIndex_[elem]+1..this.elem2edgeIndex_[elem + 1]] {
                writeln("edge ", edge, " with points ", this.edge2node_[1.., edge], " and elements ", this.edge2elem_[1.., edge]);
            }
        }
    }

    proc buildConnectivityWithGhostCells() {
        var ghostElem = this.nelem_ + 1;
        for edge in 1..this.nedge_ {
            const elem1 = this.edge2elem_[1, edge];
            const elem2 = this.edge2elem_[2, edge];
            if elem1 == 0 {
                this.edge2elem_[1, edge] = ghostElem;
                ghostElem += 1;
            }
            else if elem2 == 0 {
                this.edge2elem_[2, edge] = ghostElem;
                ghostElem += 1;
            }
        }
        this.nelemWithGhost_ = ghostElem - 1;

        var esuel = new list(int);
        var esuelIndex = new list(int);
        esuelIndex.pushBack(0);

        for elem in 1..this.nelem_ {
            var neighbors = new set(int);
            for edge in this.elem2edge_[this.elem2edgeIndex_[elem]+1..this.elem2edgeIndex_[elem + 1]] {
                const elem1 = this.edge2elem_[1, edge];
                const elem2 = this.edge2elem_[2, edge];
                if elem1 == elem && !neighbors.contains(elem2) {
                    esuel.pushBack(elem2);
                    neighbors.add(elem2);
                }
                else if elem2 == elem && !neighbors.contains(elem1) {
                    esuel.pushBack(elem1);
                    neighbors.add(elem1);
                }
            }
            esuelIndex.pushBack(esuel.size);
        }

        this.esuel_dom = {1..esuel.size};
        this.esuelIndex_dom = {1..esuelIndex.size};

        for i in 1..esuel.size {
            this.esuel_[i] = esuel[i-1];
        }
        for i in 1..esuelIndex.size {
            this.esuelIndex_[i] = esuelIndex[i-1];
        }
        
    }
    proc getEdge2Node() {
        var edgeSet = new map(string, (int, int)); // Use string keys to deduplicate

        for elem in 1..this.nelem_ {
            const nodeIndex = this.elem2nodeIndex_[elem]+1..this.elem2nodeIndex_[elem + 1];
            for i in nodeIndex {
                const i1 = this.elem2node_[i];
                const i2 = this.elem2node_[if i == nodeIndex.last then nodeIndex.first else i + 1];

                const ip = min(i1, i2);
                const jp = max(i1, i2);
                const key = try! "%i_%i".format(ip, jp);

                edgeSet.add(key, (ip, jp));
            }

        }

        const nedge = edgeSet.size;
        this.nedge_ = nedge;
        var edge2node: [1..2, 1..nedge] int;
        var idx = 1;

        for (ip, jp) in edgeSet.values() {
            edge2node[1, idx] = ip;
            edge2node[2, idx] = jp;
            idx += 1;
        }

        return edge2node;
    }

    proc getBoundaryFaces(edge2node : [] int, edge2elem : [] int) {
        // Build a map from edge (node pair) to edge index
        var edgeMap = new map((int, int), int);
        for edge in 1..this.nedge_ {
            const ip = min(edge2node[1, edge], edge2node[2, edge]);
            const jp = max(edge2node[1, edge], edge2node[2, edge]);
            edgeMap[(ip, jp)] = edge;
        }

        // Process wall edges from CGNS connectivity (2 nodes per edge)
        var edgeWall = new list(int);
        const nWallEdges = this.wallElem2node_.size / 2;
        for i in 0..<nWallEdges {
            const n1 = this.wallElem2node_[2*i + 1];
            const n2 = this.wallElem2node_[2*i + 2];
            const ip = min(n1, n2);
            const jp = max(n1, n2);
            if edgeMap.contains((ip, jp)) {
                edgeWall.pushBack(try! edgeMap[(ip, jp)]);
            }
        }

        // Process farfield edges from CGNS connectivity (2 nodes per edge)
        var edgeFarfield = new list(int);
        const nFarfieldEdges = this.farfieldElem2node_.size / 2;
        for i in 0..<nFarfieldEdges {
            const n1 = this.farfieldElem2node_[2*i + 1];
            const n2 = this.farfieldElem2node_[2*i + 2];
            const ip = min(n1, n2);
            const jp = max(n1, n2);
            if edgeMap.contains((ip, jp)) {
                edgeFarfield.pushBack(try! edgeMap[(ip, jp)]);
            }
        }

        this.edgeWall_dom = {1..edgeWall.size};
        for (i, edge) in zip(this.edgeWall_dom, edgeWall) {
            this.edgeWall_[i] = edge;
        }

        this.edgeFarfield_dom = {1..edgeFarfield.size};
        for (i, edge) in zip(this.edgeFarfield_dom, edgeFarfield) {
            this.edgeFarfield_[i] = edge;
        }
    }

    proc getElem2edge(edge2node : [] int) {
        var edgeMap = new map((int, int), int); // maps (min, max) -> edge ID

        // Step 1: Build edgeMap from edge2node
        for iedge in 1..this.nedge_ {
            const ip = min(edge2node[1, iedge], edge2node[2, iedge]);
            const jp = max(edge2node[1, iedge], edge2node[2, iedge]);
            edgeMap[(ip, jp)] = iedge;
        }

        var inedel = new list(int);
        var inedelIndex = new list(int);
        inedelIndex.pushBack(0);

        for elem in 1..nelem_ {
            const nodeIndex = this.elem2nodeIndex_[elem]+1..this.elem2nodeIndex_[elem + 1];
            for i in nodeIndex {
                const i1 = this.elem2node_[i];
                const i2 = this.elem2node_[if i == nodeIndex.last then nodeIndex.first else i + 1];

                const ip = min(i1, i2);
                const jp = max(i1, i2);
                try! inedel.pushBack(edgeMap[(ip, jp)]);
            }
            inedelIndex.pushBack(inedel.size);

        }

        var elem2edge: [1..inedel.size] int;
        var elem2edgeIndex: [1..inedelIndex.size] int;

        for i in 1..inedel.size {
            elem2edge[i] = inedel[i-1];
        }
        for i in 1..inedelIndex.size {
            elem2edgeIndex[i] = inedelIndex[i-1];
        }

        return (elem2edge, elem2edgeIndex);

    }

    proc getEdge2elem(edge2node : [] int) {
        // Map edge (ip,jp) => edge index
        var edgeMap = new map((int, int), int);

        for e in 1..this.nedge_ {
            const ip = edge2node[1, e];
            const jp = edge2node[2, e];
            edgeMap[(ip, jp)] = e;
        }

        // Prepare edge-to-element map: max 2 elems per edge
        var edge2elem: [1..2, 1..this.nedge_] int = 0;

        for elem in 1..this.nelem_ {
            const nodeIndex = this.elem2nodeIndex_[elem]+1..this.elem2nodeIndex_[elem + 1];
            for i in nodeIndex {
                const ip = this.elem2node_[i];
                const jp = this.elem2node_[if i == nodeIndex.last then nodeIndex.first else i + 1];

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

    proc getElementsAroundNode() {
        const nelem = this.nelem_;
        const npoin = this.X_.size;
        var esup2 : [1..npoin+1] int;
        var ipoi1, ipoin, istor: int;

        for i in 1..nelem {
            const nodeIndex = this.elem2nodeIndex_[i]+1..this.elem2nodeIndex_[i + 1];
            for j in nodeIndex {
                ipoi1 = this.elem2node_[j] + 1;
                esup2[ipoi1] = esup2[ipoi1] + 1;
            }
        }

        for i in 2..npoin+1 {
            esup2[i] = esup2[i] + esup2[i-1];
        }
        var esup1 : [1..esup2[npoin+1]] int;
        for i in 1..nelem {
            const nodeIndex = this.elem2nodeIndex_[i]+1..this.elem2nodeIndex_[i + 1];
            for j in nodeIndex {
                ipoin = this.elem2node_[j];
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

    proc getNodesAroundNode(esup1 : [] int, esup2 : [] int) {
        const npoin = this.X_.size;
        var lpoin : [1..npoin] int;
        var psup2 : [1..npoin+1] int;
        var istor, ielem, jpoin : int;

        for ipoin in 1..npoin {
            for iesup in (esup2[ipoin]+1)..esup2[ipoin+1] {
                ielem = esup1[iesup];
                const nodeIndex = this.elem2nodeIndex_[ielem]+1..this.elem2nodeIndex_[ielem + 1];
                for j in nodeIndex {
                    jpoin = this.elem2node_[j];
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
                const nodeIndex = this.elem2nodeIndex_[ielem]+1..this.elem2nodeIndex_[ielem + 1];
                for j in nodeIndex {
                    jpoin = this.elem2node_[j];
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
}

}