module writeCGNS 
{
use cgns;
use ElementTopology;
use CTypes;
use mesh;
use FileSystem;
use List;
use Map;
use arrayOperations;
use CGNSextern;

class MyCGNSfile_c : CGNSfile_c {
  proc init(filename: string, mode: CGNSOpenMode_t, basename: string) {
    super.init(filename, mode, basename);
  }
  proc writeMultigridElementsConnectivity(baseId : c_int, zoneId : c_int, dim : c_int, ref elementCon : [?D] int, ref elementStart : [?D1] int, ref elementTypes : [?D2] c_int, oneBase : bool = false, prefix : string = "Elements", offset : int = 0)
  {
    var uniqueElements = [elementTypes[0]];
    const nElmentType : int = uniqueElements.size;
    const baseIndex : cgsize_t = if oneBase then 0 : cgsize_t else 1 : cgsize_t;

    var etypeDom : domain(1) = {0..#nElmentType};
    var totalElementCount : int = 1 + offset;
    for i in etypeDom do
    {
      const elementType  : c_int = uniqueElements[i];
      const nElement     : int = elementTypes.count(elementType);

      if elementType == NGON_n {
        // NGON_n: Use elementCon and elementStart directly
        var sectionName : string = prefix + "_NGON_n";
        var ngonConn: [0..elementCon.size-1] cgsize_t;
        var ngonOffset: [0..elementStart.size-1] cgsize_t;
        forall idx in 0..elementCon.size-1 do
            ngonConn[idx] = elementCon[idx] + baseIndex;
        forall idx in 0..elementStart.size-1 do
            ngonOffset[idx] = elementStart[idx];

        var nBoundaryElements : c_int = 0;
        var sectionId : c_int;
        var status : c_int = cg_poly_section_write(this.fileId_, baseId, zoneId, sectionName.c_str(), elementType, totalElementCount, totalElementCount + nElement - 1,
                                                    nBoundaryElements, ngonConn, ngonOffset, sectionId);
        checkCGNSStatus(status, "Cannot write section connectivity in cgns file!");
        totalElementCount += nElement;
      } 
    }
  }
}


class potentialFlowWriter_c {
  var cgnsFileName_ : string;
  var cgnsFile_    : owned MyCGNSfile_c;
  var zoneId : c_int;

  proc init(inputFileName : string) {
    var baseName : string = "Base";

    cgnsFileName_ = inputFileName;

    var name = new list(string);
    for st in inputFileName.split(".") {
      name.pushBack(st);
    }

    var i: int = 0;
    var fileExist = try! exists(cgnsFileName_);
    while fileExist {
      i += 1;
      cgnsFileName_ = name[0] + "_" + i:string + "." + name[1];
      fileExist = try! exists(cgnsFileName_);
    }

    cgnsFile_ = new owned MyCGNSfile_c(cgnsFileName_, CGNSOpenMode_t.WRITE, baseName);

    const cellDim : c_int     = 2;
    const physDim : c_int     = 3;

    const glbBaseId : c_int = cgnsFile_.createBase(baseName, cellDim, physDim);
  }

  proc writeMeshMultigrid(ref X: [] real(64), ref Y: [] real(64), ref conn: [] int, ref connStart: [] int) {

    const nnode = X.size;
    const nelem = connStart.size - 1;
    var connType: [0..<nelem] c_int;
    connType = NGON_n; // Assuming NGON_n for multigrid

    const zoneName : string = "dom-1";
    const cellDim : c_int = 2;

    var size : [0..2] cgsize_t;

    size[0] = nnode; // number of nodes
    size[1] = nelem; // number of elements
    size[2] = 0;

    zoneId = cgnsFile_.createZone(1, zoneName, size, CGNStypes.Unstructured);
    var Z: [X.domain] real(64) = 0.0; // Assuming Z is zero for 2D problems
    cgnsFile_.writeGridCoordinates(1, zoneId, X, Y, Z);


    // writeln("conn ",conn.shape, " = ", conn);
    // writeln("connStart ",connStart.shape, " = ", connStart);
    // writeln("connType ",connType.shape, " = ", connType);
    cgnsFile_.writeMultigridElementsConnectivity(1, zoneId, cellDim, conn, connStart, connType, oneBase = true, prefix = "Elements");
    writeln("Mesh written to CGNS file: ", cgnsFileName_);
  }



  proc writeMesh(ref Mesh: borrowed MeshData) {
    ref elementsCon = Mesh.elem2node_;
    ref element2nodeIndex = Mesh.elem2nodeIndex_;

    var conn: [0..<elementsCon.size] int;
    var connStart: [0..<element2nodeIndex.size] int;
    var connType: [0..<element2nodeIndex.size-1] c_int;
    if Mesh.elementType_ == "QuadElements" {
      connType = QUAD_4;
    }
    else if Mesh.elementType_ == "TriElements" {
      connType = TRI_3;
    }
    

    // Flatten the element2node array
    for i in 1..elementsCon.size {
      conn[i-1] = elementsCon[i];
    }
    for i in 1..element2nodeIndex.size {
      connStart[i-1] = element2nodeIndex[i];
    }

    const zoneName : string = "dom-1";
    const cellDim : c_int = 2;

    var size : [0..2] cgsize_t;

    size[0] = Mesh.X_.size;
    size[1] = Mesh.nelem_;
    size[2] = 0;

    zoneId = cgnsFile_.createZone(1, zoneName, size, CGNStypes.Unstructured);

    cgnsFile_.writeGridCoordinates(1, zoneId, Mesh.X_, Mesh.Y_, Mesh.Z_);

    // writeln("conn ",conn.shape, " = ", conn);
    // writeln("connStart ",connStart.shape, " = ", connStart);
    // writeln("connType ",connType.shape, " = ", connType);
    cgnsFile_.writeMixedElementsConnectivity(1, zoneId, cellDim, conn, connStart, connType, oneBase = true, prefix = "Elements");
  }

  // proc writeMeshWithGhost(ref Mesh: shared MeshData) {
  //   ref elementsCon = Mesh.elem2node_;

  //   var conn: [0..<elementsCon.size] int;
  //   var connStart: [0..elementsCon.dim(1).size] int;
  //   var connType: [0..<elementsCon.dim(1).size] c_int;
  //   if Mesh.elementType_ == "QuadElements" {
  //     connType = QUAD_4;
  //   }
  //   else if Mesh.elementType_ == "TriElements" {
  //     connType = TRI_3;
  //   }

  //   // Flatten the element2node array
  //   for i in 1..elementsCon.dim(1).size {
  //     connStart[i] = elementsCon.dim(0).size*i;
  //     for j in 1..elementsCon.dim(0).size {
  //       conn[j-1 + (i-1) * elementsCon.dim(0).size] = elementsCon[j,i];
  //     }
  //   }

  //   const zoneName : string = "dom-1";
  //   const cellDim : c_int = 2;

  //   var size : [0..2] cgsize_t;

  //   size[0] = Mesh.X_.size;
  //   size[1] = elementsCon.dim(1).size;
  //   size[2] = 0;

  //   zoneId = cgnsFile_.createZone(1, zoneName, size, CGNStypes.Unstructured);

  //   cgnsFile_.writeGridCoordinates(1, zoneId, Mesh.X_, Mesh.Y_, Mesh.Z_);

  //   cgnsFile_.writeMixedElementsConnectivity(1, zoneId, cellDim, conn, connStart, connType, oneBase = true, prefix = "Elements");
  // }

  /**
      Writes the solution contained in global handle to a cgns file.

      :arg globalHandle: The potential floe global handle object.
  **/
  proc writeSolution(dom: domain(1), fieldMap: map(string, [dom] real(64))) {
    const baseId : c_int = cgnsFile_.baseId_;
    var solIDcc : c_int = cgnsFile_.addCellCenteredSolution(baseId, zoneId, "FLOW_SOLUTION_CC");

    for name in fieldMap.keys() {
      var values = try! fieldMap[name];
      cgnsFile_.addFieldSolution(baseId, zoneId, solIDcc, name, values);
    }

    writeln("Solution ", cgnsFileName_, " written with fields: ", fieldMap.keys());
  }

  proc writeWallSolution(ref Mesh: shared MeshData, dom: domain(1), fieldMap: map(string, [dom] real(64))) {
    // === Create a separate base for the wall ===
    const wallBaseName = "WallBase";
    const cellDim: c_int = 1;      // 1D elements (lines)
    const physDim: c_int = 2;      // 2D space (x, y)
    const wallBaseId = cgnsFile_.createBase(wallBaseName, cellDim, physDim);

    // === Wall zone creation ===
    const zoneName = "wall";
    const nWallEdges = Mesh.edgeWall_.size;
    var size: [0..2] cgsize_t;
    size[0] = nWallEdges * 2;  // total wall nodes (not necessarily unique)
    size[1] = nWallEdges;      // number of line elements
    size[2] = 0;

    zoneId = cgnsFile_.createZone(wallBaseId, zoneName, size, CGNStypes.Unstructured);

    // === Coordinates for wall nodes ===
    var wallX, wallY, wallZ: [1..size[0]] real(64);
    for i in 1..nWallEdges {
      const edge = Mesh.edgeWall_[i];
      const n1 = Mesh.edge2node_(1, edge);
      const n2 = Mesh.edge2node_(2, edge);

      wallX[2*i - 1] = Mesh.X_[n1];
      wallY[2*i - 1] = Mesh.Y_[n1];
      wallZ[2*i - 1] = Mesh.Z_[n1];

      wallX[2*i] = Mesh.X_[n2];
      wallY[2*i] = Mesh.Y_[n2];
      wallZ[2*i] = Mesh.Z_[n2];
    }

    cgnsFile_.writeGridCoordinates(wallBaseId, zoneId, wallX, wallY, wallZ);

    // === Connectivity: BAR_2 (2-node line) elements ===
    var conn: [0..2*nWallEdges - 1] int;
    var connStart: [0..nWallEdges] int;
    var connType: [0..nWallEdges - 1] c_int;
    for i in 0..<nWallEdges {
      connStart[i+1] = (i+1) * 2;
      conn[2*i]     = 2*i + 1;
      conn[2*i + 1] = 2*i + 2;
      connType[i]   = BAR_2;
    }

    cgnsFile_.writeMixedElementsConnectivity(wallBaseId, zoneId, cellDim, conn, connStart, connType, oneBase = true, prefix = "WallElements");

    // === Write wall solution fields ===
    const solIDcc = cgnsFile_.addCellCenteredSolution(wallBaseId, zoneId, "WALL_FLOW_SOLUTION_CC");

    for name in fieldMap.keys() {
      var values = try! fieldMap[name];
      cgnsFile_.addFieldSolution(wallBaseId, zoneId, solIDcc, name, values);
    }
  }

  proc writeConvergenceHistory(time: list(real(64)), iterations: list(int), residuals: list(real(64)), cls: list(real(64)), cds: list(real(64)), cms: list(real(64)), circulation: list(real(64))) {
    const nMetrics = 7;
    const maxIter = iterations.size;

    var iterationsArray: [0..<maxIter] int;
    forall i in 0..<maxIter {
      iterationsArray[i] = iterations[i];
    }

    var convData: [0..<nMetrics, 0..<maxIter] real;
    convData[0, ..] = time;
    convData[1, ..] = residuals;
    convData[2, ..] = cls;
    convData[3, ..] = cds;
    convData[4, ..] = cms;
    convData[5, ..] = circulation;

    var names = ["Time", "Residual", "Cl", "Cd", "Cm", "Circulation"];

    cgnsFile_.writeGlobalConvergenceHistory("ConvergenceHistory", iterationsArray, names, convData);


  }

  proc writeUnsteadyHistory(time: list(real(64)), iterations: list(int), alpha: list(real(64)), cls: list(real(64)), cds: list(real(64)), cms: list(real(64))) {
    const nMetrics = 6;
    const maxIter = time.size;

    var iterationsArray: [0..<maxIter] int;
    forall i in 0..<maxIter {
      iterationsArray[i] = iterations[i];
    }

    var convData: [0..<nMetrics, 0..<maxIter] real(64);
    convData[0, ..] = time;
    convData[1, ..] = alpha;
    convData[2, ..] = cls;
    convData[3, ..] = cds;
    convData[4, ..] = cms;

    var names = ["Time", "Alpha", "Cl", "Cd", "Cm"];

    cgnsFile_.writeGlobalConvergenceHistory("UnsteadyHistory", iterationsArray, names, convData);

    writeln("Unsteady history written to CGNS file ", cgnsFileName_);


  }

}

}