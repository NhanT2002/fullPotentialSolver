/**
    This module contains different element topology
**/
module elementTopology 
{
proc quadElementTopology_2d() {
    // Node layout:
    //          face 1
    //       (2)------(1)
    // face 2 |        |
    //        |        | face 4
    //       (3)------(4)
    //          face 3

    var nnode: int = 4; // number of nodes per element
    var nfael: int = 4; // number of faces per element
    var nedel: int = 4; // number of edges per element

    var lpofa: [1..2, 1..4] int;
    lpofa[1,1] = 1;
    lpofa[2,1] = 2;
    lpofa[1,2] = 2;
    lpofa[2,2] = 3;
    lpofa[1,3] = 3;
    lpofa[2,3] = 4;
    lpofa[1,4] = 4;
    lpofa[2,4] = 1;

    // 2 nodes per face
    var lnofa: [1..4] int;
    lnofa = 2;

    var lpoed: [1..2, 1..4] int;
    lpoed[1,1] = 1;
    lpoed[2,1] = 2;
    lpoed[1,2] = 2;
    lpoed[2,2] = 3;
    lpoed[1,3] = 3;
    lpoed[2,3] = 4;
    lpoed[1,4] = 4;
    lpoed[2,4] = 1;

    return (lnofa, lpofa, lpoed, nnode, nfael, nedel);

}

proc triElementTopology_2d() {
    // Node layout:
    //            (1)
    //   (face1) /  \
    //          /    \  (face3)
    //       (2)------(3)
    //         (face2)

    var nnode: int = 3; // number of nodes per element
    var nfael: int = 3; // number of faces per element
    var nedel: int = 3; // number of edges per element

    var lpofa: [1..2, 1..3] int;
    lpofa[1,1] = 1;
    lpofa[2,1] = 2;
    lpofa[1,2] = 2;
    lpofa[2,2] = 3;
    lpofa[1,3] = 3;
    lpofa[2,3] = 1;

    // 2 nodes per face
    var lnofa: [1..3] int;
    lnofa = 2;

    var lpoed: [1..2, 1..3] int;
    lpoed[1,1] = 1;
    lpoed[2,1] = 2;
    lpoed[1,2] = 2;
    lpoed[2,2] = 3;
    lpoed[1,3] = 3;
    lpoed[2,3] = 1;

    return (lnofa, lpofa, lpoed, nnode, nfael, nedel);
}

}
