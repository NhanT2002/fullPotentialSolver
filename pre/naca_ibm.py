import Generator.PyTree as G
import Converter.PyTree as C
import Geom.PyTree as D
import Converter.Internal as I

s = D.naca('0012', N=1001)
snearList = [0.01]   # smallest cell size near the airfoil
dfar = 100.0           # far-field extension (in chord lengths)
mesh = G.octree([s], snearList, dfar=dfar)
mesh = G.octree2Struct(mesh)

I.printTree(mesh)

zones = I.getZones(mesh)
print("Zones before renaming:")
for z in zones:
    print("  ", z[0])

for i, z in enumerate(zones):
    mesh = I.renameNode(mesh, z[0], f'Block_{i+1}')

C.convertPyTree2File(mesh, 'naca_octree.cgns')