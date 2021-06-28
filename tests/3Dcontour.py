import numpy as np
from numpy import cos, pi
from skimage.measure import marching_cubes_lewiner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 5
xmin = -2
xmax = 2
ymin = -2
ymax = 2
zmin = -2
zmax = 2
x, y, z = np.mgrid[xmin:xmax:n*1j, ymin:ymax:n*1j, zmin:zmax:n*1j]
vol = (x**2 + y**2 + z**2)
iso_val=4
verts, faces, _, _ = marching_cubes_lewiner(vol, iso_val, spacing=(1, 1, 1))

verts[:, 0] = (xmax - xmin)*verts[:, 0]/(n-1) + xmin
verts[:, 1] = (ymax - ymin)*verts[:, 1]/(n-1) + ymin
verts[:, 2] = (zmax - zmin)*verts[:, 2]/(n-1) + zmin
#verts = 2*verts/(n-1) -1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral', lw=3)
ax.set_box_aspect((1, 1, 1))
plt.show()

print(faces)
"""
f = open('segments.txt', 'w') 
linelist = h.collections
for i, line in enumerate(linelist):
    seglist = line.get_segments() #array of coord
    if type(seglist) == list and seglist != []:
        seglist = np.concatenate(seglist)
    #seglist = np.unique(seglist, axis = 0) remove duplicates
    f.write(f'z = {h.levels[i]} \n')
    f.write(f'{seglist} \n \n')

f.close()

"""

def in_triangle2D(p1, p2, p3, p):
    """
    For triangle in 2D defined by p1,p2,p3, see if p is inside triangle. 
    Uses barycentric coords.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = p
    #print(((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)))
    #a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    #b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    #c = 1 - a - b
    k = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3))
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3))
    c = k - a - b
    if (0 <= a and a <= k) and (0 <= b and b <= k) and (0 <= c and c <= k):
        return True
    else:
        return False

def is_inside_tri(p1, p2, p3, p):
    """
    x = -(B(y-y1) + C(z-z1))/A +x1
    """
    #yz in triangle2D
    #find x using the eq of plane, using normal
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    x, y, z = p
    v1 = [x3 - x1, y3 - y1, z3 - z1]
    v2 = [x2 - x1, y2 - y1, z2 - z1]
    n = [v1[1] * v2[2] - v1[2] * v2[1], #normal is cross product
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]]
    a, b, c = n 
    if in_triangle2D(p1[1:], p2[1:], p3[1:], p[1:]):
        x0 = -(b*(y-y1) + c*(z-z1))/a +x1 #calculate point on plane of triangle
        if x0 >= x:
            return True
        else: return False
    else: return False


def is_inside(verts, faces, point):
    count = 0
    for i in range(len(faces)):
        face = faces[i]
        p1 = verts[face[0]]
        p2 = verts[face[1]]
        p3 = verts[face[2]]
        if is_inside_tri(p1, p2, p3, point):
            count += 1
    print(count)
    if count % 2 == 1:
        return True
    else:
        return False

print(is_inside_tri((0, -1, -1), (0, 1, -1), (0, 0, 1), (1, 0, 0)))
is_inside(verts, faces, (0.2,.1,0))