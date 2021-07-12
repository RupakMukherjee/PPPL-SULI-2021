import sys
import numpy as np
from numpy import cos, pi
from numpy.lib.polynomial import polyint
from skimage.measure import marching_cubes_lewiner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

n = 10
xmin, xmax = (-1, 1)
ymin, ymax = (-1, 1)
zmin, zmax = (-1, 1)
points = np.mgrid[xmin:xmax:n*1j, ymin:ymax:n*1j, zmin:zmax:n*1j]
xv, yv, zv = points
vol = (xv**2 + yv**2 + zv**2)
iso_val=1
verts, faces, _, _ = marching_cubes_lewiner(vol, iso_val, spacing=(1, 1, 1))
verts[:, 0] = (xmax - xmin)*verts[:, 0]/(n-1) + xmin
verts[:, 1] = (ymax - ymin)*verts[:, 1]/(n-1) + ymin
verts[:, 2] = (zmax - zmin)*verts[:, 2]/(n-1) + zmin


#Graphing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral', lw=3)
ax.set_box_aspect((1, 1, 1))
#ax.plot(verts[:, 0], verts[:,1], verts[:, 2], 'bo') #plots vertices
#plt.show()

#Building data structures
verts = np.around(verts, 6)
triangles = verts[faces]
triangles = np.around(triangles, 6)
triangles = triangles.tolist()
faces = set()
edges = set()
edgedict = {}
verts = np.unique(verts, axis=0)
verts = verts.tolist()
vert = [] #round verts
for l in verts:
    tup = ()
    for t in l:
        t = round(t, 6)
        tup = tup + (t,)
    vert.append(tup)
verts = vert



for f in triangles: 
    a, b, c = tuple(f[0]), tuple(f[1]) ,tuple(f[2])
    if (a == b) or (b == c) or (c == a): #deletes degen triangles
        continue
    faces.add((a, b, c))
    edges.add(frozenset((a, b))) #adds edges, no duplicates
    edges.add(frozenset((b, c)))
    edges.add(frozenset((c, a)))
for e in edges:
    e = list(e)
    a, b = e[0], e[1]
    a = (round(a[0], 6), round(a[1], 6), round(a[2], 6))
    b = (round(b[0], 6), round(b[1], 6), round(b[2], 6))
    if a in edgedict:
        edgedict[a].append(b)
    else:
        edgedict[a] = [b]
    if b in edgedict:
        edgedict[b].append(a)
    else:
        edgedict[b] = [a]


def isBlob(verts, edges, faces):
    #checking Euler's formula F+V= E+2 
    #Euler's formula works for planar objects
    V = len(verts)
    E = len(edges)
    F = len(faces)
    return (F + V == E + 2 and 3*F == 2*E)

print(isBlob(verts, edges, faces))



#Printing trianlges to notepad
f = open('triangles.txt', 'w') 
#f.write(f'z = {h.levels[i]} \n')
f.write(f'{triangles}')
f.close()

#triangles_arr = triangles.reshape(triangles.shape[0], -1)
#np.savetxt('triangle.txt', triangles_arr, fmt = '%10.5f')


def inTri2D(p1, p2, p3, p):
    """
    For triangle in 2D defined by p1,p2,p3, see if p is inside triangle. 
    Uses barycentric coords.

    Calculates parameters
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    c = 1 - a - b
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = p

    k = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)) #fix divide by 0
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3))
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3))
    c = k - a - b
    mink = min(0, k)
    maxk = max(0, k)
    if (mink <= a <= maxk) and (mink <= b <= maxk) and (mink <= c <= maxk):
        return True
    else:
        return False

def inTri(p1, p2, p3, p):
    """    
    x = -(B(y-y1) + C(z-z1))/A +x1
    Returns Bool, x
    If a = 0, triangle is parallel, then ?
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
    if inTri2D(p1[1:], p2[1:], p3[1:], p[1:]):
        #x0 = -(b*(y-y1) + c*(z-z1))/a +x1 #calculate point on plane of triangle
        x00 = -(b*(y-y1) + c*(z-z1)) +x1*a
        if a > 0: #Checks if point is in ray
            if (x00 >= x*a) and a != 0:
                return True, x00/a
            else: return False, None
        elif a < 0:
            if (x00 <= x*a) and a != 0:
                return True, x00/a
            else: return False, None
        else: return False, None
    else: return False, None

def intersect2D(segment, point):
    """
    Calculates if a ray of x->inf from "point" intersects with the segment
    segment = [(x1, y1), (x2, y2)]
    """
    x, y = point
    
    #print(segment)
    x1, y1 = segment[0]
    x2, y2 = segment[1]
    if (y1<=y and y<y2) or (y2<=y and y<y1):
        x3 = x2*(y1-y)/(y1-y2) + x1*(y-y2)/(y1-y2)
        if x3 >= x:
            return True
        else: return False
    else: return False

def isInside2D(contour, point):
    x, y = point
    count = 0
    for i in range(len(contour)-1):
        point1 = contour[i]
        point2 = contour[i+1]
        if intersect2D(contour[i:i+2],point):
            count += 1
    #print(count)
    if count % 2 == 1:
        return True
    else:
        return False

def is_inside(verts, faces, point):
    """
    Can only intersect each point once.
    """
    count = 0
    county = 0
    visited = set()
    visitedy = set()
    for face in faces:
        p1 = face[0]
        p2 = face[1]
        p3 = face[2]
        in_tri, x = inTri(p1, p2, p3, point)
        x = np.float32(x)
        if x != None: x = round(x, 6)
        if in_tri and (x not in visited):
            visited.add(x)
            intpoint = (x, np.float32(point[1]), np.float32(point[2])) 
            if x == point[0]:
                count += 1
                continue
            #print(type(x))
            if intpoint in verts: #If hits vertex, sees if counts or not
                surround = edgedict[intpoint] #dfs to determine contour polygon
                unvisited = set(surround.copy())
                first = surround[0]
                current = first
                contour = []
                for j in range(len(surround)-1):
                    contour.append(current[1:])
                    unvisited.remove(current)
                    next = list(set(edgedict[current]) & set(unvisited))[0]
                    current = next
                contour.append(next[1:])
                contour.append(first[1:])
                if isInside2D(contour, intpoint[1:]):
                    count += 1
            else: 
                count += 1
    if (count % 2 == 1):
        return True
    else:
        return False

'''
coords = points.reshape(3, -1).T
coords = coords #+ .1
inside = np.zeros((n**3, 3))
for i, j in enumerate(coords):
    b = is_inside(verts, faces, j)
    if b:
        ax.scatter(*j)
        None
'''

point = (-1, 0, 0)
#print(type(point[1]))
print(is_inside(verts, faces, point))



plt.show()
#print(verts[faces])