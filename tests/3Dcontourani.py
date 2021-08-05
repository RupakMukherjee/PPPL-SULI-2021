import sys
import time
import numpy as np
import math
from random import random
from numpy import cos, pi
from numpy.lib.polynomial import polyint
from skimage.measure import marching_cubes_lewiner
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

start = time.time()

coords = []
filename = "Data/fort.0" 
with open(filename) as f:
    for line in f:
        coords.append(list(map(float, line.split())))
dt = 4e-3
xmin, ymin, zmin, __ = coords[0]
xmax, ymax, zmax, __ = coords[-1]
dz = coords[1][2]-zmin
nz = round((zmax - zmin)/dz + 1)
dy = coords[nz][1] - ymin
ny = round((ymax - ymin)/dy + 1)
dx = coords[ny*nz][0]-xmin
nx = round((xmax - xmin)/dx + 1)
xcoords = np.linspace(xmin, xmax, nx)
ycoords = np.linspace(ymin, ymax, ny)
zcoords = np.linspace(zmin, zmax, nz)


framecount = 100
blobframe = [[] for i in range(framecount)]
bloblist = []

def interpTestPoint(point, dens):
    x, y, z = point
    idx = int((x-xmin)/dx)
    idy = int((y-ymin)/dy)
    idz = int((z-zmin)/dz)

    probeDensity = dens[idx:idx+2, idy:idy+2, idz:idz+2]

    xWeight = abs(xcoords[idx]-x)
    yWeight = abs(ycoords[idy]-y)
    zWeight = abs(zcoords[idz]-z)

    testDensity000 = probeDensity[0,0,0] * (dx-xWeight) * (dy-yWeight) * (dz-zWeight)
    testDensity001 = probeDensity[0,0,1] * (dx-xWeight) * (dy-yWeight) * zWeight
    testDensity010 = probeDensity[0,1,0] * xWeight * (dy-yWeight) * (dz-zWeight)
    testDensity011 = probeDensity[0,1,1] * xWeight * (dy-yWeight) * (dz-zWeight)
    testDensity100 = probeDensity[1,0,0] * (dx-xWeight) * yWeight * (dz-zWeight)
    testDensity101 = probeDensity[1,0,1] * (dx-xWeight) * yWeight * zWeight
    testDensity110 = probeDensity[1,1,0] * xWeight * yWeight * (dz-zWeight)
    testDensity111 = probeDensity[1,1,1] * xWeight * yWeight * zWeight
    testDensity = ( testDensity000 + testDensity001 + testDensity010 + testDensity011 + testDensity100 + testDensity101 + testDensity110 + testDensity111 ) / (dx*dy*dz)
    return testDensity

class Blob:
    def __init__(self, dens, verts, vertind, edges, faces):
        self.frame = -1
        self.verts = verts
        self.vertind = vertind
        self.edges = edges
        self.faces = faces
        vertcoords = verts[vertind]
        self.vertcoords = vertcoords
        V = len(vertind)
        xcom = 0
        ycom = 0
        zcom = 0
        denscom = 0
        for v in vertcoords:
            vx, vy, vz = v
            vdens = interpTestPoint(v, dens)
            denscom += vdens
            xcom += vdens*vx
            ycom += vdens*vy
            zcom += vdens*vz
        self.xcom = xcom/denscom
        self.ycom = ycom/denscom
        self.zcom = zcom/denscom
        self.com = (self.xcom, self.ycom, self.zcom)
        x = vertcoords[:,0]
        y = vertcoords[:,1]
        z = vertcoords[:,2]
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        self.z_min = np.min(z)
        self.z_max = np.max(z)
        self.xrange = self.x_max-self.x_min
        self.yrange = self.y_max-self.y_min
        self.zrange = self.z_max-self.z_min
        self.id = None
        self.vel = 0
        self.vol = 0
    
    def set_id(self, id):
        self.id = id
    
    def get_id(self):
        return self.id

    def is_close(self, blob):
        distThreshold = 0.1
        rangeThreshold = 0.1
        volThreshold = 2
        dist = math.dist(self.com, blob.com)
        bool1 = dist < distThreshold
        bool2 = math.dist([self.xrange], [blob.xrange]) < rangeThreshold
        bool3 = math.dist([self.yrange], [blob.xrange]) < rangeThreshold
        bool4 = math.dist([self.zrange], [blob.xrange]) < rangeThreshold
        bool5 = blob.vol/volThreshold < self.vol and self.vol < blob.vol * volThreshold
        return (all([bool1, bool2, bool3, bool4, bool5]), dist)

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

def inTri1(p1, p2, p3, p):
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
    v1 = p3-p1
    v2 = p2-p1
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

def inTri(p1, p2, p3, p):
    #Moller-Trumbore algorithm https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    eps = 0.0000001
    r = np.array([1, 0, 0])
    v1 = p2-p1
    v2 = p3-p1
    h = np.array([0, -v2[2], v2[1]]) #np.cross(r, v2)
    a = np.dot(v1, h)
    if (a > -eps and a < eps):
        return False, None
    f = 1/a
    s = p-p1
    u = f * np.dot(s, h)
    if (u < 0 or u > 1):
        return False, None
    q = np.array([s[1] * v1[2] - s[2] * v1[1],  #np.cross(s, v1)
        s[2] * v1[0] - s[0] * v1[2],
        s[0] * v1[1] - s[1] * v1[0]])
    v = f * q[0] #f * r.dot(q)
    if (v < 0 or u + v > 1):
        return False, None
    t = f * np.dot(v2, q)
    if (t > eps): #ray intersection
        outPoint = p + r * t
        outx = p[0] + t
        return True, outx
    else: #Line intersection but no ray intersection
        return False, None


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
def isInside(verts, edgedict, faces, point):
    """
    Can only intersect each point once.
    """
    count = 0
    county = 0
    visited = set()
    visitedy = set()
    for face in faces:
        p1 = verts[face[0]]
        p2 = verts[face[1]]
        p3 = verts[face[2]]
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
                surround = edgedict[np.where(verts == intpoint)[0][0]] #dfs to determine contour polygon
                unvisited = set(surround.copy())
                first = surround[0]
                current = first
                contour = []
                b = 0
                for j in range(len(surround)-1):
                    contour.append(verts[current][1:])
                    unvisited.remove(current)
                    next = list(set(edgedict[current]) & set(unvisited))
                    if next != []: next = next[0]
                    else:  
                        b = 1
                        break
                    current = next
                if b == 1: break
                contour.append(verts[next][1:])
                contour.append(verts[first][1:])
                if isInside2D(contour, intpoint[1:]):
                    count += 1
            else: 
                count += 1
    if (count % 2 == 1):
        return True
    else:
        return False

def findVolume(verts, faces):
    vol = 0
    for f in faces:
        p1 = verts[f[0]]
        p2 = verts[f[1]]
        p3 = verts[f[2]]
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3
        v1 = [x3 - x1, y3 - y1, z3 - z1]
        v2 = [x2 - x1, y2 - y1, z2 - z1]
        n = [v1[1] * v2[2] - v1[2] * v2[1], #normal is cross product
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]]
        xcom = (x1+x2+x3)/3
        vol += xcom*n[0]/2
    return abs(vol)



def updateplt(frame):
    print(frame)
    
    coords = []
    filename = "Data/fort.%d" % (frame)
    with open(filename) as f:
        for line in f:
            coords.append(list(map(float, line.split())))
    #dens[i][j][k] gives density at (i,j,k)
    dens = np.reshape(np.array(coords)[:, 3], (nx, ny, nz))

    thresholdDensity = 1e20

    verts, faces, _, _ = marching_cubes_lewiner(dens, thresholdDensity, spacing=(1, 1, 1))
    verts[:, 0] = dx*verts[:, 0] + xmin
    verts[:, 1] = dy*verts[:, 1] + ymin
    verts[:, 2] = dz*verts[:, 2] + zmin



    #plt.show()

    #Building data structures
    """
    edgedict is dictionary mapping each vertex to a list of its adjacent vertices. 
    triangles is triangles with coords
    faces is triangles by indices
    verts is 
    """

    triangles = verts[faces]
    triangles = np.around(triangles, 6)
    #triangles = triangles.tolist()
    edges = set()
    edgedict = {}
    verts = np.around(verts, 6)
    vert = [] #round verts
    nv = len(verts)
    for l in verts:
        tup = ()
        for t in l:
            #t = round(t, 6)
            t = np.float32(t)
            tup = tup + (t,)
        vert.append(tup)
    vertdict = {j:i for i,j in enumerate(vert)} #vert to index
    
    for f in faces: 
        a, b, c = f
        #if (a == b) or (b == c) or (c == a): #deletes degen triangles, not needed for large data size
        #    continue
        edges.add(frozenset((a, b))) #adds edges, no duplicates
        edges.add(frozenset((b, c)))
        edges.add(frozenset((c, a)))
    for e in edges:
        e = list(e)
        a, b = e[0], e[1]
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

    #print(isBlob(verts, edges, faces))



    #Printing trianlges to notepad
    #f = open('triangles.txt', 'w') 
    #f.write(f'z = {h.levels[i]} \n')
    #f.write(f'{triangles}')
    #f.close()

    #triangles_arr = triangles.reshape(triangles.shape[0], -1)
    #np.savetxt('triangle.txt', triangles_arr, fmt = '%10.5f')

    def findVEF(verts):
        """
        Given set of vertices, find the edges and faces
        """
        blobedges = set()
        blobfaces = []
        for f in faces:
            if set(f).issubset(verts):
                blobfaces.append((tuple(f)))
        for v in verts:
            for v1 in edgedict[v]:
                blobedges.add(frozenset((v, v1))) #adds edges, no duplicates
        return (verts, blobedges, blobfaces)

    def separateBlobs():
        """
        Given verts, faces from Marching Cubes alg, determines what are blobs and separates
        """
        #Doing BFS to determine contiguous blob
        blobs = []
        blobVEF = []
        unvisited = set(range(nv))
        current = set()
        while len(unvisited) != 0:
            #Each blob
            current = {unvisited.pop()}
            blob = set()
            while True:
                next = set()
                for node in current.copy():
                    next.update(edgedict[node])
                blob = blob.union(current)
                next -= blob
                unvisited -= current
                current = next
                if current == set():
                    #Blob finished
                    blob = list(blob)
                    blobs.append(blob)
                    v, e, f = findVEF(blob)
                    if isBlob(v, e, f):
                        b = Blob(dens, verts, v, e, f)
                        b.frame = frame
                        blobVEF.append(b)
                    break
        return blobVEF

    ax.clear()
    plot = None
    #plot = ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral', lw=3)
    blob_counter = 0
    blobs = separateBlobs()
    blobframe[frame] = blobs
    for b in blobs:
        #print(b.com)
        vertind, edges, faces = (b.vertind, b.edges, b.faces)
        plot = ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral', lw=3)
        vertcoords = verts[vertind]
        x = vertcoords[:,0]
        y = vertcoords[:,1]
        z = vertcoords[:,2]
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        z_min = np.min(z)
        z_max = np.max(z)
        numTrial = 100
        blobConfidence = 0
        insideTrialPoints = 0
        for numT in range(numTrial):
            xT = x_min + (x_max-x_min)*random()
            yT = y_min + (y_max-y_min)*random()
            zT = z_min + (z_max-z_min)*random()
            #print("Trial point",numT,"with",round(xT,4),round(yT,4),'for contour number %d'%j)
            trialPoint = np.array([xT,yT,zT])
            if isInside(verts, edgedict, faces, trialPoint):
                insideTrialPoints = insideTrialPoints + 1
                #print("Trial point", numT, "is INSIDE for contour number %d"%j)
                testDensity = interpTestPoint(trialPoint, dens)
                if (testDensity >= thresholdDensity):
                    #print("Interpolated point",numT,"with",round(xInterp,4),round(yInterp,4)," for Contour number %d"%j+" is INSIDE & truly a BLOB! Yeyy...")
                    blobConfidence = blobConfidence + 1
                else:
                        None
            else:
                None
                #print("Trial point", numT, " lies Outside before interpolation")
        #print(blobConfidence, insideTrialPoints)
        if insideTrialPoints > 0:
            #print(insideTrialPoints)
            confidence = blobConfidence/insideTrialPoints
        else:
            print("no points inside")
            confidence = 0
        #print("Confidence = ",confidence*100,"%")
        if (confidence > 0):
            blob_counter = blob_counter + 1
            bool0 = True
            if frame == 0: 
                previousblobs = []
            else:
                previousblobs = blobframe[frame-1]
            #b.vol = b.xrange*b.yrange*b.zrange*insideTrialPoints/numTrial
            b.vol = findVolume(verts, faces)
            closeList = []
            for prev in previousblobs:
                bool1, dist = b.is_close(prev)
                if bool1:
                    closeList.append((prev, dist))
            if len(closeList) > 0:
                #print(sorted(closeList, key = lambda x: x[1]))
                prev, prevdist = sorted(closeList, key = lambda x: x[1])[0]
                b.set_id(prev.id)
                bloblist[prev.id].append(b)
                b.vel = tuple(map(lambda x, y: (x - y)/dt, b.com, prev.com))
                bool0 = False
            if bool0:
                bloblist.append([b])
                b.set_id(len(bloblist)-1)
            #plot = ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral', lw=3)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    return plot


    def testpoints():
        """
        Plots points if calcualates as inside blob
        """
        for i, j in enumerate(coords):
            b = isInside(verts, edgedict, faces, j[:3])
            if b:
                ax.scatter(*j[:3])
        return None

    point = (1.36, -.035, 0)
    ax.scatter(*point)
    #print(type(point[1]))
    #print(isInside(verts, edgedict, faces, point))
#Graphing
fps = 5
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((2, 6, 1))
ax.view_init(elev=200, azim=250)

updateplt(50)
#updateplt(51)

#ani = animation.FuncAnimation(fig, updateplt, 100, interval=1000/fps)
writergif = animation.PillowWriter(fps=fps)
FFwriter = animation.FFMpegWriter(fps = 5)
ani.save('3Dcontourani.gif',writer=writergif)
end = time.time()
#plt.show()
print("time: " + str(end - start))

#print(verts[faces])


