from operator import is_
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#get data
n = 21
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)

xv, yv = np.meshgrid(x, y)
z = (xv**2 + yv**2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1.0, adjustable='box')

h = plt.contour(x,y,z)
#plt.plot(xv, yv, 'bo')
contourdict = {}
#write contour segments
f = open('segments.txt', 'w') 
linelist = h.collections
for i, line in enumerate(linelist):
    seglist = line.get_segments() #array of coord
    if type(seglist) == list and seglist != []:
        seglist = np.concatenate(seglist)
    if seglist != []:
        seglist = seglist.tolist()
        #print(seglist)
        #segunique, indices = np.unique(seglist, axis = 0, return_counts=True) #remove duplicates for array
        #seglist = seglist[indices]
        #print(indices)
        None
    f.write(f'z = {h.levels[i]} \n')
    f.write(f'{seglist} \n \n')
    contourdict[i] = seglist
f.close()

#plt.show()

def intersect(segment, point):
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

#print(intersect([(1, 2), (2, 3)], (1.6, 2.5))) #testcase

def is_inside(contour, point):
    x, y = point
    count = 0
    for i in range(len(contour)-1):
        point1 = contour[i]
        point2 = contour[i+1]
        if intersect(contour[i:i+2],point):
            count += 1
    print(count)
    if count % 2 == 1:
        return True
    else:
        return False
#print(contourdict[1])
print(is_inside(contourdict[1], (-1,0)))
