# You will need to install postgkyl from here:
# https://gkeyll.readthedocs.io/en/latest/install.html#postgkyl-install

import postgkyl as pg
import matplotlib.pyplot as plt
import numpy as np

DIR ="Data/"

z_slice = 10

def func_data(ionDensityData):
	ionDensityInterp = pg.data.GInterpModal(ionDensityData, 1, 'ms')
	interpGrid, ionDensityValues = ionDensityInterp.interpolate()

	# get cell center coordinates
	CCC = []
	for j in range(0,len(interpGrid)):
	    CCC.append((interpGrid[j][1:] + interpGrid[j][:-1])/2)

	x_vals = CCC[0]
	y_vals = CCC[1]
	z_vals = CCC[2]
	X, Y = np.meshgrid(x_vals, y_vals)
	ionDensityGrid = np.transpose(ionDensityValues[:,:,z_slice,0])
	return x_vals,y_vals,X,Y,ionDensityGrid

fig,ax1 = plt.subplots(1,1)
ionDensity=DIR+'c2-Lz8-midres_ion_GkM0_615.bp'
ionDensityData = pg.data.GData(ionDensity)
x_vals,y_vals,X,Y,ionDensityGrid = func_data(ionDensityData)
cp1 = ax1.contour(X, Y, ionDensityGrid, 5)
fig.colorbar(cp1)
plt.show()
