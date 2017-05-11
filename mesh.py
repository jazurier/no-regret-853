import numpy as np 
import matplotlib.pyplot as plt
mesh_x_axis = np.arange(0,1.1,.1)
mesh = np.transpose([np.tile(mesh_x_axis, len(mesh_x_axis)), np.repeat(mesh_x_axis, len(mesh_x_axis))])
# print mesh
def get_limit(a,b):
	# you should have this call the Nash eq thing
	return 'ro'
for point in mesh:
	bias_p1 = np.array([point[0],1-point[0]])
	bias_p2 = np.array([point[1],1-point[1]])
	# print bias_p1
	# print bias_p2
	color = get_limit(bias_p1,bias_p2)

	plt.plot(point[0],point[1],color)
plt.show()