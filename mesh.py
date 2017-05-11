import numpy as np 
import matplotlib.pyplot as plt
import FTRL
import sys
from FTRL import player
import plotly.plotly as py
import plotly.tools as tls
from plotly.tools import FigureFactory as FF

mesh_x_axis = np.arange(0,1.1,.1)
mesh = np.transpose([np.tile(mesh_x_axis, len(mesh_x_axis)), np.repeat(mesh_x_axis, len(mesh_x_axis))])
# print mesh
def project(u):
	try: 
		return np.array([u[0][0],u[1][0]])
	except:
		return np.array([u[0],u[2]])

def get_limit(matrix_p1, matrix_p2, bias_p1,bias_p2,epochs,nashes):
	player1 = player(identity=0, n_actions = 2, payoff_matrix = matrix_p1,regularizer=FTRL.quad_regularizer,epochs=epochs,regularizer_bias=bias_p1)
	player2 = player(identity = 1, n_actions = 2, payoff_matrix = matrix_p2,regularizer=FTRL.quad_regularizer,epochs=epochs,regularizer_bias=bias_p2)

	#bias for FTPL
	player1.add_action(bias_p1)
	player2.add_action(bias_p2)
	players = [player1, player2]
	for i in range(epochs):	
		FTRL.update_FTRL(players, tick=i, box=False,mixed=True)
		# FTRL.update_FTPL(players,epochs,tick=i)
	return FTRL.closest_nash(nashes,players) 

if __name__=="__main__":
	epochs = 100
	staghunt_p1 = np.array([[2,0],[1,1]])
	staghunt_p2 = np.array([[2,1],[0,1]])

	matrix_p1 = staghunt_p1
	matrix_p2 = staghunt_p2

	nashes = [[0,1,0,1],[1,0,1,0],[.5,.5,.5,.5]]
	nashes_tup = [tuple(x) for x in nashes]
	color = ['red','blue','green']
	nash_to_color = dict(zip(nashes_tup,color))
	x = []
	y = []
	u = []
	v = []
	for nash in nashes:
		plt.scatter(nash[0],nash[2],s=80,c=nash_to_color[tuple(nash)])
	# plt.show()
	
	for point in mesh:
		print point
		bias_p1 = np.array([point[0],1-point[0]])
		bias_p2 = np.array([point[1],1-point[1]])
		# print bias_p1
		# print bias_p2
		nash = get_limit(matrix_p1, matrix_p2, bias_p1, bias_p2,epochs,nashes)
		print nash
		point_to_plot = np.array([point[0],point[1]])
		vector_to_plot = np.subtract(project(nash),point_to_plot)
		x.append(point_to_plot[0])
		y.append(point_to_plot[1])
		u.append(vector_to_plot[0])
		v.append(vector_to_plot[1])
		# try: 
			# color = nash_to_color[tuple(nash)]
			# print color
			# plt.scatter(point[0],point[1],s=80,c=color)

		# except:
		# 	print "ERROR ON ", point
		# 	print 'error', sys.exc_info()[0]
	plt.quiver(x,y,u,v)
	plt.show()

