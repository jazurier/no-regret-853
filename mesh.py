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

def get_limit(matrix_p1, matrix_p2, bias_p1,bias_p2,epochs,nashes,mid_epoch_check=None):
	
	try:
		player1 = player(identity=0, n_actions = 2, payoff_matrix = matrix_p1,regularizer=FTRL.quad_regularizer,epochs=epochs,regularizer_bias=bias_p1)
		player2 = player(identity = 1, n_actions = 2, payoff_matrix = matrix_p2,regularizer=FTRL.quad_regularizer,epochs=epochs,regularizer_bias=bias_p2)
	except:
		return (None, None)
	
	mid_check = None
	#bias for FTPL
	#for _ in range(int(epochs/2.)):
	#	player1.add_action(bias_p1,ignore=True)
	#	player2.add_action(bias_p2,ignore=True)
	players = [player1, player2]
	for i in range(epochs):	
		FTRL.update_FTRL(players, tick=i, box=False,mixed=True)
		#FTRL.update_FTPL(players,epochs,tick=i)
		if i == mid_epoch_check:
			mid_check = []
			for p in players:
				mid_check.append(p.get_most_recent())
	
	last_action = []
	for p in players:
		print 'most recent ac', p.get_most_recent()
		last_action.append(p.get_most_recent())

	return (mid_check, last_action, FTRL.closest_nash(nashes,players,avg=True))


if __name__=="__main__":
	epochs = 50
	staghunt_p1 = np.array([[2,0],[1,1]])
	staghunt_p2 = np.array([[2,1],[0,1]])
	stag_nashes = [[0,1,0,1],[1,0,1,0],[.5,.5,.5,.5]]
	prisoner_dilemma_p1 = np.array([[-1,-3],[0,-2]])
	prisoner_dilemma_p2 = np.array([[-1,-3],[0,-2]])
	prisoner_dilemma_nashes = [[0,1,0,1]]
	mixNE_ZS_p1 = np.array([[2,-1],[-1,0]])
	mixNE_ZS_p2 = -1*np.array([[2,-1],[-1,0]])
	mixNE_ZS_nashes = [[.25,.75,.25,.75]]
	battle_sexes_p1 = np.array([[2,0],[0,1]])
	battle_sexes_p2 = np.array([[1,0],[0,2]])
	battle_sexes_nashes = [[0,1,0,1],[1,0,1,0],[2./3,1./3,1./3,2./3]]
	matrix_p1 = staghunt_p1
	matrix_p2 = staghunt_p2
	nashes = stag_nashes
 	# matrix_p1 = prisoner_dilemma_p1
 	# matrix_p2 = prisoner_dilemma_p2
 	# matrix_p1 = mixNE_ZS_p1
 	# matrix_p2 = mixNE_ZS_p2
 	# nashes = mixNE_ZS_nashes
 	#matrix_p1 = battle_sexes_p1
 	#matrix_p2 = battle_sexes_p2
 	#nashes = battle_sexes_nashes
	nashes_tup = [tuple(x) for x in nashes]
	color = ['red','blue','green']
	color = color[:len(nashes)]
	nash_to_color = dict(zip(nashes_tup,color))


	x = []
	y = []
	u = []
	v = []
	for nash in nashes:
		plt.scatter(nash[0],nash[2],s=200,c=nash_to_color[tuple(nash)])
	# plt.show()
	
	for point in mesh:
		print 'point', point
		bias_p1 = np.array([point[0],1-point[0]])
		bias_p2 = np.array([point[1],1-point[1]])
		# print bias_p1
		# print bias_p2
		mid_epoch_actions, last_actions, nash = get_limit(matrix_p1, matrix_p2, bias_p1, bias_p2,epochs,nashes,mid_epoch_check=50)
		print 'avg action', avg_actions
		print 'nash', nash
		try: 
			color = nash_to_color[tuple(nash)]
			print color
			plt.scatter(point[0],point[1],s=80,c=color)
		except:
			print "ERROR ON ", point
			print 'error', sys.exc_info()[0]

		point_to_plot = np.array([point[0],point[1]])
		vector_to_plot = np.subtract(project(last_actions),point_to_plot)
		x.append(point_to_plot[0])
		y.append(point_to_plot[1])
		u.append(vector_to_plot[0])
		v.append(vector_to_plot[1])

	plt.quiver(x,y,u,v)
	plt.show()