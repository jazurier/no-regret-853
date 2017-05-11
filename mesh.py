import numpy as np 
import matplotlib.pyplot as plt
import FTRL
import sys
from FTRL import player

mesh_x_axis = np.arange(0,1.1,.1)
mesh = np.transpose([np.tile(mesh_x_axis, len(mesh_x_axis)), np.repeat(mesh_x_axis, len(mesh_x_axis))])
# print mesh

def project(u):
	return np.array([u[0][0],u[1][0]])

def get_limit(matrix_p1, matrix_p2, bias_p1,bias_p2,epochs,nashes,mid_epoch_check=None):
	player1 = player(identity=0, n_actions = 2, payoff_matrix = matrix_p1,regularizer=FTRL.quad_regularizer,epochs=epochs,regularizer_bias=bias_p1)
	player2 = player(identity = 1, n_actions = 2, payoff_matrix = matrix_p2,regularizer=FTRL.quad_regularizer,epochs=epochs,regularizer_bias=bias_p2)
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
				mid_check.append(p.get_avg_action())
	
	for p in players:
		print 'avg ac', p.get_avg_action()

	return (mid_check, FTRL.closest_nash(nashes,players,avg=True))


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

	for point in mesh:
		print 'point', point
		bias_p1 = np.array([point[0],1-point[0]])
		bias_p2 = np.array([point[1],1-point[1]])
		# print bias_p1
		# print bias_p2
		avg_actions, nash = get_limit(matrix_p1, matrix_p2, bias_p1, bias_p2,epochs,nashes,mid_epoch_check=50)
		print 'avg action', avg_actions
		print 'nash', nash
		try: 
			color = nash_to_color[tuple(nash)]
			print color
			plt.scatter(point[0],point[1],s=200,c=color)
		except:
			print "ERROR ON ", point
			print 'error', sys.exc_info()[0]
	plt.show()