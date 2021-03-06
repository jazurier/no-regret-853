import numpy as np 
import matplotlib.pyplot as plt
import FTRL
import sys
from FTRL import player


mesh_x_axis = np.arange(0,1.001,.1)
meshh = np.transpose([np.tile(mesh_x_axis, len(mesh_x_axis)), np.repeat(mesh_x_axis, len(mesh_x_axis))])
mesh = []
mesh = meshh
# for tupled in meshh:
# 	if tupled[0]+tupled[1]==1:
# 		mesh.append(tupled)
# print mesh

def project(u):
	try: 
		return np.array([u[0][0],u[1][0]])
	except:
		return np.array([u[0],u[2]])

def get_limit(matrix_p1, matrix_p2, bias_p1,bias_p2,epochs,nashes,mid_epoch_check=None):
	
	
	player1 = player(identity=0, n_actions = 2, payoff_matrix = matrix_p1,regularizer=FTRL.quad_regularizer,epochs=epochs,regularizer_bias=bias_p1)
	player2 = player(identity = 1, n_actions = 2, payoff_matrix = matrix_p2,regularizer=FTRL.quad_regularizer, epochs=epochs,regularizer_bias=bias_p2)

	mid_check = None
	#bias for FTPL
	#for _ in range(int(epochs/2.)):
	#	player1.add_action(bias_p1,ignore=True)
	#	player2.add_action(bias_p2,ignore=True)
	players = [player1, player2]
	approach = []
	for i in range(epochs):	
		FTRL.update_FTRL(players, tick=i, box=False,mixed=True,matrix_game_speedup=True)
	#	if i > 1:
	#		exit() 
		#FTRL.update_FTPL(players,epochs,tick=i)
		
		#print 'iteration', i
		approach.append(project([list(players[0].get_most_recent()),list(players[1].get_most_recent())]))
		
		#for p in players:
		#	print p.get_most_recent(),
	#	print 'iteration', i
		if i == mid_epoch_check:
			mid_check = []
			for p in players:
				mid_check.append(p.get_most_recent())
	#print approach

	plt.scatter([x[0] for x in approach],[x[1] for x in approach])
	plt.show()
	exit()
	
	last_action = []
	for p in players:
		print 'most recent ac', p.get_most_recent()
		last_action.append(p.get_most_recent())

	return (mid_check, last_action, FTRL.closest_nash(nashes,players,avg=False))


if __name__=="__main__":
	epochs = 100
	SK = 4 #1.5
	staghunt_p1 = np.array([[SK,0],[1,1]])
	staghunt_p2 = np.array([[SK+1,0],[1,1]])
	stag_nashes = [[0,1,0,1],[1,0,1,0],[0.2, 0.8,0.25,0.75]]
	#stag_nashes = [[0,1,0,1],[1,0,1,0],[1./SK, 1.-1./SK,1./SK,1.-1./SK]]
	prisoner_dilemma_p1 = np.array([[-1,-3],[0,-2]])
	prisoner_dilemma_p2 = np.array([[-1,-3],[0,-2]])
	prisoner_dilemma_nashes = [[0,1,0,1]]
	mixNE_ZS_p1 = np.array([[2,-1],[-1,0]])
	mixNE_ZS_p2 = -1*np.array([[2,-1],[-1,0]])
	mixNE_ZS_nashes = [[.25,.75,.25,.75]]
	BK = 2
	battle_sexes_p1 = np.array([[BK,0],[0,1]])
	battle_sexes_p2 = np.array([[1,0],[0,BK]])
	battle_sexes_nashes = [[0,1,0,1],[1,0,1,0],[BK/(BK+1.),1./(BK+1.),1./(BK+1.),BK/(BK+1.)]]
	#print battle_sexes_nashes
	
	# police_matrix_p1 = np.array([[0,2],[1,1.5]])
	# police_matrix_p2 = np.array([[0,2],[1,1]])
	# police_nashes = [[0,1,1,0],[1,0,0,1],[0.5, 0.5, 1./3, 2./3]]
	
	police_matrix_p1 = np.array([[0,2],[1,1.75]])
	police_matrix_p2 = np.array([[0,2],[1,1]])
	police_nashes = [[0,1,1,0],[1,0,0,1],[0.5, 0.5, 1./5, 4./5]]
	#exit(0)
	
	# matrix_p1 = staghunt_p1
	# matrix_p2 = staghunt_p2
	# nashes = stag_nashes
 	# matrix_p1 = prisoner_dilemma_p1
 	# matrix_p2 = prisoner_dilemma_p2
 	# matrix_p1 = mixNE_ZS_p1
 	# matrix_p2 = mixNE_ZS_p2
 	# nashes = mixNE_ZS_nashes
 	# matrix_p1 = battle_sexes_p1
 	# matrix_p2 = battle_sexes_p2
 	# nashes = battle_sexes_nashes
	
	matrix_p1 = police_matrix_p1
 	matrix_p2 = police_matrix_p2
 	nashes = police_nashes
	
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
	# # plt.show()
	
	specpoint = [0.8,0.5]
	#specpoint = [0.56815,0.41333333333333333333]
	spb1 = [specpoint[0],1.-specpoint[0]]
	spb2 = [specpoint[1],1.-specpoint[1]]
	mid_epoch_actions, last_actions, nash = get_limit(matrix_p1, matrix_p2,spb1,spb2,epochs,nashes,mid_epoch_check=50)
	print last_actions
	print nash
	exit()
	
	for point in mesh:
		print 'point', point
		bias_p1 = np.array([point[0],1-point[0]])
		bias_p2 = np.array([point[1],1-point[1]])
		# print bias_p1
		# print bias_p2
		mid_epoch_actions, last_actions, nash = get_limit(matrix_p1, matrix_p2, bias_p1, bias_p2,epochs,nashes,mid_epoch_check=50)
		#print 'mid epoch action', mid_epoch_actions
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

	# plt.quiver(x,y,u,v)
	plt.quiver(x,y,u,v,angles='xy', scale_units='xy', scale=1.)
	plt.show()