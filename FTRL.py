import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

class player():
	def __init__(self, identity, n_actions,regularizer,epochs,payoff_matrix=None,payoff_func=None):
		self.id = identity
		self.n_actions = n_actions
		self.regularizer = regularizer
		self.eta_recip = calculate_eta(n_actions, regularizer,epochs)
		#print 'eta', self.eta_recip
		self.payoff_matrix = payoff_matrix
		self.payoff_func = payoff_func
		if self.payoff_matrix is not None and self.payoff_func is None:
			def mat_to_payoff(action, other_actions): #warning, assumes 2 player game!!
				return np.dot(np.dot(action,payoff_matrix),other_actions[0]) + self.regularizer(action)*self.eta_recip
			self.payoff_func = mat_to_payoff
		self.action_history = []
	def add_action(self,action):
		self.action_history.append(action)
	def get_most_recent(self):
		return self.action_history[-1]
	def get_avg_action(self):
		vec = np.zeros(self.n_actions)
		for action in self.action_history:
			vec += action
		return vec/float(len(self.action_history))

def quad_regularizer(n_vector):
	nm = np.linalg.norm(n_vector)
	nm = nm**2
	return 0.5*nm

def simplex_sample(DIM):
	samples = np.random.uniform(size=DIM+1) #this is sort of jank
	samples[0]=0
	samples[1] = 1
	samples.sort()
	prob = []
	for i in range(1,DIM+1):
		prob.append(samples[i]-samples[i-1])
	assert(sum(prob) == 1.)
	return prob

def calculate_eta(DIM, regularizer, epochs):
	#eta_recip = np.sqrt(2*epochs*1./(1.-(1./2.)))
	def L1_norm_constraint(x):
		return sum(x)-1

	starting_point = simplex_sample(DIM)
	#np.asarray([1./DIM for _ in range(DIM)])

	bounds = [(0,1) for _ in range(DIM)] #make this adaptive

	argmin= scipy.optimize.fmin_slsqp(regularizer, x0 = starting_point, eqcons = [L1_norm_constraint], bounds = bounds,iprint=0)
	rmin = regularizer(argmin)

	argmax = scipy.optimize.fmin_slsqp(lambda x: -regularizer(x), x0 = starting_point, eqcons = [L1_norm_constraint], bounds = bounds,iprint=0)
	rmax = regularizer(argmax)

	# print 'argmin, argmix', argmin, argmax
	eta_recip = np.sqrt(float(epochs)/(rmax-rmin))
	return eta_recip

def F_tilde(all_players, player, action):
	s = player.eta_recip*player.regularizer(action)
	for i in range(len(player.action_history)): 
		other_actions = []
		for p in all_players:
			if p is player:
				continue
			other_actions.append(p.action_history[i]) #list of action histories
		s += player.payoff_func(action, other_actions)
	return s

def find_opt_distribution(players, p):
	DIM = p.n_actions

	bounds = [(0,1) for _ in range(DIM)]
	def L1_norm_constraint(x):
		return sum(x)-1

	starting_point = simplex_sample(DIM)
	#starting_point = [0.1, 0.1, 0.3, 0.25, 0.25]
	#starting_point = [1./DIM for _ in range(DIM)] #maybe it's better to sample from DIM-dimensional simplex

	def cost(action): #return negative since scipy.optimze finds argmin
		return -F_tilde(players,p,action)

	opt_dist = scipy.optimize.fmin_slsqp(cost,x0=starting_point,eqcons=[L1_norm_constraint],bounds=bounds,iprint=0) #acc=1e-19
	#print 'opt', opt_dist
	opt_dist = opt_dist.clip(min=0)
	opt_dist = opt_dist/sum(opt_dist)
	#print 'starting point', starting_point
	#print 'cost opt_dist', cost(opt_dist)
	#print 'cost,[1 0 0 0 0]', cost([1, 0, 0, 0, 0])
	return opt_dist

def find_opt_distribution_project(players, p):
	DIM = p.n_actions-1
	bounds = [(0,1) for _ in range(DIM)]
	def L1_norm_constraint(x):
		return 1-sum(x)

	starting_point = simplex_sample(DIM)*float(DIM)/(DIM+1) #scaled down
	starting_point = [0.5,0.3]

	def cost(action):
		last_num = 1-sum(action)
		real_action = np.append(action,last_num)
		return -F_tilde(players, p, real_action)

	opt_dist = scipy.optimize.fmin_slsqp(cost,x0=starting_point,ieqcons=[L1_norm_constraint],bounds=bounds,iprint=0) #acc=1e-19
	opt_dist  = opt_dist.clip(min=0)
	real_opt_dist = np.append(opt_dist, 1-sum(opt_dist))
	#TO DO: make sure real_opt_dist has sum 1 and all between 0,1
	return real_opt_dist


def update_FTRL(players, tick):
	#for p in players: 
		#print p.get_avg_action()
		#print p.action_history
	for p in players:
		opt_dist = find_opt_distribution_project(players, p)
		if tick == 0:
			print p.id, opt_dist
			print 'F tilde opt_dist', F_tilde(players, p, opt_dist)
			correct = np.zeros(len(opt_dist))
			correct[p.id] = 1.
			print 'F tilde correct', F_tilde(players, p, correct)
		action = np.random.choice(len(opt_dist),p=opt_dist)
		action_vec = np.zeros(len(opt_dist))
		action_vec[action] = 1
		p.add_action(action_vec)

if __name__ == '__main__':
	epochs = 500

	prisoner_dilemma_p1 = np.array([[-1,-3],[0,-2]])
	prisoner_dilemma_p2 = np.array([[-1,-3],[0,-2]])
	mixNE_ZS_p1 = np.array([[2,-1],[-1,0]])
	mixNE_ZS_p2 = -1*np.array([[2,-1],[-1,0]])
	battle_sexes_p1 = np.array([[2,0],[0,1]])
	battle_sexes_p2 = np.array([[1,0],[0,2]])
	staghunt_p1 = np.array([[2,0],[1,1]])
	staghunt_p2 = np.array([[2,1],[0,1]])

	matrix_p1 = mixNE_ZS_p1
	matrix_p2 = mixNE_ZS_p2

	#player1 = player(n_actions = 2, payoff_matrix = matrix_p1,regularizer=quad_regularizer,epochs=epochs)
	#player2 = player(n_actions = 2, payoff_matrix = matrix_p2,regularizer=quad_regularizer,epochs=epochs)
	#players = [player1, player2]

	#congestion game	
	n_bins = 3
	N_players = 3
	def congestion(action, other_actions):
		bin_number = np.argmax(action)
		balls = 1
		for ac in other_actions: 
			if np.argmax(ac) == bin_number:
				balls += 1
		return -balls

	#def fakespan(action, other_actions):
	#	bins = np.zeros(len(action))
	#	bins = np.add(bins,action)
	#	for ac in other_actions:
	#		bins = np.add(bins,ac)
	#	maxload = np.max(bins)
	#	return -maxload

	players = []
	for p in range(N_players):
		new_player = player(identity = p, n_actions=n_bins,payoff_func=congestion,regularizer=quad_regularizer,epochs=epochs)
		#adding history to each player
		correct = np.zeros(n_bins)
		correct[new_player.id] = 1
		for _ in range(100):
			new_player.add_action(correct)
		players.append(new_player)


	#police game
	# N_players = 4
	# def payoff_func(action,other_actions): #[0=not call,1=call]
	# 	#Pr no one calls = 
	# 	pr_no_call = 1
	# 	for ac in other_actions:
	# 		pr_no_call *= ac[0]

	# 	pr_call = 1-pr_no_call
	# 	util = action[1] + action[0]*(pr_no_call*0 + pr_call*2)
	# 	return util

	# players = []
	# for _ in range(N_players):
	# 	new_player = player(n_actions=2,payoff_func=payoff_func,regularizer=quad_regularizer,epochs=epochs)
	# 	players.append(new_player)

	for i in range(epochs):	
		if i % 20 == 0:
			print 'iteration',  i
		#if i % 60 == 0:
		#	for p in players:
		#		print p.get_avg_action()
		#if i % 60 >= 5 and i%60 <= 8:
		#	print 'recent actions'
		#	for p in players:
		#		print p.get_most_recent()
		update_FTRL(players, i)
		exit()

	print 'final'
	for p in players:
		print p.get_avg_action()