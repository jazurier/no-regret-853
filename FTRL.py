import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

class player():
	def __init__(self, n_actions,regularizer,epochs,payoff_matrix=None,payoff_func=None):
		self.n_actions = n_actions
		self.regularizer = regularizer
		self.eta_recip = calculate_eta(n_actions, regularizer,epochs)
		print 'eta', self.eta_recip
		self.payoff_matrix = payoff_matrix
		self.payoff_func = payoff_func
		if self.payoff_matrix is not None and self.payoff_func is None:
			def mat_to_payoff(action, other_actions):
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
	eta_recip = np.sqrt(epochs)/(rmax-rmin)
	return eta_recip

# def F_tilde(pl1,pl2,action,which_player_am_I):
# 	cumsum = eta_recip*quad_regularizer(action)
# 	for i in range(len(pl1.action_history)):
# 		if which_player_am_I == 1:
# 			cumsum += np.dot(np.dot(action,pl1.payoff),pl2.action_history[i])
# 		elif which_player_am_I == 2:
# 			cumsum += np.dot(np.dot(action,pl2.payoff),pl1.action_history[i])
# 	return cumsum

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
	starting_point = [1./DIM for _ in range(DIM)] #maybe it's better to sample from DIM-dimensional simplex

	def cost(action): #return negative since scipy.optimze finds argmin
		return -F_tilde(players,p,action)

	opt_dist = scipy.optimize.fmin_slsqp(cost,x0=starting_point,eqcons=[L1_norm_constraint],bounds=bounds,iprint=0)
	opt_dist = opt_dist.clip(min=0)
	opt_dist = opt_dist/sum(opt_dist)
	return opt_dist

def update_FTRL(players):
	for p in players:
		opt_dist = find_opt_distribution(players, p)
		action = np.random.choice(len(opt_dist),p=opt_dist)
		action_vec = np.zeros(len(opt_dist))
		action_vec[action] = 1
		p.add_action(action_vec)

if __name__ == '__main__':
	epochs = 200

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

	N_players = 4
	def payoff_func(action,other_actions): #[0=not call,1=call]
		#Pr no one calls = 
		pr_no_call = 1
		for ac in other_actions:
			pr_no_call *= ac[0]

		pr_call = 1-pr_no_call
		util = action[1] + action[0]*(pr_no_call*0 + pr_call*2)
		return util

	players = []
	for _ in range(N_players):
		new_player = player(n_actions=2,payoff_func=payoff_func,regularizer=quad_regularizer,epochs=epochs)
		players.append(new_player)

	for i in range(epochs):
		if i % 20 == 0:
			print 'iteration',  i
		update_FTRL(players)
	for p in players:
		print p.get_avg_action()