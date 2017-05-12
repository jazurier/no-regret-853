import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

class player():
	def __init__(self, identity, n_actions,regularizer,epochs,payoff_matrix=None,payoff_func=None,regularizer_bias=None):
		self.id = identity
		self.n_actions = n_actions
		self.regularizer_bias = regularizer_bias
		if regularizer_bias is not None:
			#regularizer_bias = [1./(self.n_actions) for _ in range(self.n_actions)]
			def biased_regularizer(n_vector):
				return regularizer(n_vector-self.regularizer_bias)
			self.regularizer = biased_regularizer
		else:
			self.regularizer = regularizer
		self.eta_recip = calculate_eta(n_actions, regularizer,epochs)
		#print 'eta', self.eta_recip
		self.payoff_matrix = payoff_matrix
		self.payoff_func = payoff_func
		if self.payoff_matrix is not None and self.payoff_func is None:
			def mat_to_payoff(action, other_actions): #warning, assumes 2 player game!!
				return np.dot(np.dot(action,payoff_matrix),other_actions[0])
			self.payoff_func = mat_to_payoff
		self.action_history = []
		self.weights = None
		self.ignore=0
	def add_action(self,action,ignore=False):
		self.action_history.append(action)
		if ignore:
			self.ignore += 1
	def get_most_recent(self):
		if len(self.action_history) > 0:
			return self.action_history[-1]
		return None
	def get_avg_action(self):
		if len(self.action_history) <= self.ignore:
			return None
		vec = np.zeros(self.n_actions)
		for action in self.action_history[self.ignore:]:
			vec += action
		return vec/float(len(self.action_history[self.ignore:]))

def entropic_regularizer(n_vector):
	s = 0
	for p in n_vector:
		if p == 0:
			continue
		s += p*np.log(p)
	return -s

def quad_regularizer(n_vector):
	nm = np.linalg.norm(n_vector)
	nm = nm**2
	return -0.5*nm

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
def find_argmin_concave(regularizer,num_dimensions):
	currentmin = 100
	currentargmin = np.zeros(num_dimensions)
	for k in range(num_dimensions):
		u = np.zeros(num_dimensions)
		u[k]=1
		if regularizer(u)<currentmin:
			currentmin = regularizer(u)
			currentargmin = u
	return currentargmin

def calculate_eta(DIM, regularizer, epochs):
	#eta_recip = np.sqrt(2*epochs*1./(1.-(1./2.)))
	#print 'hi', eta_recip
	def L1_norm_constraint(x):
		return sum(x)-1

	starting_point = simplex_sample(DIM)

	bounds = [(0,1) for _ in range(DIM)] #make this adaptive

	argmin= scipy.optimize.fmin_slsqp(regularizer, x0 = starting_point, eqcons = [L1_norm_constraint], bounds = bounds,iprint=0)
	rmin = regularizer(argmin)

	argmax = scipy.optimize.fmin_slsqp(lambda x: -regularizer(x), x0 = starting_point, eqcons = [L1_norm_constraint], bounds = bounds,iprint=0)
	rmax = regularizer(argmax)

	while (rmax-rmin)==0.:
		starting_point = simplex_sample(DIM)
		print 'eta recip is fucking up'
		print starting_point
		argmin= scipy.optimize.fmin_slsqp(regularizer, x0 = starting_point, eqcons = [L1_norm_constraint], bounds = bounds,iprint=0)
		rmin = regularizer(argmin)

		argmax = scipy.optimize.fmin_slsqp(lambda x: -regularizer(x), x0 = starting_point, eqcons = [L1_norm_constraint], bounds = bounds,iprint=0)
		rmax = regularizer(argmax)
		print argmin
		print argmax

	#print 'argmin, argmix', argmin, argmax
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

def find_opt_distribution(players, p,box=False):
	DIM = p.n_actions

	bounds = [(0,1) for _ in range(DIM)]
	def L1_norm_constraint(x):
		return sum(x)-1

	if box:
		eqcons = []
		starting_point = np.random.uniform(0,1,size=DIM) #draw from DIM hypercube
	else:
		eqcons = [L1_norm_constraint]
		starting_point = simplex_sample(DIM)

	#starting_point = simplex_sample(DIM)

	def neg_F_tilde(action): #return negative since scipy.optimze finds argmin
		return -F_tilde(players,p,action)

	opt_dist = scipy.optimize.fmin_slsqp(neg_F_tilde,x0=starting_point,eqcons=eqcons,bounds=bounds,iprint=0)
	if box:
		return opt_dist
	opt_dist = opt_dist.clip(min=0)
	opt_dist = opt_dist/sum(opt_dist)
	return opt_dist

def update_FTRL(players, tick,box=False,mixed=False):
	for p in players:
		opt_dist = find_opt_distribution(players, p, box)
		#print 'opt dist', opt_dist
		if box or mixed:
			p.add_action(opt_dist)
			continue
		#print p.id, opt_dist
		#print 'F tilde opt_dist', F_tilde(players, p, opt_dist)
		#correct = np.zeros(len(opt_dist))
		#correct[1] = 1.
		#print p.payoff_func()
		#print 'F tilde correct', F_tilde(players, p, correct)
		#print 'action history', len(p.action_history)
		#print 'average action', p.get_avg_action()
		#print 'identity', p.id
		action = np.random.choice(len(opt_dist),p=opt_dist)
		action_vec = np.zeros(len(opt_dist))
		action_vec[action] = 1
		p.add_action(action_vec)


def P_tilde(all_players, player, action):
	s=0
	for i in range(len(player.action_history)): 
		other_actions = []
		for p in all_players:
			if p is player:
				continue
			other_actions.append(p.action_history[i]) #list of action histories
		s += player.payoff_func(action, other_actions)
	return s

def update_FTPL(players, epochs, tick):
	for p in players:
		eta_recip = np.sqrt(p.n_actions*epochs)
		best_action_reward = -np.inf
		best_action = None
		perterb_vec = np.random.uniform(0,eta_recip, size=p.n_actions)
		for i in range(p.n_actions):
			action = np.zeros(p.n_actions)
			action[i] = 1
			reward = P_tilde(players, p, action)+perterb_vec[i]
			if  reward > best_action_reward:
				best_action = action
				best_action_reward = reward

		assert(best_action is not None)
		p.add_action(best_action)

def reward(all_players, player, action, tick): 
	#always assume reward on most recent tick
	other_actions = []
	for p in all_players:
		if p is player:
			continue
		other_actions.append(p.action_history[tick])
	return player.payoff_func(action, other_actions)

def EWU(players, epochs, tick):
	if tick == 0: #initialize weights 
		for p in players:
			p.weights = np.asarray([1./p.n_actions for _ in range(p.n_actions)])
	else: #update weights		
		for p in players:
			eta = np.sqrt(np.log(p.n_actions)/epochs)
			for i, w in enumerate(p.weights):
				pure_action = np.zeros(p.n_actions)
				pure_action[i] = 1
				rew = reward(players, p, pure_action, tick-1)
				p.weights[i] = p.weights[i]*np.exp(eta*rew) #should be -eta when dealing with losses THIS IS SUSS YOU SHOULD CHECK IT!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			p.weights = p.weights/sum(p.weights)

	for p in players:
		action = np.random.choice(p.n_actions,p=p.weights) ###DO WE NORMALIZE WEIGHTS AT EACH STREP??????????????????
		action_vec = np.zeros(p.n_actions)
		action_vec[action] = 1
		p.add_action(action_vec)

def MWU(players, epochs, tick):
	if tick == 0: #initialize weights 
		for p in players:
			p.weights = np.asarray([1. for _ in range(p.n_actions)])
			p.weights = p.weights/sum(p.weights)
	else: #update weights		
		for p in players:
			eta = np.sqrt(np.log(p.n_actions)/epochs) #what should we set eta to?!?!?!??!?!
			for i, w in enumerate(p.weights):
				pure_action = np.zeros(p.n_actions)
				pure_action[i] = 1
				rew = reward(players, p, pure_action, tick-1)
				p.weights[i] = p.weights[i]*(1.+eta*rew) #should be -eta when dealing with losses THIS IS SUSS YOU SHOULD CHECK IT!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			p.weights = p.weights/sum(p.weights)
	
	for p in players:
		action = np.random.choice(p.n_actions,p=p.weights) 
		action_vec = np.zeros(p.n_actions)
		action_vec[action] = 1
		p.add_action(action_vec)

def closest_nash(nashes, players,avg=False):
	actions = []
	for player in players:
		if avg:
			ac = player.get_avg_action()
		else:
			ac = player.get_most_recent()
		if ac is None:
			return None
		actions.extend(ac)
	actions = np.array(actions)

	for nash in nashes:
		nash = np.asarray(nash)
		dist = float('inf')
		closest = None
		for nash in nashes:
			d = np.linalg.norm(actions-nash)
			if d < dist: 
				dist = d 
				closest = nash
	return closest

def nash_distance(nash, players):
	actions = []
	for player in players:
		actions.extend(player.get_most_recent())
	actions = np.asarray(actions)
	nash = np.asarray(nash)
	return np.linalg.norm(nash-actions)

if __name__ == '__main__':
	epochs = 500

	#####MATRIX GAMES###########################
	prisoner_dilemma_p1 = np.array([[-1,-3],[0,-2]])
	prisoner_dilemma_p2 = np.array([[-1,-3],[0,-2]])
	mixNE_ZS_p1 = np.array([[2,-1],[-1,0]])
	mixNE_ZS_p2 = -1*np.array([[2,-1],[-1,0]])
	battle_sexes_p1 = np.array([[2,0],[0,1]])
	battle_sexes_p2 = np.array([[1,0],[0,2]])
	staghunt_p1 = np.array([[2,0],[1,1]])
	staghunt_p2 = np.array([[2,1],[0,1]])

	matrix_p1 = staghunt_p1#battle_sexes_p1 #mixNE_ZS_p1 #prisoner_dilemma_p1
	matrix_p2 = staghunt_p2 #battle_sexes_p2 #mixNE_ZS_p2 #prisoner_dilemma_p2

	# def quad_reg_bias1():
	# 	nm = np.linalg.norm(n_vector)
	# 	nm = nm**2
	# 	return -0.5*nm

	player1 = player(identity=0, n_actions = 2, payoff_matrix = matrix_p1,regularizer=quad_regularizer,epochs=epochs,regularizer_bias=[0.5,0.5])
	player2 = player(identity = 1, n_actions = 2, payoff_matrix = matrix_p2,regularizer=quad_regularizer,epochs=epochs,regularizer_bias=[0.5,0.5])
	players = [player1, player2]

	###########POLICE GAME###############
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
	##########################################

	########CONGESTION GAMES##################
	#def fakespan(action, other_actions): ###THIS IS WRONG, LOOK AT CONGESTION FOR WHY
	#	bins = np.zeros(len(action))
	#	bins = np.add(bins,action)
	#	for ac in other_actions:
	#		bins = np.add(bins,ac)
	#	maxload = np.max(bins)
	#	return -maxload

	#congestion game	
	#n_bins = 5
	#N_players = 5
	#def congestion(action, other_actions): #action is mixed, but other actions are all pure
	#	for ac in other_actions:
	#		assert(np.max(ac) == 1) #they're pure
	#	#bin_number = np.argmax(action)
	#	tot_balls = 0
	#	for mybin in range(len(action)):
	#		balls = 1
	#		for ac in other_actions: 
	#			if np.argmax(ac) == mybin:
	#				balls += 1
	#		tot_balls += action[mybin]*balls #prob you play the action * congestion if you were to play it
	#	return -tot_balls

	#adding players
	#players = []
	#for p in range(N_players):
	#	new_player = player(identity = p, n_actions=n_bins,payoff_func=congestion,regularizer=entropic_regularizer,epochs=epochs)
		#adding history to each player
		# correct = np.zeros(n_bins)
		# correct[new_player.id] = 1
		# for _ in range(100):
		# 	new_player.add_action(correct)
	#	players.append(new_player)
	#############################################

	######CHASING GAME###########################
	# def chasing_p1(action, other_actions):
	# 	assert(len(other_actions) == 1) #there's only 1 player
	# 	o_ac = other_actions[0]
	# 	return np.linalg.norm(action-o_ac)
	# def chasing_p2(action, other_actions):
	# 	assert(len(other_actions) == 1)
	# 	o_ac = other_actions[0]
	# 	return -np.linalg.norm(action-o_ac)

	#player1 = player(identity=0, n_actions=2, payoff_func=chasing_p1,regularizer=quad_regularizer, epochs=epochs)
	#player2 = player(identity=1, n_actions=2, payoff_func=chasing_p2,regularizer=quad_regularizer, epochs=epochs)
	#players = [player1,player2]
	
	nashes = [[0,1.,0.,1.],[1.,0,1.,0.],[0.5,0.5,0.5,0.5]]
	colors = ['blue','green','red']
	###FOR ALL GAMES#############################
	for i in range(epochs):	
		if i % 20 == 0:
			print 'iteration',  i
		if i % 60 == 0:
			for p in players:
				print 'most recent', p.get_most_recent()
			print closest_nash(nashes, players)
		if i == 100:
			cn = closest_nash(nashes,players)
			conv_norm = nash_distance(cn, players)

		#if i % 20 >= 5 and i%20 <= 10: #to check for correlated/coarse-correlated equilibria
		#	print 'recent actions'
		#	for p in players:
		#		print p.get_most_recent()
		update_FTRL(players, tick=i, box=False,mixed=True)
		#update_FTPL(players, epochs=epochs,tick=i)
		#EWU(players, epochs=epochs, tick=i)
		#MWU(players, epochs=epochs, tick=i)
		#if i > 10:
		#	exit()

	print 'final'
	for p in players:
		print 'average action', p.get_avg_action()

	print closest_nash(nashes,players)