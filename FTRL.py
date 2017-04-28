import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
prisoner_dilemma_p1 = np.array([[-1,-3],[0,-2]])
prisoner_dilemma_p2 = np.array([[-1,-3],[0,-2]])
mixNE_ZS_p1 = np.array([[2,-1],[-1,0]])
mixNE_ZS_p2 = -1*np.array([[2,-1],[-1,0]])
battle_sexes_p1 = np.array([[2,0],[0,1]])
battle_sexes_p2 = np.array([[1,0],[0,2]])
staghunt_p1 = np.array([[2,0],[1,1]])
staghunt_p2 = np.array([[2,1],[0,1]])

matrix_p1 = staghunt_p1
matrix_p2 = staghunt_p2
# print mixNE_ZS_p2
# matrix_p1 = prisoner_dilemma_p1
# matrix_p2 = prisoner_dilemma_p2
# matrix_p1 = mixNE_ZS_p1
# matrix_p2 = mixNE_ZS_p2
epochs = 2000
tolerance = .001

#eta_recip = np.sqrt(2*epochs*1./(1.-(1./2.)))

def quad_regularizer(n_vector):
	# nm = np.linalg.norm(n_vector)
	nm = np.linalg.norm(n_vector-np.array([0.65,0]))
	nm = nm**2
	return 0.5*nm


def L1_norm_constraint(x):
	return sum(x)-1

def starting_point():
	p = np.random.rand()
	return np.array([p,1-p])

bounds = [(0,1) for _ in range(2)] #make this adaptive

quad_argmin= scipy.optimize.fmin_slsqp(quad_regularizer, x0 = starting_point(), eqcons = [L1_norm_constraint], bounds = bounds)
quad_min = quad_regularizer(quad_argmin)

quad_argmax = scipy.optimize.fmin_slsqp(lambda x: -quad_regularizer(x), x0 = starting_point(), eqcons = [L1_norm_constraint], bounds = bounds)
quad_max = quad_regularizer(quad_argmax)

eta_recip = np.sqrt(epochs)/(quad_max-quad_min)

class player():

	def __init__(self, payoff_matrix):
		self.payoff = payoff_matrix
		self.action_history = []
	def add_action(self,action):
		self.action_history.append(action)
	def get_dec(self):
		return self.action_history[-1]
	def get_avg(self):
		vec = self.action_history[0]*1.
		vec = vec - self.action_history[0]
		for vector in self.action_history:
			vec += vector
		return vec*1./(len(self.action_history))

p1 = player(matrix_p1)
p2 = player(matrix_p2)

def F_tilde(pl1,pl2,action,which_player_am_I):
	cumsum = eta_recip*quad_regularizer(action)
	for i in range(len(pl1.action_history)):
		if which_player_am_I == 1:
			cumsum += np.dot(np.dot(action,pl1.payoff),pl2.action_history[i])
		elif which_player_am_I == 2:
			cumsum += np.dot(np.dot(action,pl2.payoff),pl1.action_history[i])
	return cumsum

def argmax_function(first_player,second_player,which_player_am_I,t):
	def opt(x):
		return -F_tilde(first_player,second_player,x,which_player_am_I)
	
	def L1_norm_constraint(x):
		return sum(x)-1

	bounds = [(0,1) for _ in range(2)] #make this adaptive
	p = np.random.rand()
	starting_point = np.array([p,1-p])
	amin = scipy.optimize.fmin_slsqp(opt, x0 = starting_point, eqcons = [L1_norm_constraint], bounds = bounds,iprint=0)
	# if t==0:
	# 	print amin,"TIME ZERO"
	amin = amin.clip(min=0)
	amin = amin/sum(amin)
	#assert(sum(amin)==1)
	newact = np.random.choice(2, p=amin)
	act = np.zeros(2)
	act[newact] = 1
	#assert(sum(act) == 1)
	# print act,amin
	# print opt(amin)
	# print opt([0.25,0.75])
	return act
	# return amin

	# return the action * that minimizes F_tilde(pl1,pl2,*,which_player_am_I)
def update_FTRL(pl1,pl2,t):
	newp1act = argmax_function(pl1,pl2,1,t)
	newp2act = argmax_function(pl1,pl2,2,t)
	pl1.add_action(newp1act)
	pl2.add_action(newp2act)
	return pl1,pl2

def find_Nash(p1,p2):
	t=0
	k=0
	for i in range(k):
		# p1.add_action(np.array([2./3,1./3]))
		# p2.add_action(np.array([1./3,2./3]))
		p1.add_action(np.array([0,1]))
		p2.add_action(np.array([0,1]))
		t+=1
	while t<epochs:
		if t%50 == 0:
			print 'TIME '
			print t
			# print 'TIME '
			# print t
			# print 'TIME '
			# print t
		p1,p2 = update_FTRL(p1,p2,t)
		t+=1
	# print p1.action_history
	return p1.get_dec(),p2.get_dec(),'AVG',p1.get_avg(),p2.get_avg()
print find_Nash(p1,p2)
