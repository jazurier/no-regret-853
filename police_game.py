import numpy as np 
import matplotlib.pyplot as plt
import FTRL
import sys
from FTRL import player
# Police game social welfare plot idea.
# plot num players initially calling vs. social welfare of the eventual equilibrium.

'''
2p police game:

[(1,1),(1,2)]
[(2,1),(0,0)]
Nashes: .5 .5, 0 1, 1 0






'''
'''
My understanding of the police game: Each player's strategy is some 0<=p<=1. payoff -1 for calling, 2 for someone calling.
Social welfare: 2n*Pr(someone calls)-E[number of callers]
OPT: 2n-1
calling probabilities p1,p2,...,pn
SW: 2n*prod(1-pk) - sum(pk)

'''
num_players = 10

def create_biases(num_players,num_init_callers):
	assert num_init_callers <= num_players
	biases = []
	for k in range(num_players):
		u = np.array([0,0])
		if k < num_init_callers:
			u[0]=1
		else:
			u[1]=1
		biases.append(u)
	return biases
def social_welfare(action_list):
	numplayers = len(action_list)
	sw = 0.
	sw += 2*numplayers
	for _ in range(numplayers):
		sw *= action_list[_][1]
	sw = 2*numplayers - sw
	for _ in range(numplayers):
		sw -= action_list[_][0]
	return sw
def opt_social_welfare(num_players):
	return 2*num_players - 1

def run_nash(biases):
	return biases

k_array = []
sw_array = []	
for k in range(num_players+1):
	biases = create_biases(num_players,k)
	resulting_action_list = run_nash(biases)
	k_array.append(k)
	sw_array.append(social_welfare(resulting_action_list)*1./opt_social_welfare(num_players))
plt.plot(k_array,sw_array)
plt.show()
