import networkx as nx
from networkx.generators.random_graphs import barabasi_albert_graph
import numpy as np
import pylab as plt
#from pdb import set_trace as st

#global R
#global Q
n_iter = 5000
goal = 23

LINK_FRACTION = 6
MATRIX_SIZE = 100

def available_actions(R, state):
    current_state_row = R[state, ]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act


def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act, 1))
    return next_action


def update(R, Q, current_state, action, gamma):
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    print('max_value', R[current_state, action] + gamma * max_value)
    if (np.max(Q) > 0):
        return(np.sum(Q / np.max(Q) * 100))
    else:
        return (0)

# First we generate a random graph holding a power law link distribution. This
# has the aim of keeping entropy sourced phenomena in mind.

Graph = barabasi_albert_graph(MATRIX_SIZE, int(MATRIX_SIZE / LINK_FRACTION))
pos = nx.spring_layout(Graph)
nx.draw_networkx_nodes(Graph, pos)
nx.draw_networkx_edges(Graph, pos)
nx.draw_networkx_labels(Graph, pos)
plt.show()

R = nx.adjacency_matrix(Graph).todense() - 1
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
points_list = [t for t in zip(*np.where(R == 0))]

for point in points_list:
    print(point)
    if point[1] == goal:
        R[point] = 100
    else:
        R[point] = 0

    if point[0] == goal:
        R[point[::-1]] = 100
    else:
    # reverse of point
        R[point[::-1]] = 0

# add goal point round trip
R[goal, goal] = 100
gamma = 0.8
initial_state = 1

available_act = available_actions(R, initial_state)
action = sample_next_action(available_act)
update(R, Q, initial_state, action, gamma)

scores = []
for i in range(n_iter):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)
    print('Score:', str(score))

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

steps = [current_state]

while current_state != goal:
    next_step_index = np.where(Q[current_state, ] ==
                                            np.max(Q[current_state, ]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
        steps.append(next_step_index)
        current_state = next_step_index

print("Most efficient path:")

print(steps)

plt.plot(scores)
plt.show()