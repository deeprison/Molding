import numpy as np

class Node:
    def __init__(self, state, name):
        self.state = state
        self.name = name
        self.edges = []

    def is_leaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True

class Edge:

    def __init__(self, in_node, out_node, prob, action):
        self.name = in_node.name + '-' + out_node.name
        self.in_node = in_node
        self.out_node = out_node
        self.action = action

        self.stats = {'N':0, 'W':0, 'Q':0, 'P':prob}


class Graph:

    def __init__(self, current_node, cpuct, epsilon=0.2, alpha=0.8):
        self.current_node = current_node
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(current_node)
        self.epsilon = epsilon
        self.alpha = alpha

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self, env):

        breadcrumbs = []
        current_node = self.current_node

        done = False
        value = 0
        
        check_inf_loof = 0

        while not current_node.is_leaf() and not done:

            max_QU = -9999
            if current_node == self.current_node:
                epsilon = self.epsilon
                nu = np.random.dirichlet([self.alpha]* len(current_node.edges))
            else:
                epsilon = 0
                nu = np.zeros(len(current_node.edges))

            Nb = 0
            for (action, edge) in current_node.edges:
                Nb += edge.stats['N']

            # print('current_node', current_node)
            for idx, (action, edge) in enumerate(current_node.edges):

                U = self.cpuct * ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(Nb) / (1 + edge.stats['N'])
                Q = edge.stats['Q']

                # print(f"U : {U}, Q : {Q}, N : {edge.stats['N']}, P : {edge.stats['P']}")

                if Q + U > max_QU:
                    max_QU = Q+U
                    simulation_action = action
                    simulation_edge = edge
            
            next_state, value, done, _ = env.step(simulation_action)
            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)
            
            check_inf_loof += 1
            if check_inf_loof > 5000:
                print(current_node)
                print(current_node.edges)
                print(current_node.is_leaf())
                print(simulation_action)
                print(simulation_edge.name)
                print(simulation_edge.out_node)
                
                raise "inf loof"

        return current_node, value, done, breadcrumbs


    def back_fill(self, value, breadcrumbs):

        for edge in breadcrumbs:
            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def add_node(self, node):
        self.tree[node.name] = node