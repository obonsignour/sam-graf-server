# coding=utf-8

class Status(object):
    """
    To handle several data in one struct.

    Could be replaced by named tuple, but don't want to depend on python 2.6

    self.node2com: This attribute is a dictionary that maps each node to its corresponding 
    community. For each node in the graph, node2com[node] provides the community to which 
    that node belongs.

    self.total_weight: This attribute represents the total weight of the graph. It is the sum of 
    the weights assigned to edges in the graph.

    self.degrees: This attribute is a dictionary that stores the degree of each community. 
    For each community, degrees[com] provides the sum of the degrees of all nodes in that 
    community.

    self.gdegrees: This attribute is a dictionary that stores the degree of each node. For each 
    node, gdegrees[node] provides the degree of that node.

    self.internals: This attribute is a dictionary that stores the internal weight of each 
    community. Internal weight refers to the sum of edge weights within a community. For each 
    community, internals[com] provides the internal weight of that community.

    self.loops: This attribute is a dictionary that stores the loop weight of each node. Loop 
    weight refers to the weight of self-loops in the graph. For each node, loops[node] provides 
    the loop weight of that node.
    
    Total Edges Incident to Community C=∑_(node in C) degree of node
    Number of Incoming Arcs to Community C=∑_(node in C) in-degree of node
    Number of Outgoing Arcs from Community C=∑_(node in C) out-degree of node
    result += status.internals.get(community, 0.) * resolution / links -  ((status.degrees.get(community, 0.) / (2. * links)) ** 2)

    """
    
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    degrees_in = {}
    degrees_out = {}
    gdegrees = {}
    gdegrees_in = {}
    gdegrees_out = {}

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.degrees_in = dict([])
        self.degrees_out = dict([])
        self.gdegrees = dict([])
        self.gdegrees_in = dict([])
        self.gdegrees_out = dict([])
        self.internals = dict([])
        self.loops = dict([])

    def __str__(self):
        return ("node2com : " + str(self.node2com) 
                + " degrees : " + str(self.degrees) 
                + " degrees_in : "+ str(self.degrees_in)
                + " degrees_out : "+ str(self.degrees_out)
                + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.degrees_in = self.degrees_in.copy()
        new_status.degrees_out = self.degrees_out.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.gdegrees_in = self.gdegrees_in.copy()
        new_status.gdegrees_out = self.gdegrees_out.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.degrees_in = dict([])
        self.degrees_out = dict([])
        self.gdegrees = dict([])
        self.gdegrees_in = dict([])
        self.gdegrees_out = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                #print(type(graph))
                deg_in = float(graph.in_degree(node, weight=weight))
                deg_out = float(graph.out_degree(node, weight=weight))
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.degrees_in[count] = deg_in
                self.degrees_out[count] = deg_out
                self.gdegrees[node] = deg
                self.gdegrees_in[node] = deg_in
                self.gdegrees_out[node] = deg_out
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                deg_in = float(graph.in_degree(node, weight=weight))
                deg_out = float(graph.out_degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.degrees_in[com] = self.degrees_in.get(com, 0) + deg_in
                self.degrees_out[com] = self.degrees_out.get(com, 0) + deg_out
                self.gdegrees[node] = deg
                self.gdegrees_in[node] = deg_in
                self.gdegrees_out[node] = deg_out
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc