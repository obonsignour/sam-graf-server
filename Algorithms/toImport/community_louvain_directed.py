# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
from __future__ import print_function
import random
import numbers
import warnings
import networkx as nx
import numpy as np
from AlgoToTest.community_status import Status

"Modified version for Diercted Louvain Algorithm by M'HAMDI Sami"
__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.

__PASS_MAX = -1
__MIN = 0.0000001
#__MIN = 1e-10


def check_random_state(seed):

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)

def merge_dicts(lst):
    result = {}
    current = 0
    for dct in lst:
        for key, value in dct.items():
            if key not in result:
                result[key] = current
            result[key] += value
        current += max(dct.values()) + 1
    return dict(sorted(result.items()))

def partition_at_level(dendrogram, level):

    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None):

    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state)
    return partition_at_level(dendo, len(dendo) - 1)


def community_detection_hierarchy(graph, level=None, partition=None,
                                  weight='weight',
                                  resolution=1.,
                                  randomize=None,
                                  random_state=None):
    result_partitions = []
    
    # Level 0
    partition_level_0 = best_partition(graph, partition,weight,resolution,randomize,random_state)
    result_partitions.append(partition_level_0)
    
    current_level = 0
    while True:
        subgraph_partitions = []
        for community_id in set(partition_level_0.values()):
            nodes_in_community = [node for node, community in partition_level_0.items() if community == community_id]
            #subgraph = graph.subgraph(nodes_in_community)
            subgraph = graph.subgraph(nodes_in_community)
            subgraph_partition = best_partition(subgraph, partition,weight,resolution,randomize,random_state)    
            subgraph_partitions.append(subgraph_partition)
            
        merged_partition = merge_dicts(subgraph_partitions)

        if merged_partition == result_partitions[-1]:  # Check if the current merged_partition is the same as the last one
            break

        result_partitions.append(merged_partition)

        if level is not None and current_level >= level:
            break

        partition_level_0 = merged_partition
        current_level += 1
    
    return result_partitions

def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None):
    #if graph.is_directed():
    #    raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    __one_level(current_graph, status, weight, resolution, random_state)
    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_state)
        new_mod = __modularity(status, resolution)
        if new_mod - mod < __MIN:
            #print("BREAK")
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
    return status_list[:]


def induced_graph(partition, graph, weight="weight"):
    ret = nx.DiGraph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret



def __one_level(graph, status, weight_key, resolution, random_state):
    """Compute one level of communities
    """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_state):
        #for node in graph.nodes():
            com_node = status.node2com[node]
            
            # Calculate total weight for both in and out degrees
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            degc_totw_out = status.gdegrees_out.get(node, 0.) / status.total_weight
            degc_totw_in = status.gdegrees_in.get(node, 0.) / status.total_weight

            neigh_communities = __neighcom(node, graph, status, weight_key)
            
            # Calculate the cost of removing the node from its current community
            remove_cost = -neigh_communities.get(com_node, 0) + resolution * (
                (status.degrees_out.get(com_node, 0.) - status.gdegrees_out.get(node, 0.)) * degc_totw_out +
                (status.degrees_in.get(com_node, 0.) - status.gdegrees_in.get(node, 0.)) * degc_totw_in)
            
            __remove(node, com_node, neigh_communities.get(com_node, 0.), status)
            
            best_com = com_node
            best_increase = 0

            # Try moving the node to different communities and calculate increase in modularity
            for com, dnc in __randomize(neigh_communities.items(), random_state):
            #for com, dnc in neigh_communities.items():
                incr = remove_cost + dnc - resolution * (
                    (status.degrees_out.get(com, 0.) * degc_totw_out) +
                    (status.degrees_in.get(com, 0.) * degc_totw_in))
                
                if incr > best_increase:
                    best_increase = incr
                    best_com = com

            # Move the node to the community that maximizes modularity increase
            __insert(node, best_com, neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True

        # Update modularity
        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break


def __neighcom(node, graph, status, weight_key):
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.) - status.gdegrees.get(node, 0.))
    status.degrees_in[com] = (status.degrees_in.get(com, 0.) - status.gdegrees_in.get(node, 0.))
    status.degrees_out[com] = (status.degrees_out.get(com, 0.) - status.gdegrees_out.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) - weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) + status.gdegrees.get(node, 0.))
    status.degrees_in[com] = (status.degrees_in.get(com, 0.) + status.gdegrees_in.get(node, 0.))
    status.degrees_out[com] = (status.degrees_out.get(com, 0.) + status.gdegrees_out.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) + weight + status.loops.get(node, 0.))


def __modularity(status, resolution):
    """
    Fast compute the modularity of the partition of the directed graph using
    status precomputed
    """
    m = float(status.total_weight)
    result = 0.

    for community in set(status.node2com.values()):
        com_degree = status.internals.get(community, 0.)
        in_degree = status.degrees_in.get(community, 0.)
        out_degree = status.degrees_out.get(community, 0.)

        if m > 0:
            result +=  com_degree * resolution / m - \
                      ((out_degree / (2. * m)) ** 2) - \
                      ((in_degree / (2. * m)) ** 2)

    return result

def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items

def __randomize2(items, random_state=None):
    if random_state == None :
        return items
    else :
        """Returns a List containing a random permutation of items"""
        # Set the random seed for reproducibility
        random.seed(random_state)
        
        # Use the random.shuffle function to shuffle the items in-place
        randomized_items = list(items)  # create a copy to avoid modifying the original list
        random.shuffle(randomized_items)
        
        return randomized_items