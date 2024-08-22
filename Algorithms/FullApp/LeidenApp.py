import leidenalg as la
import igraph as ig
#from neo4j_connector_igraph import Neo4jGraph
from neo4j import GraphDatabase
import time
import openai
import argparse
import concurrent.futures

from Algorithms.toImport.neo4j_connector_igraph import Neo4jGraph
import os
from dotenv import load_dotenv

start_time = time.time()

load_dotenv()

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    URI = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    database_name = os.environ.get("NEO4J_DATABASE")
except KeyError:
    print("Error: OpenAI API key or Neo4j credentials not found in environment variables.")
    exit(1)

# Name of the properties used in the similarity function
properties_of_interest = ['Type', 'Level', 'External', 'Method'] #, 'Hidden'

# A similarity function based on node properties
def similarity(node1, node2, properties_of_interest):
    similarity_sum = 0
    for prop in properties_of_interest:
        if node1[prop] == node2[prop]:
            similarity_sum += 1

    # Calculate overall similarity
    overall_similarity = similarity_sum / len(properties_of_interest)

    return overall_similarity

# Add edges with weights based on similarity
def add_semantic_as_weight(G):
    for edge in G.es():
        u = edge.source
        v = edge.target
        weight = similarity(G.vs[u], G.vs[v], properties_of_interest)
        edge['weight'] = weight

# Sets the weight attribute to 1 for all edges in the igraph graph G that do not already have a weight attribute.
def set_default_weight(G):
    for edge in G.es:
        if 'weight' not in edge.attributes():
            edge['weight'] = 1

def convert_leiden_format(leiden_partition):
    new_format_leiden_partition = {}
    for vertex, cluster in enumerate(leiden_partition.membership):
        new_format_leiden_partition[vertex] = cluster
    return new_format_leiden_partition

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

def community_detection_hierarchy(graph, level=None):
    result_partitions = []
    hierarchy_tree = {}

    # Level 0
    partition_level_0 = convert_leiden_format(la.find_partition(graph, la.ModularityVertexPartition))
    partition_level_0 = {graph.vs['id'][k]: v for k, v in partition_level_0.items()}
    #print(f"HHAA :  {partition_level_0}")
    result_partitions.append(partition_level_0)
    hierarchy_tree[0] = {community_id: [] for community_id in set(partition_level_0.values())}
    
    current_level = 1
    while True:
        subgraph_partitions = []
        subgraph_tree = {}
        for community_id in set(partition_level_0.values()):
            nodes_in_community = graph.vs.select(lambda v: v["id"] in partition_level_0 and partition_level_0[v["id"]] == community_id)
            #print(f"nodes_in_community hierar : {nodes_in_community}")
            nodes_in_community = [v.index for v in nodes_in_community]
            #print(f"nodes_in_community hierar : {nodes_in_community}")
            subgraph = graph.induced_subgraph(nodes_in_community)
            subgraph_partition = convert_leiden_format(la.find_partition(subgraph, la.ModularityVertexPartition))
            subgraph_partition = {subgraph.vs['id'][k]: v for k, v in subgraph_partition.items()}
            subgraph_partitions.append(subgraph_partition)
            subgraph_tree[community_id] = list(set(subgraph_partition.values()))
        
        merged_partition = merge_dicts(subgraph_partitions)

        if merged_partition == result_partitions[-1]:
            break
        
        result_partitions.append(merged_partition)
        hierarchy_tree[current_level] = subgraph_tree
        
        if level is not None and current_level >= level:
            break
        
        partition_level_0 = merged_partition
        current_level += 1
    
    return result_partitions, hierarchy_tree

# Fonction pour générer un nom de communauté à partir d'une liste de termes
def generate_community_name(terms):
    #prompt = f"Generate a name for the community based on the following terms:\n{', '.join(terms)}"
    #prompt = f"Generate and return only one concise and meaningful name without symbols and of maximum 30 characters, grouping the following terms:\n{', '.join(terms)}"
    prompt = f"""
    CONTEXT: Identification of functional processes based on coding objetcs names.
    TASK: Generate a meaningful group of word summurazing the list of input words.
    GUIDELINES: Returned answer must be without symbols and must have 1-to-4-word separated with space.
    EXAMPLES: 
    - Prdouct Management Functions
    - Error Handling Process
    - Discount Management Process
    INPUT: 
    {', '.join(terms)}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            #{"role": "system", "content": "You are a helpful assis tant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5,
        temperature=0.7
    )
    return response.choices[0].message["content"].strip()

def get_most_important_nodes(graph, node_list, metric='betweenness', top_n=10):
    """
    Identifie les nœuds les plus importants d'une liste dans un graphe igraph selon une métrique.

    :param graph: Graphe igraph.
    :param node_list: Liste des identifiants de nœuds.
    :param metric: La métrique à utiliser pour déterminer l'importance des nœuds ('degree', 'closeness', 'betweenness', etc.).
    :param top_n: Le nombre de nœuds les plus importants à retourner.
    :return: Liste des nœuds les plus importants.
    """
    if len(node_list)>top_n : 
        # Calcul de la centralité selon la métrique spécifiée
        if metric == 'degree':
            centrality = {node: graph.degree(node) for node in node_list}
        elif metric == 'betweenness':
            centrality = graph.betweenness()
        elif metric == 'eigenvector':
            centrality = graph.evcent()
        elif metric == 'pagerank':
            centrality = graph.pagerank(node_list)
        else:
            raise ValueError("Métrique non reconnue. Choisissez parmi 'degree', 'pagerank', 'betweenness', 'eigenvector'.")

        # Mapping des centralités avec les nœuds
        centrality_dict = {v.index: centrality[v.index] for v in graph.vs if v.index in node_list}

        # Tri des nœuds par ordre décroissant de centralité
        sorted_nodes = sorted(centrality_dict, key=centrality_dict.get, reverse=True)

        # Retourner les top_n nœuds les plus importants
        return sorted_nodes[:top_n]
    
    else :
        return node_list


def generate_name_for_community(G, community_id, community_nodes_ids):
    community_nodes_index = [v.index for v in community_nodes_ids]
    important_community_nodes_index = get_most_important_nodes(G, community_nodes_index, metric='degree')
    important_community_nodes_names = [G.vs[node]['Name'] for node in important_community_nodes_index]
    
    if len(important_community_nodes_names) < 2:
        return community_id, important_community_nodes_names[0]
    else:
        # Use cached version to reduce API calls
        return community_id, generate_community_name(important_community_nodes_names)

def generate_name_for_current_level_community(G, current_level, next_level, community_id, communities_at_current_level, dendrogram, CommunitiesNames):
    # Get nodes in the current level community
    nodes_in_current_community = [node for node, cid in communities_at_current_level.items() if cid == community_id]
    # Get subcommunity IDs at the next level
    subcommunity_ids = set(dendrogram[next_level][node] for node in nodes_in_current_community if node in dendrogram[next_level])
    
    if len(subcommunity_ids) < 2:
        # Retrieve names of these subcommunities
        subcommunity_names = [CommunitiesNames[next_level][sub_id] for sub_id in subcommunity_ids]
        return community_id, subcommunity_names[0]

    if len(subcommunity_ids) > 1000:
        # Count the number of nodes in each subcommunity
        subcommunity_node_counts = {}
        for node in nodes_in_current_community:
            sub_id = dendrogram[next_level][node]
            if sub_id not in subcommunity_node_counts:
                subcommunity_node_counts[sub_id] = 0
            subcommunity_node_counts[sub_id] += 1
        # Sort subcommunities by node count and select the top 1000
        sorted_subcommunity_ids = sorted(subcommunity_node_counts, key=subcommunity_node_counts.get, reverse=True)[:1000]
        subcommunity_names = [CommunitiesNames[next_level][sub_id] for sub_id in sorted_subcommunity_ids]
        # Generate the name for the current level community
        community_name = generate_community_name(subcommunity_names)
        return community_id, community_name
    
    else:    
        # Retrieve names of these subcommunities
        subcommunity_names = [CommunitiesNames[next_level][sub_id] for sub_id in subcommunity_ids]
        # Generate the name for the current level community
        community_name = generate_community_name(subcommunity_names)
        return community_id, community_name

def communitiesNamesThread(G, dendrogram):
    CommunitiesNames = {}

    # Start from the highest level
    level = len(dendrogram) - 1
    communities_at_level = dendrogram[level]
    level_community_names = {}

    # Using parallel processing for name generation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_community = {
            executor.submit(generate_name_for_community, G, community_id, G.vs.select(lambda v: v["id"] in communities_at_level and communities_at_level[v["id"]] == community_id)): community_id
            for community_id in set(communities_at_level.values())
        }
        for future in future_to_community:
            community_id, community_name = future.result()
            level_community_names[community_id] = community_name

    CommunitiesNames[level] = level_community_names
    print(f"Communities names at level {level}: DONE")

    # Process lower levels
    for current_level in range(len(dendrogram) - 2, -1, -1):
        print(f"Processing level: {current_level}")
        next_level = current_level + 1
        communities_at_current_level = dendrogram[current_level]
        level_community_names = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_community = {
                executor.submit(generate_name_for_current_level_community, G, current_level, next_level, community_id, communities_at_current_level, dendrogram, CommunitiesNames): community_id
                for community_id in set(communities_at_current_level.values())
            }

            for future in concurrent.futures.as_completed(future_to_community):
                community_id, community_name = future.result()
                level_community_names[community_id] = community_name

        CommunitiesNames[current_level] = level_community_names
        print(f"Communities names at level {current_level}: DONE")

    return CommunitiesNames


def add_community_attributes(graph, community_list, model, communitiesNames):
    num_levels = len(community_list)

    for level in range(num_levels):
        #for node_idx, community_value in community_list[level].items():
        for node_id, community_id in community_list[level].items():
            # Find the index of the node with the specified "id" attribute
            node_idx = graph.vs.find(id=node_id).index
            node = graph.vs[node_idx]
            node[f"community_level_{level}_{model}"] = communitiesNames[level][community_id]

def get_graph_name(application, graph_id):
    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=(user, password), database=database_name)
    # Build the Cypher query
    query = (f"""
        match (n:{application})
        where ID(n) = {graph_id}
        RETURN n.Name AS nodeName
        """
    )
    # Execute the Cypher query
    with driver.session() as session:
        result = session.run(query)
        record = result.single()  # Assuming you expect only one result
    # Extract the node label from the result
    node_name = record['nodeName'] if record else None
    # Close the Neo4j driver
    driver.close()
    return node_name

def nodes_of_interest(G, application, graph_type, graph_id):
    if graph_type == "DataGraph":
        table_name =  get_graph_name(application, graph_id)
        start_nodes = [node for node, attr in enumerate(G.vs) if 'DgStartPoint' in attr.attributes() and attr['DgStartPoint'] == "start"]
        start_nodes = [node for node in start_nodes if G.vs[node]['Name'] == table_name]
        end_nodes = [node for node, attr in enumerate(G.vs) if 'DgEndPoint' in attr.attributes() and attr['DgEndPoint'] == "end"]
        return start_nodes, end_nodes
    elif graph_type == "Transaction":
        entry_name = get_graph_name(application, graph_id)
        start_nodes = [node for node, attr in enumerate(G.vs) if 'StartPoint' in attr.attributes() and attr['StartPoint'] == "start"]
        start_nodes = [node for node in start_nodes if G.vs[node]['Name'] == entry_name]
        end_nodes = [node for node, attr in enumerate(G.vs) if 'EndPoint' in attr.attributes() and attr['EndPoint'] == "end"]
        """
        start_nodes = [edge.target for edge in G.es if 'type' in edge.attributes() and edge['type'] == "STARTS_WITH"]
        start_nodes = [node for node in start_nodes if G.vs[node]['Name'] == entry_name]
        end_nodes = [edge.target for edge in G.es if 'type' in edge.attributes() and edge['type'] == "ENDS_WITH"]
        """
        return start_nodes, end_nodes
    else:
        print("nodes_of_interest is built for DataGraph or Transaction")

def generate_cypher_query(application, linkTypes):
    if linkTypes == ["all"]:
        cypher_query = (f"""
            CALL cast.linkTypes(['CALL_IN_TRAN', 'SEMANTIC']) yield linkTypes
            WITH linkTypes + [] AS updatedLinkTypes
            MATCH p=(n:{application})<-[r]-(m:{application})
            WHERE (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND type(r) IN updatedLinkTypes
            RETURN DISTINCT n, r, m
            """
        )
    else : 
        cypher_query = (f"""
            WITH {linkTypes} as linkTypes2
            CALL cast.linkTypes(['SEMANTIC']) yield linkTypes
            WITH linkTypes + linkTypes2 + [] AS updatedLinkTypes
            MATCH p=(n:{application})<-[r]-(m:{application})
            WHERE (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND type(r) IN updatedLinkTypes
            RETURN DISTINCT n, r, m
            """
        )
    return cypher_query

def update_neo4j_graph(G, new_attributes_name, application, model, linkTypes):
    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=(user, password), database=database_name)

    # New node name. ex: LeidenUSESELECT
    newNodeName = f"{model}App"
    for item in linkTypes:
        newNodeName += item

    # First query to create a new Model node and to link it to all the nodes with new links IS_IN_MODEL 
    # and store the concern link types as a property of the new node.
    if linkTypes == ["all"]:
        cypher_query = (f"""
            CALL cast.linkTypes(['CALL_IN_TRAN', 'SEMANTIC']) yield linkTypes
            WITH linkTypes + [] AS updatedLinkType
            MATCH p=(n:{application})<-[r]-(m:{application})
            WHERE (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND type(r) IN updatedLinkTypes
            WITH DISTINCT n, m, updatedLinkTypes
            MERGE (new:Model:{application} {{name: '{newNodeName}'}})
            ON CREATE SET new.LinkTypes = {linkTypes}
            MERGE (new)<-[:IS_IN_MODEL]-(n)
            MERGE (new)<-[:IS_IN_MODEL]-(m)
            """
        )
    else :
        cypher_query = (f"""
            WITH {linkTypes} as linkTypes2
            CALL cast.linkTypes(['SEMANTIC']) yield linkTypes
            WITH linkTypes + linkTypes2 + [] AS updatedLinkTypes
            MATCH p=(n:{application})<-[r]-(m:{application})
            WHERE (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND type(r) IN updatedLinkTypes
            WITH DISTINCT n, m, updatedLinkTypes
            MERGE (new:Model:{application} {{name: '{newNodeName}'}})
            ON CREATE SET new.LinkTypes = {linkTypes}
            MERGE (new)<-[:IS_IN_MODEL]-(n)
            MERGE (new)<-[:IS_IN_MODEL]-(m)
            """
        )

    with driver.session() as session:
                session.run(cypher_query)

    # Second query to store the infomation about community membership as property of the new link IS_IN_MODEL between the new node Model and nodes.
    # Iterate through vertices in the graph
    for vertex in G.vs:
        #node_id = vertex.index
        node_id = vertex['id']
        #if f'community_level_0_{model}_{graph_type}_{graph_id}' in vertex.attributes():
        if vertex[f'community_level_0_{model}'] is not None:
            # Query to update nodes attributes with communities by level
            query = (f"""
                MATCH (new:Model:{application} {{name: '{newNodeName}'}})
                MATCH p2 = (new)<-[r:IS_IN_MODEL]-(m:{application})
                WHERE ID(m) = {node_id}
                SET r.Community =  [{', '.join([f"'{vertex[attr]}'" for attr in new_attributes_name])}]
                """    
                )
        
            # Execute the Cypher query
            with driver.session() as session:
                session.run(query)

    # Close the Neo4j driver
    driver.close()
    print(f"The new attributes (community by level) have been loaded to the neo4j {application} graph.")


def Leiden_App_Graph(application, linkTypes=["all"]):
    model = "Leiden"

    linkTypes = sorted(linkTypes)

    start_time_loading_graph = time.time()

    # Crée une instance de la classe Neo4jGraph
    neo4j_graph = Neo4jGraph(URI, user, password, database=database_name)

    # Cypher query to retrieve the graph
    cypher_query = generate_cypher_query(application, linkTypes)

    # Retrieve the graph based on the Cypher query
    G = neo4j_graph.get_graph(cypher_query)

    # Close the connection to Neo4j
    neo4j_graph.close()

    # Check if G is not empty (bad analizer analyzis can lead to it)
    if G.vcount() == 0:
        print(f"The {application} graph is Object/SubOject node empty.")
        return
    
    # Print the number of nodes
    print(f"The {application} graph has {G.vcount()} Object/SubOject nodes.")

    # Print the number of disconnected parts (connected components)
    connected_components = G.components(mode="weak")
    num_components = len(connected_components)
    print(f"The graph has {num_components} disconnected parts.")

    # Print the number of nodes in each component
    #for i, component in enumerate(connected_components, start=1):
    #    num_nodes = len(component)
    #    print(f"Component {i}: {num_nodes} nodes")
    
    end_time_loading_graph = time.time()
    print(f"Graph loading time:  {end_time_loading_graph-start_time_loading_graph}")
    
    set_default_weight(G)
    
    start_time_algo = time.time()

    # Adding semantic through weight on edges based on similarity
    #add_semantic_as_weight(G)

    # Identify nodes of interest (start and end points) to exclude from the induced subgraph
    #start_nodes, end_nodes = nodes_of_interest(G, application, graph_type, graph_id)
    #exclude_indices = set(start_nodes + end_nodes)

    # Perform community detection
    result, hierarchy_tree = community_detection_hierarchy(G, level=2)

    # Print the number of communities by level
    for level, partition in enumerate(result):
        print(f"Level {level}: {len(set(partition.values()))} communities")

        # Dictionary to store node count by community
        community_node_count = {}
        # Count the number of nodes in each community
        for node, community in partition.items():
            if community not in community_node_count:
                community_node_count[community] = 0
            community_node_count[community] += 1
        # Print the number of nodes by community if more than one node
        for community, count in community_node_count.items():
            if count > 1000:
                print(f"  Community {community}: {count} nodes")
    
    end_time_algo = time.time()
    print(f"Algo time:  {end_time_algo-start_time_algo}")
    
    start_time_names = time.time()

    # Compute the communities names for each level
    communities_names = communitiesNamesThread(G, result)

    for i in range(len(communities_names)):
        print(f"Nb of communities at level {len(communities_names)-1-i} : {len(communities_names[i])}")

    end_time_names = time.time()
    print(f"Naming time:  {end_time_names-start_time_names}")

    # Add attributes to the igraph graph G
    add_community_attributes(G, result, model, communities_names)

    # Retrieve the name of the attribute
    for i in range(len(result)):
        attribute_name = f"community_level_{i}_{model}"
        attribute_values = G.vs[attribute_name]
        unique_values = set(attribute_values)
        print(f"unique_values level {i}: {len(unique_values)}")

    # Create a list with the new attributes
    new_attributes_name = [f'community_level_{level}_{model}' for level in range(len(result))]

    start_time_neo = time.time()

    # Update back to neo4j the new attributes to the link property
    update_neo4j_graph(G, new_attributes_name, application, model, linkTypes)

    end_time_neo = time.time()
    print(f"Update time:  {end_time_neo-start_time_neo}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Leiden community detection on Neo4j graph")
    parser.add_argument("application", type=str, help="Application name")
    #parser.add_argument("graph_id", type=int, help="Graph ID")
    #parser.add_argument("graph_type", type=str, choices=["DataGraph", "Transaction"], help="Graph type")
    parser.add_argument("linkTypes", nargs="*", type=str, default=["all"], help="List of link types considered in community detection")

    args = parser.parse_args()

    application = args.application
    #graph_id = args.graph_id
    #graph_type = args.graph_type
    linkTypes = args.linkTypes

    print(application)

    start_time = time.time()

    Leiden_App_Graph(application, linkTypes)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for the Leiden Algorithm on the {application} application: {elapsed_time} seconds")