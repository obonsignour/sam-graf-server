import networkx as nx
import community as community
#from neo4j_connector_nx import Neo4jGraph
import time
from neo4j import GraphDatabase
import openai
import argparse
import concurrent.futures

from Algorithms.toImport.neo4j_connector_nx import Neo4jGraph
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


# Sets the weight attribute to 1 for all edges in the graph G that do not already have a weight attribute.
def set_default_weight(G):
    for u, v, data in G.edges(data=True):
        if 'weight' not in data:
            G[u][v]['weight'] = 1

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

# Iterative version of the Louvain Algorithm (tree of community partitions of the graph)
def community_detection_hierarchy(graph, level=None):
    result_partitions = []
    hierarchy_tree = {}

    # Level 0
    partition_level_0 = community.best_partition(graph, random_state=42, weight='weight')
    result_partitions.append(partition_level_0)
    hierarchy_tree[0] = {community_id: [] for community_id in set(partition_level_0.values())}
    #print(f"Level 0 merged partition: {partition_level_0}")

    current_level = 1
    while True:
        subgraph_partitions = []
        subgraph_tree = {}
        for community_id in set(partition_level_0.values()):
            nodes_in_community = [node for node, cid in partition_level_0.items() if cid == community_id]
            subgraph = graph.subgraph(nodes_in_community)
            subgraph_partition = community.best_partition(subgraph, random_state=42, weight='weight')
            subgraph_partitions.append(subgraph_partition)
            subgraph_tree[community_id] = list(set(subgraph_partition.values()))
        
        merged_partition = merge_dicts(subgraph_partitions)

        # Debugging: Check merging consistency
        #print(f"Level {current_level} merged partition: {merged_partition}")
        #print(f"Subgraph tree at level {current_level}: {subgraph_tree}")

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


def get_most_important_nodes(graph, node_list, metric='betweenness', top_n=100):
    """
    Identifie les nœuds les plus importants d'une liste dans un graphe NetworkX selon une métrique.

    :param graph: Graphe NetworkX.
    :param node_list: Liste des identifiants de nœuds.
    :param metric: La métrique à utiliser pour déterminer l'importance des nœuds ('degree', 'closeness', 'betweenness', etc.).
    :param top_n: Le nombre de nœuds les plus importants à retourner.
    :return: Liste des nœuds les plus importants.
    """

    if len(node_list)>top_n : 
        # Calcul de la centralité selon la métrique spécifiée
        #if metric == 'degree':
        #    centrality = nx.degree_centrality(graph)
        if metric == 'degree':
            centrality = {node: val for node, val in graph.degree(node_list)}
        elif metric == 'closeness':
            centrality = nx.closeness_centrality(graph)
        elif metric == 'betweenness':
            centrality = nx.betweenness_centrality_subset(graph)
        elif metric == 'eigenvector':
            centrality = nx.eigenvector_centrality(graph)
        else:
            raise ValueError("Métrique non reconnue. Choisissez parmi 'degree', 'closeness', 'betweenness', 'eigenvector'.")

        # Filtrage des centralités pour ne garder que celles des nœuds dans node_list
        centrality = {node: centrality[node] for node in node_list if node in centrality}

        # Tri des nœuds par ordre décroissant de centralité
        sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)

        # Retourner les top_n nœuds les plus importants
        return sorted_nodes[:top_n]
    
    else :
        return node_list

# Helper function to generate community names taking only the most 'important' nodes
def generate_name_for_community(G, community_id, community_nodes_ids):
    important_community_nodes_ids = get_most_important_nodes(G, community_nodes_ids, metric='degree')
    important_community_nodes_names = [G.nodes[node]['Name'] for node in important_community_nodes_ids]
    if len(important_community_nodes_names) < 2 : 
        return community_id, important_community_nodes_names[0]
    else :
        return community_id, generate_community_name(important_community_nodes_names)


def generate_name_for_current_level_community(G, current_level, next_level, community_id, communities_at_current_level, dendrogram, CommunitiesNames):
    # Get nodes in the current level community
    nodes_in_current_community = [node for node, cid in communities_at_current_level.items() if cid == community_id]
    # Get subcommunity IDs at the next level
    subcommunity_ids = set(dendrogram[next_level][node] for node in nodes_in_current_community if node in dendrogram[next_level])
    # Retrieve names of these subcommunities
    subcommunity_names = [CommunitiesNames[next_level][sub_id] for sub_id in subcommunity_ids]
    
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

    # Generate names for the top level communities
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_community = {
            executor.submit(generate_name_for_community, G, community_id, [node for node, cid in communities_at_level.items() if cid == community_id]): community_id
            for community_id in set(communities_at_level.values())
        }

        for future in concurrent.futures.as_completed(future_to_community):
            community_id, community_name = future.result()
            level_community_names[community_id] = community_name

    CommunitiesNames[level] = level_community_names
    
    # Debug: Track which communities are being processed
    print(f"Communities names at level {level}: DONE")
    #print(f"Level {level} community names: {level_community_names}")
    
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
        
        # Debug: Track which communities are being processed
        print(f"Communities names at level {current_level}: DONE")
        #print(f"Level {current_level} community names: {level_community_names}")

    return CommunitiesNames


def add_community_attributes(graph, community_list, model, graph_type, graph_id, communitiesNames, exclude_indices):
    num_levels = len(community_list)

    for level in range(num_levels):
        for node, community_id in community_list[level].items():
            # Skip the nodes in exclude_indices
            if node in exclude_indices:
                continue

            # Add the community attribute for this node
            graph.nodes[node][f'community_level_{level}_{model}_{graph_type}_{graph_id}'] = communitiesNames[level][community_id]

# Function to get both startNodes and endNodes
def nodes_of_interest(application, graph_type, graph_id):
    # Connect to the Neo4j database
    driver = GraphDatabase.driver(URI, auth=(user, password), database=database_name)
    
    end_nodes = []
    start_nodes = []

    # Dynamically build the query with the application and graph_type variables inserted directly
    query = f"""
    MATCH (n:{application})-[:ENDS_WITH]-(d:{graph_type}:{application})
    WHERE ID(d) = {graph_id}
    AND (n:Object OR n:SubObject)
    RETURN DISTINCT ID(n) AS nodeId, 'endNode' AS nodeType
    UNION
    MATCH (n:{application})-[:STARTS_WITH]-(d:{graph_type}:{application})
    WHERE ID(d) = {graph_id}
    AND (n:Object OR n:SubObject)
    RETURN DISTINCT ID(n) AS nodeId, 'startNode' AS nodeType
    """
    
    # Execute query and collect results
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            if record["nodeType"] == "endNode":
                end_nodes.append(record["nodeId"])
            elif record["nodeType"] == "startNode":
                start_nodes.append(record["nodeId"])

    # Close the driver connection
    driver.close()

    return start_nodes, end_nodes

def generate_cypher_query(application, graph_type, graph_id, linkTypes):
    if graph_type == "DataGraph":
        relationship_type = "IS_IN_DATAGRAPH"
    elif graph_type == "Transaction":
        relationship_type = "IS_IN_TRANSACTION"
    else :
        return print("generate_cypher_query is build for DataGraph or Transaction")
    if linkTypes == ["all"]:
        cypher_query = (f"""
            CALL cast.linkTypes(['CALL_IN_TRAN', 'SEMANTIC']) yield linkTypes
            WITH linkTypes + [] AS updatedLinkTypes
            MATCH (d:{graph_type}:{application})<-[:{relationship_type}]-(n)
            WITH collect(id(n)) AS nodeIds,updatedLinkTypes
            MATCH p=(d:{graph_type}:{application})<-[:{relationship_type}]-(n:{application})<-[r]-(m:{application})-[:{relationship_type}]->(d)
            WHERE ID(d) = {graph_id}
            AND (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND id(n) IN nodeIds AND id(m) IN nodeIds
            AND type(r) IN updatedLinkTypes
            RETURN DISTINCT n, r, m
            """
        )
    else : 
        cypher_query = (f"""
            WITH {linkTypes} as linkTypes2
            CALL cast.linkTypes(['SEMANTIC']) yield linkTypes
            WITH linkTypes + linkTypes2 + [] AS updatedLinkTypes
            MATCH (d:{graph_type}:{application})<-[:{relationship_type}]-(n)
            WITH collect(id(n)) AS nodeIds,updatedLinkTypes
            MATCH p=(d:{graph_type}:{application})<-[:{relationship_type}]-(n:{application})<-[r]-(m:{application})-[:{relationship_type}]->(d)
            WHERE ID(d) = {graph_id}
            AND (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND id(n) IN nodeIds AND id(m) IN nodeIds
            AND type(r) IN updatedLinkTypes
            RETURN DISTINCT n, r, m
            """
        )
    return cypher_query

def update_neo4j_graph(G, new_attributes_name, application, graph_id, graph_type, model, linkTypes):
    if graph_type == "DataGraph":
        relationship_type = "IS_IN_DATAGRAPH"
    elif graph_type == "Transaction":
        relationship_type = "IS_IN_TRANSACTION"
    else :
        return print("update_neo4j_graph is build for DataGraph or Transaction")

    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=(user, password), database=database_name)

    # New node name. ex: LeidenUSESELECT
    newNodeName = f"{model}"
    for item in linkTypes:
        newNodeName += item

    # First query to create a new Model node and to link it to all the nodes with new links IS_IN_MODEL 
    # and store the concern link types as a property of the new node.
    if linkTypes == ["all"]:
        cypher_query = (f"""
            CALL cast.linkTypes(['CALL_IN_TRAN', 'SEMANTIC']) yield linkTypes
            WITH linkTypes + [] AS updatedLinkType
            MATCH (d:{graph_type}:{application})<-[:{relationship_type}]-(n)
            WITH collect(id(n)) AS nodeIds,updatedLinkTypes
            MATCH p=(d:{graph_type}:{application})<-[:{relationship_type}]-(n:{application})<-[r]-(m:{application})-[:{relationship_type}]->(d)
            WHERE ID(d) = {graph_id}
            AND (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND id(n) IN nodeIds AND id(m) IN nodeIds
            AND type(r) IN updatedLinkTypes
            WITH DISTINCT n, m, d, updatedLinkTypes
            MERGE (new:Model:{application} {{name: '{newNodeName}'}})-[:RELATES_TO]->(d)
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
            MATCH (d:{graph_type}:{application})<-[:{relationship_type}]-(n)
            WITH collect(id(n)) AS nodeIds,updatedLinkTypes
            MATCH p=(d:{graph_type}:{application})<-[:{relationship_type}]-(n:{application})<-[r]-(m:{application})-[:{relationship_type}]->(d)
            WHERE ID(d) = {graph_id}
            AND (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND id(n) IN nodeIds AND id(m) IN nodeIds
            AND type(r) IN updatedLinkTypes
            WITH DISTINCT n, m, d, updatedLinkTypes
            MERGE (new:Model:{application} {{name: '{newNodeName}'}})-[:RELATES_TO]->(d)
            ON CREATE SET new.LinkTypes = {linkTypes}
            MERGE (new)<-[:IS_IN_MODEL]-(n)
            MERGE (new)<-[:IS_IN_MODEL]-(m)
            """
        )

    with driver.session() as session:
                session.run(cypher_query)

    # Second query to store the infomation about community membership as property of the new link IS_IN_MODEL between the new node Model and nodes.
    # Iterate through nodes in the graph
    for node_id, data in G.nodes(data=True):
        if f'community_level_0_{model}_{graph_type}_{graph_id}' in G.nodes[node_id]:
            # Query to update nodes attributes with communities by level
            query = (f"""
                MATCH p1=(n:{graph_type}:{application})<-[:RELATES_TO]-(new:Model:{application} {{name: '{newNodeName}'}})
                WHERE ID(n) = {graph_id}
                MATCH p2 = (new)<-[r:IS_IN_MODEL]-(m:{application})
                WHERE ID(m) = {node_id}
                SET r.Community = [{', '.join([f"'{data.get(attr)}'" for attr in new_attributes_name])}]
                """    
                )
        
            # Execute the Cypher query
            with driver.session() as session:
                session.run(query)

    # Close the Neo4j driver
    driver.close()
    print(f"The new attributes (community by level) have been loaded to the neo4j {graph_type} graph {graph_id}.")


def Undirected_Louvain_Call_Graph(application, graph_id, graph_type, linkTypes=["all"]):
    model = "UndirectedLouvain"

    linkTypes = sorted(linkTypes)

    start_time_loading_graph = time.time()

    # Crée une instance de la classe Neo4jGraph
    neo4j_graph = Neo4jGraph(URI, user, password, database=database_name)

    # Cypher query to retrieve the graph
    cypher_query = generate_cypher_query(application, graph_type, graph_id, linkTypes)

    # Retrieve the graph based on the Cypher query
    G = neo4j_graph.get_graph(cypher_query)

    # Close the connection to Neo4j
    neo4j_graph.close()

    # Check if G is not empty (bad analizer analyzis can lead to it)
    if G.number_of_nodes() == 0:
        print(f"The Neo4j graph {graph_id} is Object/SubOject node empty.")
        return
    
    # Print the number of nodes
    print(f"The Neo4j graph {graph_id} has {G.number_of_nodes()} Object/SubOject nodes.")

    # Print the number of disconnected parts (connected components)
    connected_components = list(nx.connected_components(G))
    num_components = len(connected_components)
    print(f"The Neo4j graph {graph_id} has {num_components} disconnected parts.")
    """
    # Print the number of nodes in each component
    for i, component in enumerate(connected_components, start=1):
        num_nodes = len(component)
        print(f"Component {i}: {num_nodes} nodes")
        if num_nodes < 5:
            node_types = [G.nodes[node]['Type'] for node in component]
            print(f"Types of nodes in Component {i}: {node_types}")
            print("")
    """
    end_time_loading_graph = time.time()
    print(f"Graph loading time:  {end_time_loading_graph-start_time_loading_graph}")

    set_default_weight(G)

    start_time_algo = time.time()

    # Perform community detection using Undirected Louvain method
    #partition = community.partition_at_level(dendrogram, len(dendrogram) - 1)
    #dendrogram = community.generate_dendrogram(G, random_state=42, weight='weight') #, random_state=42
    dendrogram, hierarchy_tree = community_detection_hierarchy(G, level=2)

    # Print the number of communities by level
    for level, partition in enumerate(dendrogram):
        print(f"Level {level}: {len(set(partition.values()))} communities")

    end_time_algo = time.time()
    print(f"Algo time:  {end_time_algo-start_time_algo}")
    
    start_time_names = time.time()

    # Compute the communities names for each level
    communities_names = communitiesNamesThread(G, dendrogram)

    for i in range(len(communities_names)):
        print(f"Nb of communities at level {len(communities_names)-1-i} : {len(communities_names[i])}")

    end_time_names = time.time()
    print(f"Naming time:  {end_time_names-start_time_names}")

    # Identify nodes of interest (start and end points) to exclude
    start_nodes, end_nodes = nodes_of_interest(application, graph_type, graph_id)
    exclude_indices = set(start_nodes + end_nodes)

    # Add attributes to the graph G
    add_community_attributes(G, dendrogram, model, graph_type, graph_id, communities_names, exclude_indices)
    
    # Retrieve the name of the attribute
    for i in range(len(dendrogram)):
        attribute_name = f"community_level_{i}_{model}_{graph_type}_{graph_id}"
        attribute_values = [attrs.get(attribute_name, None) for node, attrs in G.nodes(data=True)]
        unique_values = set(attribute_values)
        print(f"unique_values level {i}: {len(unique_values)}")

    # Create a list with the new attributes
    new_attributes_name = [f'community_level_{level}_{model}_{graph_type}_{graph_id}' for level in range(len(dendrogram))]

    start_time_neo = time.time()

    # Update back to neo4j the new attributes to the link property after creating the new Model node
    update_neo4j_graph(G, new_attributes_name, application, graph_id, graph_type, model, linkTypes)

    end_time_neo = time.time()
    print(f"Update time:  {end_time_neo-start_time_neo}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ULouvain community detection on Neo4j graph")
    parser.add_argument("application", type=str, help="Application name")
    parser.add_argument("graph_id", type=int, help="Graph ID")
    parser.add_argument("graph_type", type=str, choices=["DataGraph", "Transaction"], help="Graph type")
    parser.add_argument("linkTypes", nargs="*", type=str, default=["all"], help="List of link types considered in community detection")

    args = parser.parse_args()

    application = args.application
    graph_id = args.graph_id
    graph_type = args.graph_type
    linkTypes = args.linkTypes

    print(application)

    start_time = time.time()

    # Call Leiden_on_one_graph with parsed arguments
    Undirected_Louvain_Call_Graph(application, graph_id, graph_type, linkTypes)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for the UndirectedLouvain Algorithm on the neo4j {graph_type} graph {graph_id}: {elapsed_time} seconds")