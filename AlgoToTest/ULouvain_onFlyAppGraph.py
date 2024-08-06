import networkx as nx
import community as community
from neo4j_connector_nx import Neo4jGraph
import time
from neo4j import GraphDatabase
import openai
import argparse
import concurrent.futures

#from AlgoToTest.neo4j_connector_nx import Neo4jGraph
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
    for u, v in G.edges():
        weight = similarity(G.nodes(data=True)[u], G.nodes(data=True)[v], properties_of_interest)
        G[u][v]['weight'] = weight

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
    #print("i am here 1.1")
    """
    Identifie les nœuds les plus importants d'une liste dans un graphe NetworkX selon une métrique.

    :param graph: Graphe NetworkX.
    :param node_list: Liste des identifiants de nœuds.
    :param metric: La métrique à utiliser pour déterminer l'importance des nœuds ('degree', 'closeness', 'betweenness', etc.).
    :param top_n: Le nombre de nœuds les plus importants à retourner.
    :return: Liste des nœuds les plus importants.
    """

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


    #print("i am here 1.2")

    # Filtrage des centralités pour ne garder que celles des nœuds dans node_list
    centrality = {node: centrality[node] for node in node_list if node in centrality}

    #print("i am here 1.3")

    # Tri des nœuds par ordre décroissant de centralité
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)

    #print("i am here 1.4")

    # Retourner les top_n nœuds les plus importants
    return sorted_nodes[:top_n]

# Helper function to generate community names taking only the most 'important' nodes
def generate_name_for_community(G, community_id, community_nodes_ids):
    #print("i am here 0.3.1")
    important_community_nodes_ids = get_most_important_nodes(G, community_nodes_ids, metric='degree')
    important_community_nodes_names = [G.nodes[node]['Name'] for node in important_community_nodes_ids]
    #print("i am here 0.3.2")
    return community_id, generate_community_name(important_community_nodes_names)

# Helper function to generate community names
def generate_name_for_community2(community_id, community_subcommunities_names):
    return community_id, generate_community_name(community_subcommunities_names)

def communitiesNamesThread(G, dendrogram):
    #print("i am here 0.1")
    # Initialize a dictionary to store community names at each level
    CommunitiesNames = {}

    # Process the smallest communities (level 0)
    level = 0
    communities_at_level = dendrogram[level]
    level_community_names = {}

    #print("i am here 0.2")

    # Use ThreadPoolExecutor to parallelize generate_community_name calls
    with concurrent.futures.ThreadPoolExecutor() as executor:
        #print("i am here 0.3")
        future_to_community = {
            #executor.submit(generate_name_for_community, G, community_id, [G.nodes[node]['Name'] for node in [node for node, cid in communities_at_level.items() if cid == community_id]]): community_id
            executor.submit(generate_name_for_community, G, community_id, [node for node, cid in communities_at_level.items() if cid == community_id]): community_id
            for community_id in set(communities_at_level.values())
        }
        #print("i am here 0.4")

        for future in concurrent.futures.as_completed(future_to_community):
            community_id, community_name = future.result()
            level_community_names[community_id] = community_name
        
        #print("i am here 0.5")

    CommunitiesNames[level] = level_community_names
    
    #print("i am here 0.6")
    
    # Process higher levels (level 1 to len(dendrogram) - 1)
    for level in range(1, len(dendrogram)):
        communities_at_level = dendrogram[level]
        level_community_names = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_community = {
                executor.submit(generate_name_for_community2, community_id, [
                    CommunitiesNames[level - 1][com] for com in set(
                        com for com, cid in communities_at_level.items() if cid == community_id
                    )
                ]): community_id
                for community_id in set(communities_at_level.values())
            }

            for future in concurrent.futures.as_completed(future_to_community):
                community_id, community_name = future.result()
                level_community_names[community_id] = community_name

        CommunitiesNames[level] = level_community_names

    # Invert dictionary
    reversed_values = list(CommunitiesNames.values())[::-1]
    CommunitiesNames = {k: v for k, v in zip(CommunitiesNames.keys(), reversed_values)}

    return CommunitiesNames

def communitiesNames(G, dendrogram):
    # Initialize a dictionary to store community names at each level
    CommunitiesNames = {}

    # Process the smallest communities (level 0)
    level = 0
    communities_at_level = dendrogram[level]
    level_community_names = {}

    for community_id in set(communities_at_level.values()):
        community_nodes = [node for node, cid in communities_at_level.items() if cid == community_id]
        community_nodes_names = [G.nodes[node]['Name'] for node in community_nodes]
        community_name = generate_community_name(community_nodes_names)
        level_community_names[community_id] = community_name

    CommunitiesNames[level] = level_community_names

    # Process higher levels (level 1 to len(dendrogram) - 1)
    for level in range(1, len(dendrogram)):
        communities_at_level = dendrogram[level]
        level_community_names = {}

        for community_id in set(communities_at_level.values()):
            # Retrieve the keys having the same value in communities_at_level
            previous_level_community_ids = set(
                com for com, cid in communities_at_level.items() if cid == community_id
            )

            # Retrieve names from the previous level's communities
            previous_level_community_names = [
                CommunitiesNames[level - 1][com] for com in previous_level_community_ids
            ]
            community_name = generate_community_name(previous_level_community_names)
            level_community_names[community_id] = community_name

        CommunitiesNames[level] = level_community_names
        
    # Invert dictionary
    reversed_values = list(CommunitiesNames.values())[::-1]
    CommunitiesNames = {k: v for k, v in zip(CommunitiesNames.keys(), reversed_values)}
    #print(f"Level {level} communities: {CommunitiesNames[level]}")

    return CommunitiesNames


def add_community_attributes(G, dendrogram, model, communitiesNames):
    # Add community attributes to nodes for each level in the dendrogram
    for level in range(len(dendrogram) ):
        #communities_at_level = community.partition_at_level(dendrogram, level)
        communities_at_level = community.partition_at_level(dendrogram, len(dendrogram)-1 - level)
        for node, community_id in communities_at_level.items():
            G.nodes[node][f'community_level_{level}_{model}'] = communitiesNames[level][community_id]

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
        start_nodes = [node for node in G.nodes if G.nodes[node].get('DgStartPoint') == "start"]
        start_nodes = [node for node in start_nodes if G.nodes[node].get('Name') == table_name]
        end_nodes = [node for node in G.nodes if G.nodes[node].get('DgEndPoint') == "end"]
        return start_nodes, end_nodes
    elif graph_type == "Transaction":
        entry_name = get_graph_name(application, graph_id)
        """
        start_nodes = [edge[1] for edge in G.edges if G.edges[edge].get('type') == "STARTS_WITH"]
        start_nodes = [node for node in start_nodes if G.nodes[node].get('Name') == entry_name]
        end_nodes = [edge[1] for edge in G.edges if G.edges[edge].get('type') == "ENDS_WITH"]
        """
        start_nodes = [node for node in G.nodes if G.nodes[node].get('StartPoint') == "start"]
        start_nodes = [node for node in start_nodes if G.nodes[node].get('Name') == entry_name]
        end_nodes = [node for node in G.nodes if G.nodes[node].get('EndPoint') == "end"]
        return start_nodes, end_nodes
    else :
        return print("nodes_of_interest is build for DataGraph or Transaction")

def generate_cypher_query(application, linkTypes):
    if linkTypes == ["all"]:
        cypher_query = (f"""
            CALL cast.linkTypes(['CALL_IN_TRAN']) yield linkTypes
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
            WITH {linkTypes} as linkTypes
            WITH linkTypes + [] AS updatedLinkTypes
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
            CALL cast.linkTypes(['CALL_IN_TRAN']) yield linkTypes
            WITH linkTypes + [] AS updatedLinkTypes
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
            //CALL cast.linkTypes(['CALL_IN_TRAN']) yield linkTypes
            WITH {linkTypes} as linkTypes
            WITH linkTypes + [] AS updatedLinkTypes
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
    # Iterate through nodes in the graph
    for node_id, data in G.nodes(data=True):
        if f'community_level_0_{model}' in G.nodes[node_id]:
            # Query to update nodes attributes with communities by level
            query = (f"""
                MATCH (new:Model:{application} {{name: '{newNodeName}'}})
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
    print(f"The new attributes (community by level) have been loaded to the neo4j {application} graph.")


def Undirected_Louvain_on_one_graph(application, linkTypes=["all"]):
    model = "UndirectedLouvain"

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
    if G.number_of_nodes() == 0:
        print(f"The {application} graph is Object/SubOject node empty.")
        return
    
    end_time_loading_graph = time.time()
    print(f"Graph loading time:  {end_time_loading_graph-start_time_loading_graph}")

    start_time_algo = time.time()

    # Adding semantic through weight on edges based on similarity
    add_semantic_as_weight(G)

    # Identify nodes of interest (start and end points) to exclude from the induced subgraph
    #start_nodes, end_nodes = nodes_of_interest(G, application, graph_type, graph_id)
    #exclude_indices = set(start_nodes + end_nodes)

    # Perform community detection using Undirected Louvain method
    dendrogram = community.generate_dendrogram(G, random_state=42, weight='weight') #, random_state=42
    #partition = community.partition_at_level(dendrogram, len(dendrogram) - 1)

    end_time_algo = time.time()
    print(f"Algo time:  {end_time_algo-start_time_algo}")

    start_time_names = time.time()

    # Compute the communities names for each level
    communities_names = communitiesNamesThread(G, dendrogram)

    end_time_names = time.time()
    print(f"Names time:  {end_time_names-start_time_names}")

    # Add attributes to the graph G
    add_community_attributes(G, dendrogram, model, communities_names)
    
    # Create a list with the new attributes
    new_attributes_name = [f'community_level_{level}_{model}' for level in range(len(dendrogram))]

    start_time_neo = time.time()

    # Update back to neo4j the new attributes to the link property after creating the new Model node
    update_neo4j_graph(G, new_attributes_name, application, model, linkTypes)

    end_time_neo = time.time()
    print(f"Update time:  {end_time_neo-start_time_neo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ULouvain community detection on Neo4j graph")
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

    # Call Leiden_on_one_graph with parsed arguments
    Undirected_Louvain_on_one_graph(application, linkTypes)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for the UndirectedLouvain Algorithm on the {application} application: {elapsed_time} seconds")