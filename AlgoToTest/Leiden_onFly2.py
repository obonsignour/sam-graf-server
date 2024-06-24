import leidenalg as la
import igraph as ig

from neo4j import GraphDatabase
import time
import openai
import argparse

from AlgoToTest.neo4j_connector_igraph import Neo4jGraph
import os
from dotenv import load_dotenv

start_time = time.time()

# Link to OpenAI
# PUT THE 2 LINES ON THE OPENAI KEY HERE
# Load the .env file
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
properties_of_interest = ['Type', 'Level', 'External', 'Hidden', 'Method']

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


def get_graph_type(graph_id, application):
    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=(user, password), database=database_name)
    # Build the Cypher query
    query = (
        f"match (n:{application})\n"
        f"where ID(n) = {graph_id}\n"
        f"RETURN labels(n)[0] AS nodeLabel"
    )
    # Execute the Cypher query
    with driver.session() as session:
        result = session.run(query)
        record = result.single()  # Assuming you expect only one result
    # Extract the node label from the result
    node_label = record['nodeLabel'] if record else None
    # Close the Neo4j driver
    driver.close()
    return node_label


def community_detection_hierarchy(graph, level=None):
    result_partitions = []

    # Level 0
    partition_level_0 = convert_leiden_format(la.find_partition(graph, la.ModularityVertexPartition))
    partition_level_0 = {graph.vs['id'][k]: v for k, v in partition_level_0.items()}
    result_partitions.append(partition_level_0)

    current_level = 0
    while True:
        subgraph_partitions = []
        for community_id in set(partition_level_0.values()):
            nodes_in_community = graph.vs.select(lambda v: v["id"] in partition_level_0 and partition_level_0[v["id"]] == community_id)
            nodes_in_community = [v.index for v in nodes_in_community]
            # nodes_in_community = [index for index, node_id in enumerate(graph.vs["id"]) if partition_level_0.get(node_id) == community_id]
            subgraph = graph.induced_subgraph(nodes_in_community)
            subgraph_partition = convert_leiden_format(la.find_partition(subgraph, la.ModularityVertexPartition))
            # Replace keys in subgraph_partition with original node IDs
            subgraph_partition = {subgraph.vs['id'][k]: v for k, v in subgraph_partition.items()}
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

# Fonction pour générer un nom de communauté à partir d'une liste de termes


def generate_community_name(terms):
    # prompt = f"Generate a name for the community based on the following terms:\n{', '.join(terms)}"
    # prompt = f"Generate and return only one concise and meaningful name without symbols and of maximum 30 characters, grouping the following terms:\n{', '.join(terms)}"
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
            # {"role": "system", "content": "You are a helpful assis tant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5,
        temperature=0.7
    )
    return response.choices[0].message["content"].strip()


def communitiesNames(G, community_list):
    # print("length dendro : ", len(community_list))
    # Initialize a list to store community names for each level
    CommunitiesNames = {}
    # Iterate through each level of the dendrogram
    for level in range(len(community_list)):
        level_community_names = {}
        _community_for_level = set(community_list[level].values())
        print('Computing community name for communities at level', level)
        print('Number of communities at level', level, ':', len(_community_for_level))
        # Iterate through unique community IDs at the current level
        for community_id in set(community_list[level].values()):
            # Retrieve nodes belonging to the current community
            # community_nodes = [node for node, cid in community_list[level].items() if cid == community_id]

            community_nodes = G.vs.select(lambda v: v["id"] in community_list[level] and community_list[level][v["id"]] == community_id)
            community_nodes = [v.index for v in community_nodes]
            # Retrieve the names of nodes within each community
            # community_nodes_names = [node['Name'] for node in G.vs.select(community_nodes)]
            community_nodes_names = [G.vs[node]['Name'] for node in community_nodes]
            # Generate a community name for each community within one level
            community_name = generate_community_name(community_nodes_names)
            # Assign community name to the dictionary with community ID as key
            level_community_names[community_id] = community_name
        # Append the dictionary of community names for the current level to the main list
        CommunitiesNames[level] = level_community_names

    # CommunitiesNames[level][community_id] give the name of the community community_id
    # at level level.
    return CommunitiesNames


def add_community_attributes(graph, community_list, model, graph_type, graph_id, communitiesNames):
    num_levels = len(community_list)

    for level in range(num_levels):
        # for node_idx, community_value in community_list[level].items():
        for node_id, community_id in community_list[level].items():
            # Find the index of the node with the specified "id" attribute
            node_idx = graph.vs.find(id=node_id).index
            node = graph.vs[node_idx]
            node[f"community_level_{level}_{model}_{graph_type}_{graph_id}"] = communitiesNames[level][community_id]


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
        table_name = get_graph_name(application, graph_id)
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


def generate_cypher_query(application, graph_type, graph_id, linkTypes="all"):
    if graph_type == "DataGraph":
        relationship_type = "IS_IN_DATAGRAPH"
    elif graph_type == "Transaction":
        relationship_type = "IS_IN_TRANSACTION"
    else:
        return print("generate_cypher_query is build for DataGraph or Transaction")
    if linkTypes == "all":
        cypher_query = (f"""
            CALL cast.linkTypes(['CALL_IN_TRAN']) yield linkTypes
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
    else:
        cypher_query = (f"""
            WITH {linkTypes} as linkTypes
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
    return cypher_query


def update_neo4j_graph(updated_graph, new_attributes_name, application, graph_id, graph_type, model, linkTypes):
    if graph_type == "DataGraph":
        relationship_type = "IS_IN_DATAGRAPH"
    elif graph_type == "Transaction":
        relationship_type = "IS_IN_TRANSACTION"
    else:
        return print("generate_cypher_query is build for DataGraph or Transaction")

    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=(user, password), database=database_name)

    # New node name. ex: LeidenUSESELECT
    newNodeName = f"{model}"
    for item in linkTypes:
        newNodeName += item

    # First query to create a new Model node, link it to all the IS_IN_DATAGRAPH nodes whith a new links IS_IN_MODEL
    # and store the concern link types as a property of the new node.
    cypher_query = (f"""
        //CALL cast.linkTypes(['CALL_IN_TRAN']) yield linkTypes
        WITH {linkTypes} as linkTypes
        WITH linkTypes + [] AS updatedLinkTypes
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
        # Iterate through vertices in the graph
    for vertex in updated_graph.vs:
        # node_id = vertex.index
        node_id = vertex['id']
        # if f'community_level_0_{model}_{graph_type}_{graph_id}' in vertex.attributes():
        if vertex[f'community_level_0_{model}_{graph_type}_{graph_id}'] is not None:
            # Query to update nodes attributes with communities by level
            query = (f"""
                MATCH p1=(n:{graph_type}:{application})<-[:RELATES_TO]-(new:Model:{application} {{name: '{newNodeName}'}})
                WHERE ID(n) = {graph_id}
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
    print(f"The new attributes (community by level) have been loaded to the neo4j {graph_type} graph {graph_id}.")


def Leiden_on_one_graph(application, graph_id, graph_type, linkTypes=["all"]):
    model = "Leiden"

    linkTypes = sorted(linkTypes)

    # Crée une instance de la classe Neo4jGraph
    neo4j_graph = Neo4jGraph(URI, user, password, database=database_name)

    # Cypher query to retrieve the graph
    cypher_query = generate_cypher_query(application, graph_type, graph_id, linkTypes)

    # Retrieve the graph based on the Cypher query
    print('Retrieving the Neo4j graph...')
    G = neo4j_graph.get_graph(cypher_query)
    print('Neo4j graph retrieved.')

    # Close the connection to Neo4j
    neo4j_graph.close()

    # Check if G is not empty (bad analizer analyzis can lead to it)
    if G.vcount() == 0:
        print(f"The Neo4j graph {graph_id} is Object/SubOject node empty.")
        return

    """
    # Change the attribute name of attribute "id" (idk why igraph changed it in the load from neo4j)
    G.vs["id"] = G.vs["name"]
    del G.vs["name"]
    """

    # Adding semantic through weight on edges based on similarity
    add_semantic_as_weight(G)

    # Identify nodes of interest (start and end points) to exclude from the induced subgraph
    start_nodes, end_nodes = nodes_of_interest(G, application, graph_type, graph_id)
    exclude_indices = set(start_nodes + end_nodes)

    # Perform community detection
    result = community_detection_hierarchy(G.induced_subgraph([v for v in G.vs if v.index not in exclude_indices]), level=0)

    # Compute the communities names for each level
    print('Computing communities names...')
    communities_names = communitiesNames(G, result)
    print('Communities names computed.')

    # Add attributes to the igraph graph G
    add_community_attributes(G, result, model, graph_type, graph_id, communities_names)

    # Create a list with the new attributes
    new_attributes_name = [f'community_level_{level}_{model}_{graph_type}_{graph_id}' for level in range(len(result))]

    # Update back to neo4j the new attributes to the link property
    print('Updating the Neo4j graph...')
    update_neo4j_graph(G, new_attributes_name, application, graph_id, graph_type, model, linkTypes)
    print('Neo4j graph updated.')


def get_all_graphs(graph_type, application):
    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=(user, password), database=database_name)
    # Build the Cypher query
    query = (
        f"MATCH (n:{graph_type}:{application})\n"
        f"RETURN ID(n) AS nodesID"
    )
    # Execute the Cypher query
    with driver.session() as session:
        result = session.run(query)
        node_ids = [record['nodesID'] for record in result]
    # Close the Neo4j driver
    driver.close()
    return node_ids


def get_relations_types_graphs(application, graph_type, graph_id):

    if graph_type == "DataGraph":
        relationship_type = "IS_IN_DATAGRAPH"
    elif graph_type == "Transaction":
        relationship_type = "IS_IN_TRANSACTION"
    else:
        return print("generate_cypher_query is build for DataGraph or Transaction")
    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=(user, password), database=database_name)
    # Build the Cypher query
    query = (f"""
            CALL cast.linkTypes(["CALL_IN_TRAN"]) yield linkTypes
            WITH linkTypes + [] AS updatedLinkTypes //"EXEC", "RELYON"
            MATCH (d:{graph_type}:{application})<-[:{relationship_type}]-(n)
            WITH collect(id(n)) AS nodeIds,updatedLinkTypes
            MATCH p=(d:{graph_type}:{application})<-[:{relationship_type}]-(n:{application})<-[r]-(m:{application})-[:{relationship_type}]->(d)
            WHERE ID(d) = {graph_id}
            AND (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND id(n) IN nodeIds AND id(m) IN nodeIds
            AND type(r) IN updatedLinkTypes
            RETURN DISTINCT type(r) as relationsTypes
            """
             )
    # Execute the Cypher query
    with driver.session() as session:
        result = session.run(query)
        relations_types = [record['relationsTypes'] for record in result]
    # Close the Neo4j driver
    driver.close()
    return relations_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Leiden community detection on Neo4j graph")
    parser.add_argument("application", type=str, help="Application name")
    parser.add_argument("graph_id", type=int, help="Graph ID")
    parser.add_argument("graph_type", type=str, choices=["DataGraph", "Transaction"], help="Graph type")
    parser.add_argument("linkTypes", nargs="+", type=str, help="List of link types considered in community detection")

    args = parser.parse_args()

    application = args.application
    graph_id = args.graph_id
    graph_type = args.graph_type
    linkTypes = args.linkTypes

    print(application)

    start_time = time.time()

    # Call Leiden_on_one_graph with parsed arguments
    Leiden_on_one_graph(application, graph_id, graph_type, linkTypes)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for the Leiden Algorithm on the {application} application: {elapsed_time} seconds")
