import networkx as nx
import community
from neo4j_connector_nx import Neo4jGraph


# ATTENTION DE CHOSIR LE BON URI - ICI VM LINUX DONC IP WINDOWS
# uri = "http://localhost:7474/browser/"
# uri = "bolt://localhost:7687"
uri = "bolt://172.24.144.1:7687"
user = "neo4j"
password = "imaging"
database_name = "neo4j"

# Crée une instance de la classe Neo4jGraph
neo4j_graph = Neo4jGraph(uri, user, password, database=database_name)

# Cypher query to retrieve the graph
cypher_query = (
    "CALL cast.linkTypes([\"CALL_IN_TRAN\"]) yield linkTypes\n"
    "MATCH (d:DataGraph:ecommerce)<-[:IS_IN_DATAGRAPH]-(n)\n"
    "WITH collect(id(n)) AS nodeIds,linkTypes\n"
    "MATCH p=(d:DataGraph:ecommerce {Name: 'PRODUCTS'})<-[:IS_IN_DATAGRAPH]-(n:ecommerce)<-[r]-(m:ecommerce)-[:IS_IN_DATAGRAPH]->(d)\n"
    "WHERE (n:Object OR n:SubObject)\n"
    "AND (m:Object OR m:SubObject)\n"
    "AND id(n) IN nodeIds AND id(m) IN nodeIds\n"
    "AND type(r) IN linkTypes\n"
    "RETURN DISTINCT n, r, m"
)

# Retrieve the graph based on the Cypher query
G = neo4j_graph.get_graph(cypher_query)

# Close the connection to Neo4j
neo4j_graph.close()

# Graph successfully imported
print('Graph successfully imported from Neo4j')

# Create a graph
# G = nx.karate_club_graph()

# Identify nodes of interest (start and end points)
start_nodes = [node for node in G.nodes if G.nodes[node].get('DgStartPoint') == "start"]
end_nodes = [node for node in G.nodes if G.nodes[node].get('DgEndPoint') == "end"]
print("Number of start nodes:", len(start_nodes))
print("Number of end nodes:", len(end_nodes))

# G.subgraph(set(G.nodes) - set(start_nodes + end_nodes)

# Detect communities using Louvain method
partition = community.best_partition(G.subgraph(set(G.nodes) - set(start_nodes + end_nodes)))
dendrogram = community.generate_dendrogram(G.subgraph(set(G.nodes) - set(start_nodes + end_nodes)))

# Add community attributes to nodes for each level in the dendrogram
for level in range(len(dendrogram)-1):
    # communities_at_level = community.partition_at_level(dendrogram, level)
    communities_at_level = community.partition_at_level(dendrogram, len(dendrogram)-1 - level)
    for node, community_id in communities_at_level.items():
        G.nodes[node][f'community_level_{level}'] = community_id

# Print the number of communities
num_communities = len(set(partition.values()))
print(f"Number of communities at level 0: {num_communities}")

# Print the number of levels
num_levels = len(dendrogram)
print(f"Number of levels: {num_levels}")

# Print all nodes and their attributes
print("All Nodes and Their Attributes:")
# for node, attributes in G.nodes(data=True):
for node, attributes in list(G.nodes(data=True))[:5]:
    print(f"Node {node}: {attributes}")
"""
print("\n")
print (dendrogram)
print("\n")
for level in range(len(dendrogram)) :
    print("Partition at level", level, "is", community.partition_at_level(dendrogram, level))
    """

# Iterate through nodes and print those without 'community_level_0' attribute
nodes_without_attribute = [node for node in G.nodes if 'community_level_0' not in G.nodes[node]]
print("Nodes without 'community_level_0' attribute:", nodes_without_attribute)
print(len(nodes_without_attribute))
