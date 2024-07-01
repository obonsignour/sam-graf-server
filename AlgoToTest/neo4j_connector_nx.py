from neo4j import GraphDatabase
import networkx as nx

class Neo4jGraph:
    def __init__(self, uri, user, password, database=None):
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password), database=self._database)

    def close(self):
        self._driver.close()

    def get_graph(self, cypher_query):
        with self._driver.session() as session:
            result = session.run(cypher_query)
            neo4j_graph = result.graph()
            
            # Create a directed NetworkX graph
            G = nx.Graph()

            # Add nodes
            for node in neo4j_graph.nodes:
                node_properties = {key: node[key] for key in node.keys() if key not in ["id", "labels"]}
                G.add_node(node.id, **node_properties, id=node.id, labels=node.labels)

            # Add edges with attributes
            for rel in neo4j_graph.relationships:
                source = rel.start_node.id
                target = rel.end_node.id
                attributes = {key: rel[key] for key in rel.keys() if key not in ["id", "type"]}
                G.add_edge(source, target, **attributes, id=rel.id, type=rel.type)

            return G
