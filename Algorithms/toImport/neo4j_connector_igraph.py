from neo4j import GraphDatabase
from igraph import Graph

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

            # Create an iGraph graph
            igraph_graph = Graph(directed=True)

            # Add nodes with attributes
            node_id_map = {}
            for node in neo4j_graph.nodes:
                node_properties = {key: node[key] for key in node.keys() if key not in ["id", "labels"]}
                #node_properties["id"] = node.id
                node_properties["labels"] = node.labels
                node_index = igraph_graph.add_vertex(id=str(node.id), **node_properties).index
                node_id_map[node.id] = node_index

            # Add edges with attributes
            for rel in neo4j_graph.relationships:
                source_index = node_id_map[rel.start_node.id]
                target_index = node_id_map[rel.end_node.id]
                attributes = {key: rel[key] for key in rel.keys() if key not in ["id", "type"]}
                attributes["id"] = rel.id
                attributes["type"] = rel.type
                igraph_graph.add_edge(source_index, target_index, **attributes)

            return igraph_graph


