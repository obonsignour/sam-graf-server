from platform import node
from textwrap import dedent
from typing import Callable, List, LiteralString, Tuple, Union, cast
from flask import jsonify
from neo4j import GraphDatabase, RoutingControl
from neo4j.graph import Node, Relationship, Path

from ogma_types import OgmaEdge, OgmaNode

NodeOrRel = Union[Node, Relationship]


class NeoQuery:
    def __init__(self, uri, auth, database):
        self.__URI = uri
        self.__AUTH = auth
        self.__DATABASE = database
        self.__nodes = []
        self.__node_ids = []
        self.__edges = []
        self.__edge_ids = []

    def __sanitize_query(self, query: str) -> LiteralString:
        return cast(LiteralString, dedent(query).strip())

    def execute_query(self, query: str, limit: int = 100):
        sanitized_query = self.__sanitize_query(query)
        with GraphDatabase.driver(self.__URI, auth=self.__AUTH) as driver:
            driver.verify_connectivity()
        records, summary, _ = driver.execute_query(sanitized_query, limit=limit, database_=self.__DATABASE, routing_=RoutingControl.READ,)

        graph_data = {}
        for record in records:
            for item in record:
                if type(item) == Path:
                    print('We have a path', item, item.relationships)
                    # Raise a 5xx error here
                    raise Exception("Internal Server Error: the query returned a path. This is not supported.")
                if type(item) == Node:
                    self.__add_node(item)
                if issubclass(type(item), Relationship):
                    self.__add_edge(cast(Relationship, item))

        print(len(self.__nodes), ' ', len(self.__edges))
        print(len(self.__nodes), ' ', len(self.__edges))

        graph_data["nodes"] = self.__nodes
        graph_data["edges"] = self.__edges
        return jsonify(graph_data)

    def __add_node(self, node: Node):
        if node.id not in self.__node_ids:
            self.__nodes.append(self.__extract_node(node).to_dict())
            self.__node_ids.append(node.id)

    def __add_edge(self, edge: Relationship):
        if edge.id not in self.__edge_ids:
            self.__edges.append(self.__extract_edge(edge).to_dict())
            self.__edge_ids.append(edge.id)

    def __extract_node(self, item: Node) -> OgmaNode:
        return OgmaNode(item.id, self.__extract_data(item))

    def __extract_edge(self, item: Relationship) -> OgmaEdge:
        source = cast(Node, item.start_node).id
        target = cast(Node, item.end_node).id
        return OgmaEdge(item.id, source, target, self.__extract_data(item))

    def __extract_data(self, item: NodeOrRel) -> dict:
        data = {}
        for key in item.keys():
            data[key] = item[key]
        return data
