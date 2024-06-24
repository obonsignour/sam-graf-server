from textwrap import dedent
from typing import List, LiteralString, Union, cast
from neo4j import GraphDatabase, Record, Result, ResultSummary, RoutingControl
from neo4j.graph import Node, Relationship, Path
from ogma_types import OgmaEdge, OgmaGraph, OgmaNode

NodeOrRel = Union[Node, Relationship]


class NeoQuery:
    """A class for executing queries on a Neo4j database."""

    def __init__(self, uri, auth, database):
        self.__URI = uri
        self.__AUTH = auth
        self.__DATABASE = database

    def __sanitize_query(self, query: str) -> LiteralString:
        return cast(LiteralString, dedent(query).strip())

    def execute_query(self, query: str, limit: int = 100):
        sanitized_query = self.__sanitize_query(query)
        print('URI: ', self.__URI)
        with GraphDatabase.driver(self.__URI, auth=self.__AUTH) as driver:
            driver.verify_connectivity()
            records, summary = driver.execute_query(sanitized_query, limit=limit, database_=self.__DATABASE,
                                                    routing_=RoutingControl.READ, result_transformer_=self.__transform_result)

            print("Query `{query}` returned {records_count} records in {time} ms.".format(
                query=summary.query, records_count=len(records),
                time=summary.result_available_after
            ))
            # no need toj sonify the response as flask is doing it for us in the view function that has called this method
            return (records)

    def __transform_result(self, result: Result) -> tuple[Union[dict, list], ResultSummary]:
        """A custom transformer. Transforms the result of a query into a graph or a list."""
        record = result.peek()
        if record == None:
            return [], result.consume()
        elif self.__is_a_graph(record):
            return self.__get_graph(result)
        else:
            return self.__get_list(result)

    def __is_a_graph(self, record: Record) -> bool:
        item = record[0]
        return (type(item) == Path or type(item) == Node or type(item) == Relationship)

    def __get_list(self, result: Result) -> tuple[list, ResultSummary]:
        data: List = []
        for record in result:
            __record_row = {}
            for key in record.keys():
                __record_row[key] = record[key]
            data.append(__record_row)
        return data, result.consume()

    def __get_graph(self, result: Result) -> tuple[dict, ResultSummary]:
        graph = result.graph()
        nodes: set[OgmaNode] = set()
        edges: set[OgmaEdge] = set()

        for node in graph.nodes:
            nodes.add(self.__get_ogma_node(node))
        for relationship in graph.relationships:
            edges.add(self.__get_ogma_edge(relationship))
        summary = result.consume()
        return OgmaGraph(list(nodes), list(edges)).to_dict(), summary

    def __get_ogma_node(self, item: Node):
        labels = list(item.labels)
        properties = {key: item[key] for key in item.keys()}
        return OgmaNode(item.id, labels, properties)

    def __get_ogma_edge(self, item: Relationship):
        source = cast(Node, item.start_node).id
        target = cast(Node, item.end_node).id
        return OgmaEdge(item.id, source, target, item.type, {key: item[key] for key in item.keys()})
