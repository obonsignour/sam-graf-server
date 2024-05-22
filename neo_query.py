from textwrap import dedent
from typing import List, LiteralString, Union, cast
from neo4j import GraphDatabase, Record, Result, ResultSummary, RoutingControl, READ_ACCESS
from neo4j.graph import Node, Relationship, Path
from ogma_types import OgmaEdge, OgmaGraph, OgmaNode
import json
import tempfile
import time

NodeOrRel = Union[Node, Relationship]


class NeoQuery:
    """A class for executing queries on a Neo4j database."""

    def __init__(self, uri, auth, database):
        self.__URI = uri
        self.__AUTH = auth
        self.__DATABASE = database
        self.__transformers = {
            'custom_graph': self.__transform_custom_graph,
            'decorated_graph': self.__transform_decorated_graph,
            'overall_architecture': self.__transform_overall_architecture,
            "list_of_nodes_and_rels": self.__transform_a_list,
            'regular': self.__transform_result
        }

    def run_query(self, query: str):
        start_time = time.time()
        with GraphDatabase.driver(self.__URI, auth=self.__AUTH) as driver:
            driver.verify_connectivity()
            with driver.session(database=self.__DATABASE, default_access_mode=READ_ACCESS) as session:
                query = self.__sanitize_query(query)
                print("Executing query: ", query)
                results = session.run(query)
                print(f"Query executed in {time.time() - start_time} ms.")

                # Case results is a list of custom nodes and rels
                nodes: set[OgmaNode] = set()
                edges: set[OgmaEdge] = set()

                try:
                    nodes_and_edges = results.single(strict=True)
                    for __node in nodes_and_edges['nodes']:
                        node = self.__get_ogma_node(__node)
                        nodes.add(node)
                    print(f"{len(nodes)} nodes consumed in {time.time() - start_time} ms.")
                    for __edge in nodes_and_edges['rels']:
                        edge = self.__get_ogma_edge(__edge)
                        edges.add(edge)
                    print(f"{len(edges)} edges consumed in {time.time() - start_time} ms.")

                    summary = results.consume()
                    print(f"{len(nodes)} nodes and {len(edges)} edges available after {summary.result_available_after} ms. Total time: {time.time() - start_time} ms.")
                    return OgmaGraph(list(nodes), list(edges)).to_dict()
                except Exception as e:
                    print(e)
                    return "No nodes returned", 500

    def __sanitize_query(self, query: str) -> LiteralString:
        return cast(LiteralString, dedent(query).strip())

    def __get_transformer_func(self, transformer: str):
        return self.__transformers.get(transformer, self.__transform_result)

    def execute_query(self, query: str, **kwargs) -> Union[dict, list]:
        start_time = time.time()
        limit = kwargs.get('limit', 0)
        transformer_func = self.__get_transformer_func(kwargs.get('transformer', 'regular'))
        sanitized_query = self.__sanitize_query(query)

        with GraphDatabase.driver(self.__URI, auth=self.__AUTH) as driver:
            driver.verify_connectivity()
            records, summary = driver.execute_query(sanitized_query, limit=limit, database_=self.__DATABASE,
                                                    routing_=RoutingControl.READ, result_transformer_=transformer_func)

            print("Query `{query}` returned {records_count} records available in {time} ms. Total time: {total_time}".format(
                query=summary.query, records_count=len(records),
                time=summary.result_available_after,
                total_time=time.time() - start_time
            ))

            # Specify the file path in the temp folder
            # export = tempfile.NamedTemporaryFile(delete=False, prefix='neo4j_export_', suffix='.json')
            # export.write(json.dumps(records).encode())

            # no need toj sonify the response as flask is doing it for us in the view function that has called this method
            return (records)

    def __transform_decorated_graph(self, result: Result) -> tuple[Union[dict, list], ResultSummary]:
        """A transformer for a query returning a decorated graph. """
        keys = result.keys()
        record = result.peek()
        if record == None or not self.__is_a_decorated_graph(keys):
            return self.__transform_result(result)
        else:
            return self.__get_decorated_graph(result)

    def __transform_overall_architecture(self, result: Result) -> tuple[Union[dict, list], ResultSummary]:
        """A transformer for a query returning the decorated graph for overall architecture. """
        keys = result.keys()
        record = result.peek()
        if record == None or not self.__is_a_decorated_graph(keys):
            return self.__transform_result(result)
        else:
            return self.__get_overall_architecture_graph(result)

    def __transform_a_list(self, results: Result) -> tuple[dict, ResultSummary]:
        """ transforms the results of a query returning a list of nodes and relations.
        Expects the resultset to be with two elements: nodes and rels."""
        nodes: set[OgmaNode] = set()
        edges: set[OgmaEdge] = set()

        for nodes_and_edges in results:
            for __node in nodes_and_edges['nodes']:
                node = self.__get_ogma_node(__node)
                nodes.add(node)
            for __edge in nodes_and_edges['rels']:
                edge = self.__get_ogma_edge(__edge)
                edges.add(edge)
        summary = results.consume()
        return OgmaGraph(list(nodes), list(edges)).to_dict(), summary

    def __transform_custom_graph(self, results: Result) -> tuple[dict, ResultSummary]:
        """A transformer for a query returning a collection of nodes and edges manually built in cypher. """
        start_time = time.time()
        # Case results is a list of custom nodes and rels
        nodes: set[OgmaNode] = set()
        edges: set[OgmaEdge] = set()

        nodes_and_edges = results.single(strict=True)
        for __node in nodes_and_edges['nodes']:
            node = self.__get_ogma_node(__node)
            nodes.add(node)
        print(f"{len(nodes)} nodes consumed in {time.time() - start_time} ms.")
        for __edge in nodes_and_edges['rels']:
            edge = self.__get_ogma_edge(__edge)
            edges.add(edge)
        print(f"{len(edges)} edges consumed in {time.time() - start_time} ms.")

        summary = results.consume()
        print(f"{len(nodes)} nodes and {len(edges)} edges available after {summary.result_available_after} ms. Total time: {time.time() - start_time} ms.")
        return OgmaGraph(list(nodes), list(edges)).to_dict(), summary

    def __transform_result(self, result: Result) -> tuple[Union[dict, list], ResultSummary]:
        """A custom transformer. Transforms the result of a query into a graph or a list."""
        record = result.peek()
        if record == None:
            return [], result.consume()
        elif self.__is_a_graph(record):
            return self.__get_graph(result)
        else:
            return self.__get_list(result)

    def __is_a_decorated_graph(self, keys) -> bool:
        if 'decorator' in keys:
            return True
        return False

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

    def __get_decorated_graph(self, result: Result) -> tuple[dict, ResultSummary]:
        nodes: set[OgmaNode] = set()
        edges: set[OgmaEdge] = set()
        keys = result.keys()
        if 'n' in keys:
            for record in result:
                if (record['n'] != None):
                    node = self.__get_ogma_node(record['n'])
                    if (record['decorator'] != None):
                        node.add_properties(record['decorator'])
                    nodes.add(node)
                if (record['r'] != None):
                    relationship = self.__get_ogma_edge(record['r'])
                    edges.add(relationship)
        if 'source' in keys:
            for record in result:
                if (record['source'] != None):
                    node = self.__get_ogma_node(record['source'])
                    if (record['decorator'] != None):
                        node.add_properties(record['decorator'])
                    nodes.add(node)
                if (record['rel'] != None):
                    relationship = self.__get_ogma_edge(record['rel'])
                    edges.add(relationship)
        summary = result.consume()
        return OgmaGraph(list(nodes), list(edges)).to_dict(), summary

    def __get_overall_architecture_graph(self, result: Result) -> tuple[dict, ResultSummary]:
        nodes: set[OgmaNode] = set()
        edges: set[OgmaEdge] = set()
        keys = result.keys()
        if 'n' in keys:
            for record in result:
                if (record['n'] is not None):
                    node = self.__get_ogma_node(record['n'])
                    node.keep_only(['AipId', 'Name', 'Type', 'InternalType'])
                    if (record['decorator'] is not None):
                        node.add_properties(record['decorator'])
                    nodes.add(node)
                if (record['r'] is not None):
                    relationship = self.__get_ogma_edge(record['r'])
                    edges.add(relationship)
        if 'source' in keys:
            for record in result:
                if (record['source'] is not None):
                    node = self.__get_ogma_node(record['source'])
                    if (record['decorator'] is not None):
                        node.add_properties(record['decorator'])
                    nodes.add(node)
                if (record['rel'] != None):
                    relationship = self.__get_ogma_edge(record['rel'])
                    edges.add(relationship)
        summary = result.consume()
        return OgmaGraph(list(nodes), list(edges)).to_dict(), summary

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
        if (type(item) == Node):
            labels = list(item.labels)
            properties = {key: item[key] for key in item.keys()}
            return OgmaNode(item.id, labels, properties)
        else:
            return OgmaNode(item['identity'], item['labels'], item['properties'])

    def __get_ogma_edge(self, item: Relationship):
        source = cast(Node, item.start_node).id
        target = cast(Node, item.end_node).id
        type = item.type
        data = {key: item[key] for key in item.keys() if item[key]}
        data = {'type': type, **data}
        return OgmaEdge(item.id, source, target, data)
