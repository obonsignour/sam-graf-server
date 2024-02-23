from json import dumps
from textwrap import dedent
from typing import Literal, LiteralString, cast
from flask import Flask, Response, jsonify, request
from neo4j import GraphDatabase, RoutingControl, Record
from neo4j.graph import Node, Relationship, Path
import neo4j

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "imaging")
DATABASE = "neo4j"

app = Flask(__name__)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()


def sanitize_query(q: LiteralString) -> LiteralString:
    # this is a safe transform:
    # no way for cypher injection by trimming whitespace
    # hence, we can safely cast to LiteralString
    return cast(LiteralString, dedent(q).strip())


@app.route('/<app_name>/Level5s', methods=['GET'])
def print_friends(app_name):
    query = sanitize_query("""
        MATCH p=(l1:Level5:""" + app_name + """)-[r]->(l2:Level5:""" + app_name + """)
        RETURN p LIMIT $limit""")

    queryo = sanitize_query("""
        MATCH (o1:Object:""" + app_name + """)-[r]->(o2:Object:""" + app_name + """)
        RETURN o1, o2, r, o1.Name LIMIT $limit""")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
    records, summary, _ = driver.execute_query(query, limit=request.args.get("limit", 100), database_=DATABASE, routing_=RoutingControl.READ,)

    data = []
    graph_data = {
        "nodes": [{
            "id": "node1",
            "data": {
                "key1": "value1",
                "key2": "value2",
            }
        }],
        "edges": [{id: 0, "source": 0, "target": 4, "data": {type: 'lives_in'}}]
    }

    nodes = []
    node_ids = []
    edges = []
    edge_ids = []

    for record in records:
        for item in record:
            if type(item) == Path:
                print('We have a path', item, item.relationships)
            if type(item) == Node:
                if item.id not in node_ids:
                    node = {"id": item.id, "data": {}}
                    node_data = {}
                    for key in item.keys():
                        node_data[key] = item[key]
                    node["data"] = node_data
                    nodes.append(node)
                    node_ids.append(item.id)

            if issubclass(type(item), Relationship):
                if item.id not in edge_ids:
                    edge = {"id": item.id, "source": item.start_node.id, "target": item.end_node.id, "data": {}}
                    data = {}
                    for key in item.keys():
                        data[key] = item[key]
                    edge["data"] = data
                    edges.append(edge)
                    edge_ids.append(item.id)

    print(len(nodes), ' ', len(edges))
    print(len(nodes), ' ', len(edges))

    graph_data["nodes"] = nodes
    graph_data["edges"] = edges
    # graph_data["nodes"].append(node)
    # graph_data["edges"].append(edge)

    return jsonify(graph_data)
    # return jsonify(data)

# def create_node(id, data):


if __name__ == '__main__':
    app.run(debug=True)
