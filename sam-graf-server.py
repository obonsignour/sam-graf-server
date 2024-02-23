from neo_query import NeoQuery

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


@app.route('/<app_name>/Level5s', methods=['GET'])
def get_level5s(app_name):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = """
        MATCH p=(o1:Object:""" + app_name + """)-[r]->(o2:Object:""" + app_name + """)
        RETURN o1, o2, r LIMIT $limit"""
    limit = int(request.args.get("limit", 100))
    return my_query.execute_query(query, limit)


if __name__ == '__main__':
    app.run(debug=True)
