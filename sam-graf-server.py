import logging
from colorama import Fore, Style, Back
import os
from neo_query import NeoQuery

from flask import Flask, Response, request


app = Flask(__name__)
__version__ = "1.0.0-alpha.2"

URI = os.environ.get("NEO4J_URI")
# URI = "bolt://172.24.144.1:7687"
AUTH = ("neo4j", "imaging")
DATABASE = "neo4j"

msg = f"""
Running Sam-Graf-Server version {__version__} with URI: {URI}.
Use .env file to configure it
"""
print(Fore.BLACK + Back.WHITE + msg + Style.RESET_ALL)


@app.route('/Applications', methods=['GET'])
def get_applications():
    logging.info("Getting applications")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = "MATCH (a:Application) RETURN a.Name AS appName ORDER BY appName"
    return my_query.execute_query(query)


@app.route('/Applications/<app_name>/DataGraphs', methods=['GET'])
def get_datagraphs(app_name):
    logging.info("Getting Datagraphs")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    #query = f"MATCH (a:DataGraph:{app_name}) RETURN elementId(a) AS id, a.Name AS name ORDER BY name"
    query = f"MATCH (a:DataGraph:{app_name}) RETURN ID(a) AS id, a.Name AS name ORDER BY name"
    return my_query.execute_query(query)


@app.route('/Applications/<app_name>/Transactions', methods=['GET'])
def get_transactions(app_name):
    logging.info("Getting Transactions")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = f"MATCH (a:Transaction:{app_name}) RETURN ID(a) AS id, a.Name AS name ORDER BY name"
    return my_query.execute_query(query)


@app.route('/Applications/<app_name>/Objects', methods=['GET', 'POST'])
def get_objects(app_name):
    logging.info("Getting objects for " + app_name)
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = """
        MATCH (o1:Object:""" + app_name + """)-[r]->(o2:Object:""" + app_name + """)
        RETURN o1, o2, r """
    if request.method == 'POST':
        limit = int(request.args.get("limit", 100))
        query = query + " LIMIT $limit"
        return my_query.execute_query(query, limit)
    else:
        return my_query.execute_query(query)


def __graphs_query(app_name: str, graph_type: str, relationship_type: str, element_id: int) -> str:
    return (
        f"CALL cast.linkTypes([\"CALL_IN_TRAN\"]) yield linkTypes\n"
        f"WITH linkTypes + [\"STARTS_WITH\", \"ENDS_WITH\"] AS updatedLinkTypes\n"
        f"MATCH (d:{graph_type}:{app_name})<-[:{relationship_type}]-(n)\n"
        f"WITH collect(id(n)) AS nodeIds,updatedLinkTypes\n"
        f"MATCH p=(d:{graph_type}:{app_name})<-[:{relationship_type}]-(n:{app_name})<-[r]-(m:{app_name})-[:{relationship_type}]->(d)\n"
        #f"WHERE elementId(d) = '{element_id}'\n"
        f"WHERE ID(d) = {element_id}\n"
        f"AND (n:Object OR n:SubObject)\n"
        f"AND (m:Object OR m:SubObject)\n"
        f"AND id(n) IN nodeIds AND id(m) IN nodeIds\n"
        f"AND type(r) IN updatedLinkTypes\n"
        "RETURN DISTINCT n, r, m"
    )


@app.route('/Applications/<app_name>/Transactions/<element_id>', methods=['GET'])
def get_transaction(app_name, element_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __graphs_query(app_name, "Transaction", "IS_IN_TRANSACTION", element_id)
    return my_query.execute_query(cypher_query)


@app.route('/Applications/<app_name>/DataGraphs/<element_id>', methods=['GET'])
def get_datagraph(app_name, element_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __graphs_query(app_name, "DataGraph", "IS_IN_DATAGRAPH", element_id)
    return my_query.execute_query(cypher_query)


# Name of the attributes for each nodes in Ne4j :
# community_level_{level}_{model}_{graph_type}_{graph_name}

@app.route('/Applications/<app_name>/<model>/<graph_type>/<graph_name>/Level/<level_number>', methods=['GET'])
def get_level(app_name, level_number, model, graph_type, graph_name):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    level_label = f"community_level_{level_number}_{model}_{graph_type}_{graph_name}"
    cypher_query = """
        MATCH p=(l1:""" + level_label + """:""" + app_name + """)-[r]->(l2:""" + level_label + """:""" + app_name + """)
        RETURN p """
    # limit = int(request.args.get("limit", 100))
    # return my_query.execute_query(query, limit)
    return my_query.execute_query(cypher_query)
