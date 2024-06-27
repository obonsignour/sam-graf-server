import logging
import threading
import time
from colorama import Fore, Style, Back
import os

from AlgoToTest.Leiden_onFly2 import Leiden_on_one_graph
from neo_query import NeoQuery
import subprocess
import paramiko
import json
from flask import Flask, Response, request, jsonify


app = Flask(__name__)
__version__ = "1.0.0-alpha.2"

try:
    URI = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    AUTH = (user, password)
    DATABASE = os.environ.get("NEO4J_DATABASE")
except KeyError:
    print("Error: OpenAI API key or Neo4j credentials not found in environment variables.")
    exit(1)

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
    # query = f"MATCH (a:DataGraph:{app_name}) RETURN elementId(a) AS id, a.Name AS name ORDER BY name"
    query = f"MATCH (a:DataGraph:{app_name}) RETURN id(a) AS id, a.Name AS name ORDER BY name"
    return my_query.execute_query(query)


@app.route('/Applications/<app_name>/Transactions', methods=['GET'])
def get_transactions(app_name):
    logging.info("Getting Transactions")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = f"MATCH (a:Transaction:{app_name}) RETURN id(a) AS id, a.Name AS name ORDER BY name"
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


@app.route('/Applications/<app_name>/Transactions/<element_id>', methods=['GET'])
def get_transaction(app_name, element_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __graphs_query(app_name, "Transaction", element_id)
    return my_query.execute_query(cypher_query)


@app.route('/Applications/<app_name>/DataGraphs/<element_id>', methods=['GET'])
def get_datagraph(app_name, element_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __graphs_query(app_name, "DataGraph", element_id)
    print('URI: ', URI)
    return my_query.execute_query(cypher_query)


# Name of the attributes for each nodes in Ne4j :
# community_level_{level}_{model}_{graph_type}_{graph_name}

# @app.route('/Applications/<app_name>/<model>/<graph_type>/<graph_name>/Level/<level_number>', methods=['GET'])
# def get_level(app_name, level_number, model, graph_type, graph_name):
#    my_query = NeoQuery(URI, AUTH, DATABASE)
#    level_label = f"community_level_{level_number}_{model}_{graph_type}_{graph_name}"
#    cypher_query = """
#        MATCH p=(l1:""" + level_label + """:""" + app_name + """)-[r]->(l2:""" + level_label + """:""" + app_name + """)
#        RETURN p """
#    # limit = int(request.args.get("limit", 100))
#    # return my_query.execute_query(query, limit)
#    return my_query.execute_query(cypher_query)

# APIs for the search


@app.route('/Applications/<app_name>/Concepts', methods=['GET'])
def get_concepts(app_name):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = f"""
        MATCH (n:{app_name}) WHERE (n:Object OR n:SubObject)
        WITH collect(DISTINCT n.InternalType) AS types
        MATCH (i:InternalType) WHERE i.Name IN types
        RETURN DISTINCT i.Concept AS name, count(i) AS count ORDER BY name """
    # limit = int(request.args.get("limit", 100))
    # return my_query.execute_query(query, limit)
    return my_query.execute_query(cypher_query)

# CLEAN CODE TO KEEP


def __graphs_query(app_name: str, graph_type: str, graph_id: int) -> str:
    _relationship_type = "IS_IN_" + graph_type.upper()
    return (f"""
        MATCH p = (d:{graph_type}:{app_name})<-[i1:{_relationship_type}]-(n:{app_name})
        WHERE id(d) = {graph_id}
        AND (n:Object OR n:SubObject)
        WITH n, apoc.map.setLists(properties(n), ['UndirectedLouvain', 'DirectedLouvain', 'Leiden'], [i1.UndirectedLouvain, i1.DirectedLouvain, i1.Leiden]) AS propsCompleted
        WITH apoc.create.vNode(labels(n), propsCompleted) AS n1
        WITH collect(n1) AS vNodes, collect(apoc.any.property(n1, "AipId")) AS aipIds
        CALL cast.linkTypes([\"CALL_IN_TRAN\"]) yield linkTypes
        WITH vNodes, aipIds, linkTypes
        MATCH (n:{app_name})<-[r]-(m:{app_name})
        WHERE (n:Object OR n:SubObject)
        AND (m:Object OR m:SubObject)
        AND n.AipId IN aipIds
        AND m.AipId IN aipIds
        AND type(r) IN linkTypes
        WITH n, m, r, vNodes
        WITH [n1 IN vNodes WHERE apoc.any.property(n1, "AipId")=n.AipId | n1][0] AS n1,  [m1 IN vNodes WHERE apoc.any.property(m1, "AipId")=m.AipId | m1][0] AS m1, r
        WITH n1, m1, apoc.create.vRelationship(n1, type(r), properties(r), m1) AS r1
        RETURN n1, m1, r1
        """
            )


@app.route('/Applications/<app_name>/Graphs/<graph_type>/<graph_id>', methods=['GET'])
def get_graph(app_name, graph_type, graph_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __graphs_query(app_name, graph_type, graph_id)
    return my_query.execute_query(cypher_query)


@app.route('/Algos/<algo>/Compute', methods=['POST'])
def compute_algo_with_link_types(algo):
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
    except Exception as e:
        return jsonify({"error": "No data or data isn't a json" + str(e)}), 500

    _app_name = data['appName']
    _graph_type = data['graphType']
    _graph_id = data['graphId']
    _link_types = data['linkTypes']
    print(f"Computing Leiden with {data}")
    thread = threading.Thread(target=Leiden_on_one_graph, args=(_app_name, _graph_id, _graph_type, _link_types))
    thread.start()
    _task_id = thread.ident
    print(f"Thread {thread} started for task {_task_id}")
    return jsonify({"message": "Algo computing started", "application": _app_name, "taskId": _task_id}), 202


@app.route('/Algos/Tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    nb_tasks_running = threading.active_count()
    tasks = threading.enumerate()
    for task in tasks:
        if str(task.ident) == task_id:
            print(f"Task {task_id} is still running")
            return jsonify({"message": "Task is still running", "taskId": task_id}), 202
    print(f"Task {task_id} has been completed")
    return jsonify({"message": "Task has been completed", "taskId": task_id}), 200


@app.route('/Algos/Threads', methods=['GET'])
def get_threads():
    nb_tasks_running = threading.active_count()
    tasks = threading.enumerate()
    tasks_info = [{"id": task.ident, "name": task.name} for task in tasks]
    return jsonify({"message": "List of running tasks", "nbTasks": nb_tasks_running, "tasks": tasks_info}), 200
# END CLEAN CODE TO KEEP


def __linkTypes_query(app_name: str, graph_type: str, relationship_type: str, element_id: int) -> str:
    query = f"""
        CALL cast.linkTypes(["CALL_IN_TRAN"]) yield linkTypes
        WITH linkTypes + [] AS updatedLinkTypes //"EXEC", "RELYON"
        MATCH (d:{graph_type}:{app_name})<-[:{relationship_type}]-(n)
        WITH collect(id(n)) AS nodeIds,updatedLinkTypes
        MATCH p=(d:{graph_type}:{app_name})<-[:{relationship_type}]-(n:{app_name})<-[r]-(m:{app_name})-[:{relationship_type}]->(d)
        WHERE ID(d) = {element_id}
        AND (n:Object OR n:SubObject)
        AND (m:Object OR m:SubObject)
        AND id(n) IN nodeIds AND id(m) IN nodeIds
        AND type(r) IN updatedLinkTypes
        RETURN DISTINCT type(r) AS relationType ORDER BY relationType
        """
    return query


@app.route('/Applications/<app_name>/Transactions/<element_id>/LinkTypes', methods=['GET'])
def get_linkTypes_transaction(app_name, element_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __linkTypes_query(app_name, "Transaction", "IS_IN_TRANSACTION", element_id)
    return my_query.execute_query(cypher_query)


@app.route('/Applications/<app_name>/DataGraphs/<element_id>/LinkTypes', methods=['GET'])
def get_linkTypes_datagraph(app_name, element_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __linkTypes_query(app_name, "DataGraph", "IS_IN_DATAGRAPH", element_id)
    return my_query.execute_query(cypher_query)


# To store the data received from the POST request
stored_data = {}


@app.route('/Applications/<app_name>/<graph_types>/<element_id>/LinkTypes/Selected', methods=['POST', 'GET'])
def get_selected_linkTypes(app_name, graph_types, element_id):
    if graph_types == "DataGraphs":
        graph_type = "DataGraph"
    elif graph_types == "Transactions":
        graph_type = "Transaction"

    if request.method == 'POST':
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        stored_data[(app_name, graph_type, element_id)] = data
        print(f"Received data for {app_name} {graph_type} {element_id}: {data}")

        # Call submit_items function
        return submit_items(app_name, graph_type, element_id, data)
        # return submit_items(stored_data)

    elif request.method == 'GET':
        data = stored_data.get((app_name, graph_type, element_id), {})
        print(f"Retrieved data for {app_name} {graph_type} {element_id}: {data}")
        sample_data = {
            "app_name": app_name,
            "graph_type": graph_type,
            "element_id": element_id,
            "data": data
        }
        return jsonify(sample_data), 200

    # Default return statement to ensure a Response object is always returned
    return jsonify({"error": "Invalid request method"}), 400

# def submit_items(app_name, graph_type, element_id, data):


def submit_items(app_name, graph_type, element_id, data):
    # print("I'm in submit_items")
    # print(data)
    """
    # Extracting keys and values
    key = next(iter(data))  # This gets the first (and only) key in the dictionary
    value = data[key]       # This gets the corresponding value

    # Extracting values from the key tuple
    app_name = key[0]
    graph_type = key[1]
    element_id = key[2]

    # Extracting relationTypesSelected from the value dictionary
    relation_types = value['relationTypesSelected']
    """
    relation_types = data['relationTypesSelected']

    """
    # Printing extracted values
    print(f"app_name: {app_name}")
    print(f"graph_type: {graph_type}")  
    print(f"element_id: {element_id}")    
    print(f"relation_types: {relation_types}")
    """

    # Trigger the script execution
    result = run_script(app_name, element_id, graph_type, relation_types)
    return jsonify({'status': 'success', 'result': result})


def run_script(app_name, element_id, graph_type, relation_types):
    # print("I'm in run_script")

    try:
        # VM credentials
        hostname = '172.16.20.137'
        port = 22
        username = 'smm'

        # Construct the command to run the script on the VM
        relation_types_str = ' '.join(relation_types)

        print("--- Args used to run the aglo ---")
        print(f"app_name: {app_name}")
        print(f"graph_type: {graph_type}")
        print(f"element_id: {element_id}")
        print(f"relation_types: {relation_types}")

        command = (
            # f"source /path/to/myenv/bin/activate && "
            f"conda activate myenv \n"
            f"python /home/smm/PhD/Grouping/CommunityDetection/OnFly/Leiden_onFly2.py "
            f"{app_name} {element_id} {graph_type} {relation_types_str}"
        )

        print("Before subprocess SSH command execution")

        # Execute the command using subprocess
        process = subprocess.Popen(
            ['ssh', f'{username}@{hostname}', '-p', str(port), command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate()
        print("Done")

        if stderr:
            print(f"Error: {stderr.decode('utf-8')}")
            return f"Error: {stderr.decode('utf-8')}"

        print(f"Output: {stdout.decode('utf-8')}")
        return stdout.decode('utf-8')

    except Exception as e:
        print(e)
        return str(e)
