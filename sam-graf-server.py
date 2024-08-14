import logging
import threading
from colorama import Fore, Style, Back
import os

"""
from AlgoToTest.Leiden_onFly2 import Leiden_on_one_graph
from AlgoToTest.ULouvain_onFly2 import Undirected_Louvain_on_one_graph
from AlgoToTest.ULouvain_onFlyAppGraph import Undirected_Louvain_on_one_app
from AlgoToTest.DLouvain_onFly2 import Directed_Louvain_on_one_graph
from AlgoToTest.SLPA_onFly2 import SLPA_on_one_graph
"""

# import for call graph
from Algorithms.DGTransac.Leiden import Leiden_Call_Graph
from Algorithms.DGTransac.DLouvainV2 import Directed_Louvain_Call_Graph
from Algorithms.DGTransac.ULouvainV2 import Undirected_Louvain_Call_Graph

# import for app graph
from Algorithms.FullApp.LeidenApp import Leiden_App_Graph
from Algorithms.FullApp.DLouvainApp import Directed_Louvain_App_Graph
from Algorithms.FullApp.ULouvainApp import Undirected_Louvain_App_Graph

from neo_query import NeoQuery
import subprocess
from flask import Flask, request, jsonify

from query_texts import get_callgraph_query, callgraph_query, appgraph_query, modelgraph_query


app = Flask(__name__)
__version__ = "0.1.0"

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
    """
    GET /Applications
    Fetches a list of all applications from the Neo4j database.

    Returns:
        JSON: A list of application names.
    """
    logging.info("Getting applications")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = "MATCH (a:Application) RETURN a.Name AS appName ORDER BY appName"
    return my_query.execute_query(query)


@app.route('/Applications/<app_name>/Concepts', methods=['GET'])
def get_concepts(app_name):
    """
    GET /Applications/<app_name>/Concepts
    Fetches concepts related to the specified application from the Neo4j database. 
    Concepts are categories to define the type of object

    Args:
        app_name (str): The name of the application. Case sensitive.

    Returns:
        JSON: A list of concepts and their counts.
    """
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = f"""
        MATCH (n:{app_name}) WHERE (n:Object OR n:SubObject)
        WITH collect(DISTINCT n.InternalType) AS types
        MATCH (i:InternalType) WHERE i.Name IN types
        RETURN DISTINCT i.Concept AS name, count(i) AS count ORDER BY name """
    # limit = int(request.args.get("limit", 100))
    # return my_query.execute_query(query, limit)
    return my_query.execute_query(cypher_query)


@app.route('/Applications/<app_name>', methods=['GET'])
def get_appgraph(app_name):
    """
    GET /Applications/<app_name>
    Fetches the application graph for the specified application from the Neo4j database.

    Args:
        app_name (str): The name of the application. Case sensitive.

    Returns:
        JSON: The application graph.
    """
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = appgraph_query(app_name)
    print('URI: ', URI)
    return my_query.execute_query(cypher_query)


@app.route('/Applications/<app_name>/Transactions', methods=['GET'])
def get_transactions(app_name):
    """
    GET /Applications/<app_name>/Transactions
    Fetches transactions related to the specified application from the Neo4j database.

    Args:
        app_name (str): The name of the application. Case sensitive.

    Returns:
        JSON: A list of transactions. (id and name of the transactions, ordered by name).
    """
    logging.info("Getting tranasctions")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = get_callgraph_query(app_name, 'Transaction')
    return my_query.execute_query(query)


@app.route('/Applications/<app_name>/DataGraphs', methods=['GET'])
def get_datagraphs(app_name):
    """
    GET /Applications/<app_name>/DataGraphs
    Fetches data graphs related to the specified application from the Neo4j database.

    Args:
        app_name (str): The name of the application. Case sensitive.

    Returns:
        JSON: A list of data graphs (id and name of the datagraphs, ordered by name).
    """
    logging.info("Getting DataGraphs")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = get_callgraph_query(app_name, 'DataGraph')
    return my_query.execute_query(query)


@app.route('/Applications/<app_name>/Models', methods=['GET'])
def get_models(app_name):
    """
    GET /Applications/<app_name>/Models
    Fetches models related to the specified application from the Neo4j database.

    Args:
        app_name (str): The name of the application. Case sensitive.

    Returns:
        JSON: A list of models and their link types ordered by model name.
    """
    logging.info("Getting Models")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = f"MATCH (a:Model:{app_name}) RETURN a.name AS modelName, a.LinkTypes AS linkTypes ORDER BY modelName"
    return my_query.execute_query(query)


def _get_a_callgraph(app_name, callgraph_type, callgraph_id):
    """
    Fetches a specific call graph for the specified application from the Neo4j database.

    Args:
        app_name (str): The name of the application. Case sensitive.
        callgraph_type (str): The type of the call graph.
        callgraph_id (str): The ID of the call graph.

    Returns:
        JSON: The specified call graph.
    """
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = callgraph_query(app_name, callgraph_type, callgraph_id)
    return my_query.execute_query(cypher_query)


@app.route('/Applications/<app_name>/Transactions/<graph_id>', methods=['GET'])
def get_a_transaction(app_name, graph_id):
    """
    GET /Applications/<app_name>/Transactions/<graph_id>
    Fetches a specific transaction for the specified application from the Neo4j database.

    Args:
        app_name (str): The name of the application. Case sensitive.
        graph_id (str): The ID of the transaction graph.

    Returns:
        JSON: The specified transaction graph.
    """
    return _get_a_callgraph(app_name, 'Transaction', graph_id)


@app.route('/Applications/<app_name>/DataGraphs/<graph_id>', methods=['GET'])
def get_a_datagraph(app_name, graph_id):
    """
    GET /Applications/<app_name>/Transactions/<graph_id>
    Fetches a specific transaction for the specified application from the Neo4j database.

    Args:
        app_name (str): The name of the application. Case sensitive.
        graph_id (str): The ID of the transaction graph.

    Returns:
        JSON: The specified transaction graph.
    """
    return _get_a_callgraph(app_name, 'DataGraph', graph_id)


@app.route('/Applications/<app_name>/Models/<model_name>', methods=['GET'])
def get_a_model(app_name, model_name):
    """
    GET /Applications/<app_name>/Models/<model_name>
    Fetches an instance of model: the graph resulting from the computation of a similarity model for a given graph.

    Args:
        app_name (str): The name of the application. Case sensitive.
        model_name (str): The name of the instance of the model.

    Returns:
        JSON: The specified graph.
    """
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = modelgraph_query(app_name, model_name)
    return my_query.execute_query(cypher_query)


# ========================================================================================================================================
# Algo APIs
# ========================================================================================================================================
def _get_function_name(algorithm, graph_type):
    # Define the mapping dictionary
    algo_function_prefixes = {
        'Leiden': "Leiden",
        'UndirectedLouvain': "Undirected_Louvain",
        'DirectedLouvain': "Directed_Louvain",
        'SLPA': "SLPA",
    }
    if graph_type == "Application":
        scope = "_App_Graph"
    else:
        scope = "_Call_Graph"

    function_name = f"{algo_function_prefixes[algorithm]}{scope}"
    return function_name


@app.route('/Algos/<algo>/Compute', methods=['POST'])
def compute_algo_with_link_types(algo):
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
    except Exception as e:
        return jsonify({"error": "No data or data isn't a json: " + str(e)}), 500

    _app_name = data['appName']
    _graph_type = data['graphType']
    _graph_id = data['graphId']
    _link_types = data['linkTypes']

    # Retrieve the function from the dictionary
    # algo_function = algo_function_prefixes.get(algo)
    algo_function_name = _get_function_name(algo, _graph_type)
    print(f"Algo function: {algo_function_name}")

    if not globals()[algo_function_name]:
        return jsonify({"error": f"Algorithm {algo} is not supported"}), 400

    print(f"Computing {algo} with {data}")
    if (_graph_type == "Application"):
        print(f"Running {algo} on {_app_name} application")
        thread = threading.Thread(target=globals()[algo_function_name], args=(_app_name, _link_types))
    else:
        thread = threading.Thread(target=globals()[algo_function_name], args=(_app_name, _graph_id, _graph_type, _link_types))
    thread.start()
    _task_id = thread.ident
    print(f"Thread {thread} started for task {_task_id}")
    return jsonify({"message": f"Algo {algo} computing started", "application": _app_name, "taskId": _task_id}), 202


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


# ========================================================================================================================================
# ========================================================================================================================================
# Previous version of the APIs
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


def __linkTypes_query_graphs(app_name: str, graph_type: str, relationship_type: str, element_id: int) -> str:
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


def __linkTypes_query_appgraph(app_name: str) -> str:
    query = f"""
        CALL cast.linkTypes(['CALL_IN_TRAN']) yield linkTypes
        WITH linkTypes + [] AS updatedLinkTypes
        MATCH p=(n:{app_name})<-[r]-(m:{app_name})
        WHERE (n:Object OR n:SubObject)
        AND (m:Object OR m:SubObject)
        AND type(r) IN updatedLinkTypes
        RETURN DISTINCT type(r) AS relationType ORDER BY relationType
        """
    return query


@app.route('/Applications/<app_name>/Transactions/<element_id>/LinkTypes', methods=['GET'])
def get_linkTypes_transaction(app_name, element_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __linkTypes_query_graphs(app_name, "Transaction", "IS_IN_TRANSACTION", element_id)
    return my_query.execute_query(cypher_query)


@app.route('/Applications/<app_name>/DataGraphs/<element_id>/LinkTypes', methods=['GET'])
def get_linkTypes_datagraph(app_name, element_id):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __linkTypes_query_graphs(app_name, "DataGraph", "IS_IN_DATAGRAPH", element_id)
    return my_query.execute_query(cypher_query)


@app.route('/Applications/<app_name>/ApplicationGraph/LinkTypes', methods=['GET'])
def get_linkTypes_appgraph(app_name):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    cypher_query = __linkTypes_query_appgraph(app_name)
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
