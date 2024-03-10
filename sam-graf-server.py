import logging
from webbrowser import get
from neo_query import NeoQuery

from flask import Flask, Response, request

# URI = "neo4j://localhost:7687"
URI = "bolt://172.24.144.1:7687"
AUTH = ("neo4j", "imaging")
DATABASE = "neo4j"

app = Flask(__name__)
__version__ = "1.0.0-alpha.2"


@app.route('/Applications', methods=['GET'])
def get_applications():
    logging.info("Getting applications")
    my_query = NeoQuery(URI, AUTH, DATABASE)
    query = "MATCH (a:Application) RETURN a.Name AS appName ORDER BY appName"
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
        query = query + " LIMIT 50"
        return my_query.execute_query(query)


@app.route('/Applications/<app_name>/Graphs/<graph_type>/<graph_name>', methods=['GET'])
def get_graph(app_name, graph_type, graph_name):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    if graph_type == "DataGraph":
        relationships = "IS_IN_DATAGRAPH"
    elif graph_type == "Transaction":
        relationships = "IS_IN_TRANSACTION"
    else:
        print("generate_cypher_query is build for DataGraph or Transaction")
        return Response("generate_cypher_query is build for DataGraph or Transaction", status=400, mimetype='application/json')

    cypher_query = (
        f"CALL cast.linkTypes([\"CALL_IN_TRAN\"]) yield linkTypes\n"
        f"WITH linkTypes + [\"STARTS_WITH\", \"ENDS_WITH\"] AS updatedLinkTypes\n"
        f"MATCH (d:{graph_type}:{app_name})<-[:{relationships}]-(n)\n"
        f"WITH collect(id(n)) AS nodeIds,updatedLinkTypes\n"
        f"MATCH p=(d:{graph_type}:{app_name} {{Name: '{graph_name}'}})<-[:{relationships}]-(n:{app_name})<-[r]-(m:{app_name})-[:{relationships}]->(d)\n"
        f"WHERE (n:Object OR n:SubObject)\n"
        f"AND (m:Object OR m:SubObject)\n"
        f"AND id(n) IN nodeIds AND id(m) IN nodeIds\n"
        f"AND type(r) IN updatedLinkTypes\n"
        "RETURN DISTINCT n, r, m"
    )
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


def main():
    log_format = ' %(asctime)-15s \t%(levelname)s\tMODULMSG ; Body\t%(message)s'
    # logging.basicConfig(filename='sam-graf-server.log', encoding='utf-8', level=logging.DEBUG, format=log_format)
    app.logger.info("Sam Graf Server " + __version__)
    app.run(debug=True)


if __name__ == '__main__':
    main()
