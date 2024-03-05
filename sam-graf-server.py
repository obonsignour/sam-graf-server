import logging
from webbrowser import get
from neo_query import NeoQuery

from flask import Flask, request

URI = "neo4j://localhost:7687"
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


@app.route('/Applications/<app_name>/Levels/<level_number>', methods=['GET'])
def get_levels(app_name, level_number):
    logging.info("Getting levels for " + app_name + " at level " + level_number)
    my_query = NeoQuery(URI, AUTH, DATABASE)
    level_label = "Level" + level_number
    query = """
        MATCH p=(l1:""" + level_label + """:""" + app_name + """)-[r]->(l2:""" + level_label + """:""" + app_name + """)
        RETURN p LIMIT $limit"""
    limit = int(request.args.get("limit", 100))
    return my_query.execute_query(query, limit)


def main():
    log_format = ' %(asctime)-15s \t%(levelname)s\tMODULMSG ; Body\t%(message)s'
    logging.basicConfig(filename='sam-graf-server.log', encoding='utf-8', level=logging.DEBUG, format=log_format)
    logging.info("Sam Graf Server " + __version__)
    app.run(debug=True)


if __name__ == '__main__':
    main()
