from neo_query import NeoQuery

from flask import Flask, request

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "imaging")
DATABASE = "neo4j"

app = Flask(__name__)


@app.route('/<app_name>/Objects', methods=['GET', 'POST'])
def get_objects(app_name):
    my_query = NeoQuery(URI, AUTH, DATABASE)

    query = """
        MATCH (o1:Object:""" + app_name + """)-[r]->(o2:Object:""" + app_name + """)
        RETURN o1, o2, r """
    if request.method == 'POST':
        limit = int(request.args.get("limit", 100))
        query = query + " LIMIT $limit"
        return my_query.execute_query(query, limit)
    else:
        query = query + " LIMIT 250"
        return my_query.execute_query(query)


@app.route('/<app_name>/Levels/<level_number>', methods=['GET'])
def get_levels(app_name, level_number):
    my_query = NeoQuery(URI, AUTH, DATABASE)
    level_label = "Level" + level_number
    query = """
        MATCH p=(l1:""" + level_label + """:""" + app_name + """)-[r]->(l2:""" + level_label + """:""" + app_name + """)
        RETURN p LIMIT $limit"""
    limit = int(request.args.get("limit", 100))
    return my_query.execute_query(query, limit)


if __name__ == '__main__':
    app.run(debug=True)
