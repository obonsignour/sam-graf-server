import cgi


def get_callgraph_query(app_name: str, graph_type: str) -> str:
    return f"""
        MATCH (cg:{graph_type}:{app_name}) 
        CALL {{
            WITH cg
            MATCH (cg)<-[:RELATES_TO]-(m:{app_name}:Model)
            RETURN collect(m.name) AS modelNames    
        }}
        RETURN id(cg) AS id, cg.Name AS graphName, modelNames ORDER BY graphName"""


def callgraph_query(app_name: str, graph_type: str, graph_id: int) -> str:
    _relationship_type = "IS_IN_" + graph_type.upper()
    return (f"""
        MATCH p = (d:{graph_type}:{app_name})<-[i1:{_relationship_type}]-(n:{app_name})
        WHERE id(d) = {graph_id}
          AND (n:Object OR n:SubObject)
        WITH collect(n.AipId) AS aipIds
        CALL cast.linkTypes([\"CALL_IN_TRAN\"]) yield linkTypes
        WITH aipIds, linkTypes
        MATCH p = (n:{app_name})<-[r]-(m:{app_name})
        WHERE (n:Object OR n:SubObject)
          AND (m:Object OR m:SubObject)
          AND n.AipId IN aipIds
          AND m.AipId IN aipIds
          AND type(r) IN linkTypes
        RETURN p
        """
            )


# def callgraph_query(app_name: str, graph_type: str, graph_id: int) -> str:
#     _relationship_type = "IS_IN_" + graph_type.upper()
#     return (f"""
#         MATCH p = (d:{graph_type}:{app_name})<-[i1:{_relationship_type}]-(n:{app_name})
#         WHERE id(d) = {graph_id}
#         AND (n:Object OR n:SubObject)
#         WITH n, apoc.map.setLists(properties(n), ['UndirectedLouvain', 'DirectedLouvain', 'Leiden', 'SLPA'], [i1.UndirectedLouvain, i1.DirectedLouvain, i1.Leiden, i1.SLPA]) AS propsCompleted
#         WITH apoc.create.vNode(labels(n), propsCompleted) AS n1
#         WITH collect(n1) AS vNodes, collect(apoc.any.property(n1, "AipId")) AS aipIds
#         CALL cast.linkTypes([\"CALL_IN_TRAN\"]) yield linkTypes
#         WITH vNodes, aipIds, linkTypes
#         MATCH (n:{app_name})<-[r]-(m:{app_name})
#         WHERE (n:Object OR n:SubObject)
#         AND (m:Object OR m:SubObject)
#         AND n.AipId IN aipIds
#         AND m.AipId IN aipIds
#         AND type(r) IN linkTypes
#         WITH n, m, r, vNodes
#         WITH [n1 IN vNodes WHERE apoc.any.property(n1, "AipId")=n.AipId | n1][0] AS n1,  [m1 IN vNodes WHERE apoc.any.property(m1, "AipId")=m.AipId | m1][0] AS m1, r
#         WITH n1, m1, apoc.create.vRelationship(n1, type(r), properties(r), m1) AS r1
#         RETURN n1, m1, r1
#         """
#             )


def modelgraph_query(app_name: str, graph_id: int, modelgraph_name: str) -> str:
    return (f"""
        MATCH p = (t:{app_name})<-[:RELATES_TO]-(d:Model:{app_name})<-[i1:IS_IN_MODEL]-(n:{app_name})
        WHERE d.name = "{modelgraph_name}"
        AND ID(t) = {graph_id}
        AND (n:Object OR n:SubObject)
        WITH n, apoc.map.setLists(properties(n), ['Community'], [i1.Community]) AS propsCompleted
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


def get_appgraph_query(app_name: str) -> str:
    return f"""
        CALL cast.linkTypes([\"CALL_IN_TRAN\"]) yield linkTypes
        MATCH p = (n:{app_name}&(Object|SubObject))-[r]->(m:{app_name}&(Object|SubObject))
        RETURN p LIMIT 500
"""


def appgraph_query(app_name: str) -> str:
    return (f"""
        MATCH p = (d:Model:{app_name})<-[i1:IS_IN_MODEL]-(n:{app_name})
        WHERE (n:Object OR n:SubObject)
        WITH n, apoc.map.setLists(properties(n), ['UndirectedLouvain', 'DirectedLouvain', 'Leiden', 'SLPA'], [i1.UndirectedLouvain, i1.DirectedLouvain, i1.Leiden, i1.SLPA]) AS propsCompleted
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
        RETURN n1, m1, r1 limit 1000
        """
            )


def generate_cypher_query(application, graph_type, graph_id, linkTypes=["all"]):
    if graph_type == "DataGraph":
        relationship_type = "IS_IN_DATAGRAPH"
    elif graph_type == "Transaction":
        relationship_type = "IS_IN_TRANSACTION"
    else:
        return print("generate_cypher_query is build for DataGraph or Transaction")
    if linkTypes == ["all"]:
        cypher_query = (f"""
            CALL cast.linkTypes(['CALL_IN_TRAN']) yield linkTypes
            WITH linkTypes + [] AS updatedLinkTypes
            MATCH (d:{graph_type}:{application})<-[:{relationship_type}]-(n)
            WITH collect(id(n)) AS nodeIds,updatedLinkTypes
            MATCH p=(d:{graph_type}:{application})<-[:{relationship_type}]-(n:{application})<-[r]-(m:{application})-[:{relationship_type}]->(d)
            WHERE ID(d) = {graph_id}
            AND (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND id(n) IN nodeIds AND id(m) IN nodeIds
            AND type(r) IN updatedLinkTypes
            RETURN DISTINCT n, r, m
            """
                        )
    else:
        cypher_query = (f"""
            WITH {linkTypes} as linkTypes
            WITH linkTypes + [] AS updatedLinkTypes
            MATCH (d:{graph_type}:{application})<-[:{relationship_type}]-(n)
            WITH collect(id(n)) AS nodeIds,updatedLinkTypes
            MATCH p=(d:{graph_type}:{application})<-[:{relationship_type}]-(n:{application})<-[r]-(m:{application})-[:{relationship_type}]->(d)
            WHERE ID(d) = {graph_id}
            AND (n:Object OR n:SubObject)
            AND (m:Object OR m:SubObject)
            AND id(n) IN nodeIds AND id(m) IN nodeIds
            AND type(r) IN updatedLinkTypes
            RETURN DISTINCT n, r, m
            """
                        )
    return cypher_query
