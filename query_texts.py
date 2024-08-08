def __graphs_query(app_name: str, graph_type: str, graph_id: int) -> str:
    _relationship_type = "IS_IN_" + graph_type.upper()
    return (f"""
        MATCH p = (d:{graph_type}:{app_name})<-[i1:{_relationship_type}]-(n:{app_name})
        WHERE id(d) = {graph_id}
        AND (n:Object OR n:SubObject)
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
        RETURN n1, m1, r1
        """
            )


def __appgraph_query(app_name: str) -> str:
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
        RETURN n1, m1, r1 limit 10
        """
            )
