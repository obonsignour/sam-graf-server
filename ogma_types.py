from turtle import st
from typing import Any, List


class OgmaNode():
    def __init__(self, id: int, labels: List[str], properties: Any):
        self.__id: int = id
        self.__data = {'labels': labels, 'properties': properties}

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.__id,
            'data': self.__data,
        }


class OgmaEdge():
    def __init__(self, id: int, source: int, target: int, data: dict[str, Any]):
        self.__id = id
        self.__source = source
        self.__target = target
        self.__data = data

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.__id,
            'source': self.__source,
            'target': self.__target,
            'data': self.__data
        }


class OgmaGraph:
    def __init__(self, nodes: List[OgmaNode], edges: List[OgmaEdge]):
        self.__nodes: List[OgmaNode] = nodes
        self.__edges: List[OgmaEdge] = edges

    def to_dict(self) -> dict[str, Any]:
        nodes = [node.to_dict() for node in self.__nodes]
        edges = [edge.to_dict() for edge in self.__edges]
        return {
            'nodes': nodes,
            'edges': edges
        }
