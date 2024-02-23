from typing import Any


class OgmaNode:
    def __init__(self, id: int, data: Any):
        self.__id = id
        self.__data = data

    def to_dict(self):
        return {
            'id': self.__id,
            'data': self.__data
        }


class OgmaEdge:
    def __init__(self, id: int, source: int, target: int, data: Any):
        self.__id = id
        self.__source = source
        self.__target = target
        self.__data = data

    def to_dict(self):
        return {
            'id': self.__id,
            'source': self.__source,
            'target': self.__target,
            'data': self.__data
        }
