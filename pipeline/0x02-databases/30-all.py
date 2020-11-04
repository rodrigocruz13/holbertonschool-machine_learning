#!/usr/bin/env python3
""" Module used to
"""

def list_all(mongo_collection):
    """[Lists all documents in a collection]

    Args:
        mongo_collection ([pymongo]): [collection object]

    Returns:
        docs [list]: [Elements in the mongo collection]
    """

    mongodb = mongo_collection.find()
    return [doc for doc in mongodb]