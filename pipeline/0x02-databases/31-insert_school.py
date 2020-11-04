#!/usr/bin/env python3
""" Module used to
"""


def insert_school(mongo_collection, **kwargs):
    """[function that inserts a new document in a collection based on kwargs]

    Args:
        mongo_collection ([type]): [collection object]

    Returns:
        [id]: [_id]  New id
    """

    # insert new element and return its id
    return mongo_collection.insert_one(kwargs).inserted_id
