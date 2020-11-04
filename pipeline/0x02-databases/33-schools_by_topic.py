#!/usr/bin/env python3
""" Module used to
"""


def schools_by_topic(mongo_collection, topic):
    """[changes all topics of a school document based on the name]

    Args:
        mongo_collection ([db]): [mongo collection object]
        topics ([list of strings]): [list of topics approached in the school]

    Returns:
        list of school having a specific topic
    """

    # create the data dictionary

    all_ = mongo_collection.find({"topics": {"$all": [topic]}})
    return [school for school in all_]
