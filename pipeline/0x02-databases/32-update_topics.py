#!/usr/bin/env python3
""" Module used to
"""


def update_topics(mongo_collection, name, topics):
    """[changes all topics of a school document based on the name]

    Args:
        mongo_collection ([db]): [mongo collection object]
        name ([string]): [ school name to update]
        topics ([list of strings]): [list of topics approached in the school]

    Returns:
        Nothing
    """

    # create the data dictionary
    new_topics = {"$set": {"topics": topics}}

    # update the mongo_collection
    mongo_collection.update_many({"name": name}, new_topics)
