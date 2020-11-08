#!/usr/bin/env python3
""" Module used to
"""


def top_students(mongo_collection):
    """ function that returns all students sorted by average score
        The top must be ordered
        The average score must be part of each item returns with
            key = averageScore
    """

    student_lst = []

    students_collection = mongo_collection.find()
    for student in students_collection:

        total_grades = 0
        for i, topic in enumerate(student["topics"]):
            total_grades = total_grades + topic["score"]
        student["averageScore"] = total_grades / (i + 1)
        student_lst.append(student)

        # sort list
    r = sorted(student_lst, key=lambda std: std["averageScore"], reverse=True)

    return r
