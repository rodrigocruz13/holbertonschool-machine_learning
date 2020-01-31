#!/usr/bin/env python3
"""
Module used to add two arrays
"""


def canUnlockAll(boxes):
    """
    Method that determines if all the boxes can be opened.
    Args:
        boxes (lst): List of all the boxes.

    Returns:
        bool: The return value. True for success, False otherwise.
    """

    if (type(boxes) is not list):
        return False

    if (len(boxes) == 0):
        return False

    n_boxes = len(boxes)
    open_lst = ["Locked"] * n_boxes  # List to check all unlocked boxes
    open_lst[0] = "Unlocked"  # The first box boxes[0] is unlocked

    next = boxes[0]
    while (open_lst.count("Unlocked") < n_boxes and next is not None):
        open_me = next
        next = go_open(open_me, boxes, open_lst)

    if (open_lst.count("Unlocked") == n_boxes):
        return True
    return False


def go_open(open_me, boxes, open_lst):
    """
    Opens a single box (open_me), then check it in the open_ list as unlocked.
    Args:
        open_me (lst): List with the info of the current boxes to be opened.
        boxes (lst): List with all the boxes.
        open_lst (lst): List of Locked - Unlocked boxes.

    Returns:
        next_boxes (lst): In case of success, a list of next boxes to open.
        None: In case of failure.
    """

    if (open_me is None):  # There are no current boxes to be opened
        return None

    elif (len(open_me) == 0):  # Empty. There are no current boxes to open
        return None

    elif (len(open_me) == 1):  # 1 box to be open.
        i = open_me[0]
        if (len(boxes) <= i):  # crazy position
            return None
        if (open_lst[i] == "Unlocked"):  # Already been there
            return None
        else:
            open_lst[i] = "Unlocked"
            if(boxes[i] == open_me):  # Same position
                return None
        return boxes[i]

    else:  # More than 1 box to be opened
        next_boxes = [None] * len(open_me)
        i = 0
        for box_i in open_me:
            if (type(box_i) == int):
                n = [box_i]
            else:
                n = box_i
            next_boxes[i] = go_open(n, boxes, open_lst)
            i += 1

        if len(next_boxes) == 0:
            return None
        if next_boxes.count(None) == i:
            return None
        return next_boxes
