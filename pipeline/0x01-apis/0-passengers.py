#!/usr/bin/python3
"""script for getting info from web pages
"""

import requests
# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def to_int(a_str):
    """[Converts strings into numbers]

        Args:
            a_string ([str]): [a string number]
    """

    if not isinstance(a_str, str) or (a_str == 'n/a') or (a_str == 'unknown'):
        return 0

    elif (len(a_str.split(',')) > 1):
        return int(a_str.split(',')[0]) * 1000 + int(a_str.split(',')[1])

    else:
        return int(a_str)


def availableShips(passengerCount):
    """[Returns the ships that can hold a given number of passengers]

    Args:
        passengerCount ([int]): [number of passangers]

    Returns
        The list of ships
    """

    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships'
    response = requests.get(url)  # No authentication is required
    r_code = response.status_code
    next_url = url
    while(r_code == 200):

        json_data = response.json()['results']

        for ship in json_data:
            count = to_int(ship['passengers'])
            if (count >= passengerCount):
                ships.append(ship['name'])

        next_url = response.json()['next']
        if(next_url is None):
            break

        response = requests.get(next_url)

    return (ships)
