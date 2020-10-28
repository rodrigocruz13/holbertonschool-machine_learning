#!/usr/bin/env python3
"""
By using the (unofficial) SpaceX API write a script that displays the number
of launches per rocket.

- All launches should be taking in consideration
- Each line should contain the rocket name and the number of launches
  separated by : (format below in the example)
- Order the result by the number launches (descending)
- If multiple rockets have the same amount of launches, order them by
 alphabetic order (A to Z)
- Your code should not be executed when the file is imported (you should use
  if __name__ == '__main__':)
"""

import requests

if __name__ == '__main__':

    base_url = "https://api.spacexdata.com/v4"
    launches_url = base_url + "/launches"
    response = requests.get(launches_url)

    code = response.status_code
    frecuency = {}

    if(code == 200):
        launches = response.json()

        # Find each launch
        for launch in launches:
            rocket_num = launch["rocket"]

            # Find the rocket for that launch
            rocket_url = base_url + "/rockets/" + str(rocket_num)
            r_name = requests.get(rocket_url).json()["name"]
            keys = frecuency.keys()

            # Count
            frecuency[r_name] = frecuency[r_name] + 1 if r_name in keys else 1

        # Sorting
        frecuency = sorted(frecuency.items(), key=lambda x: x[0])
        frecuency = sorted(frecuency, key=lambda x: x[1], reverse=True)

        for rocket in frecuency:
            print("{}: {}".format(rocket[0], rocket[1]))
