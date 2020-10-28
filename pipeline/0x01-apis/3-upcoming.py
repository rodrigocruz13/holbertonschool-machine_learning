#!/usr/bin/env python3
"""
By using the (unofficial) SpaceX API write a script that displays the upcoming
launch with these information:

Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad
Format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)

The “upcoming launch” is the one which is the soonest from now, in UTC (we
encourage you to use the date_unix for sorting it) - and if 2 launches have
the same date, use the first one in the API result.

Your code should not be executed when the file is imported (you should use
if __name__ == '__main__':)

"""

import requests


if __name__ == '__main__':

    base_url = "https://api.spacexdata.com/v4"
    upcoming_launch_url = base_url + "/launches/upcoming"
    response = requests.get(upcoming_launch_url)

    r_code = response.status_code

    if(r_code == 200):
        next_date = response.json()[0]['date_unix']
        launches_dict = response.json()

        # find the date, name, local date and rocket of the next launch date
        for launch in launches_dict:
            launch_date = launch["date_unix"]

            if(launch_date <= next_date):
                next_date = launch_date
                launch_name = launch["name"]  # 1
                launch_date_local = launch["date_local"]  # 2
                launch_rocket_number = str(launch["rocket"])
                launchpad_number = str(launch["launchpad"])

                rockets_url = base_url + "/rockets/" + launch_rocket_number
                rocket_name = requests.get(rockets_url).json()["name"]  # 3

                launchpad_url = base_url + "/launchpads/" + launchpad_number
                launchpad_dict = requests.get(launchpad_url).json()
                launchpad_name = launchpad_dict["name"]  # 4
                launchpad_locality = launchpad_dict["locality"]  # 5

print("{} ({}) {} - {} ({})".format(launch_name,
                                    launch_date_local,
                                    rocket_name,
                                    launchpad_name,
                                    launchpad_locality))
