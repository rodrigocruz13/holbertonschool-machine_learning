#!/usr/bin/env python3
"""script for getting info from web pages
"""

import requests
import sys
import time

if __name__ == '__main__':

    url = sys.argv[1]
    params = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, params=params)

    if (response.status_code == 200):
        location = response.json()["location"]
        print(location)

    if (response.status_code == 403):
        limit = response.headers["X-Ratelimit-Reset"]
        time_out = int(limit) - int(time.time())
        time_out = time_out / 60
        print("Reset in {} min".format(int(time_out)))

    if (response.status_code == 404):
        print("Not found")
