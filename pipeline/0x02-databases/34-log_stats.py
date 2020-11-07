#!/usr/bin/env python3
""" Module used to
"""

from pymongo import MongoClient

method_list = ["GET", "POST", "PUT", "PATCH", "DELETE"]
db_address = 'mongodb://127.0.0.1:27017'

client = MongoClient(db_address)

# create dict of all dbs and its collections
d = dict((db, [collection for collection in client[db].collection_names()])
         for db in client.database_names())

# validate logs and nginx exists
if ('logs' not in d.keys() or 'nginx' != d['logs'][0]):
    exit()

collection = client.logs.nginx  # client.database.collection
docs = collection.count_documents({})
print(docs, "logs")

print("Methods:")
for method in method_list:
    count = collection.count_documents({"method": method})
    print("\tmethod {}: {}".format(method, count))

GET_status = collection.count_documents({"method": "GET", "path": "/status"})
print(GET_status, "status check")
