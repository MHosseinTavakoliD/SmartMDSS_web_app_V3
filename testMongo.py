import pymongo

def get_mongo_client():
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    return client
