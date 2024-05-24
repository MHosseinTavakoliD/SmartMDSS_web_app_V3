from pymongo import MongoClient


def add_state_field():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['SmartMDSS_data']
    collection = db['User_input_point']

    # Assuming the existence of a field or pattern that can definitively set the state to Michigan
    collection.update_many(
        {'road_name': {'$regex': '^Michigan'}},
        {'$set': {'state': 'Michigan'}}
    )

    print("State field added to documents where applicable.")

add_state_field()
