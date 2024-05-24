from django.conf import settings

def get_database_engine():
    return settings.DATABASES['default']['ENGINE']

# Example usage
engine_name = get_database_engine()
print("The database engine is:", engine_name)