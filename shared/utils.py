from pathlib import Path
from uuid import UUID


def is_valid_uuid(uuid_to_test, version=4):
    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False

    return str(uuid_obj) == uuid_to_test

def get_throw_id(file):
    throw_id = Path(file).stem

    if not is_valid_uuid(throw_id):
        raise Exception('Throw Id is not valid')

    return throw_id