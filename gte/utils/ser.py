import pickle
import os
# from functools import wraps

def serialize(obj, obj_name, base_dir='./', dir_structure='', pre_hash='', post_hash='', protocol=pickle.HIGHEST_PROTOCOL):
    path = os.path.join(base_dir, pre_hash, dir_structure, post_hash)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, obj_name + '.pickle')
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)

def deserialize(obj_name, base_dir='./', dir_structure='', pre_hash='', post_hash='', protocol=pickle.HIGHEST_PROTOCOL):
    # file_name = os.path.join(self.path, name + '.pickle')
    path = os.path.join(base_dir, pre_hash, dir_structure, post_hash)
    file_name = os.path.join(path, obj_name + '.pickle')
    obj = None
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj

def exists_serialized_obj(obj_name, base_dir='./', dir_structure='', pre_hash='', post_hash=''):
    path = os.path.join(base_dir, pre_hash, dir_structure, post_hash, obj_name + '.pickle')
    return os.path.exists(path)
