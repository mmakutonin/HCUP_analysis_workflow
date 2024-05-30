import os
import pickle
from datetime import datetime

# loads a pickled file using a relative path from the ../data/ directory.
def load_file(data_path):
    with open('../pickled_data/' + data_path, "rb") as input_file:
        return pickle.load(input_file)

# saves a Python object to a pickled file using a relative path from the ../data/ directory.
def pickle_file(data_path, python_obj):
    with open('../pickled_data/' + data_path, 'wb') as pickle_file:
        pickle.dump(python_obj, pickle_file, protocol=pickle.DEFAULT_PROTOCOL)

# prints operation start dialogue
def starting_run(operation_name=''):
    print('Starting ' + operation_name + ' ' + str(datetime.now().time()))

# prints operation completion dialogue
def finished_run(operation_name=''):
    print('Finished ' + operation_name + ' ' + str(datetime.now().time()))