import matplotlib.pyplot as plt
import pandas as pd
import owncloud
from io import BytesIO



subject_id = ['215', '216', '217', '218']

file_names = []
files = oc.list('/Smartphone3/', depth = 'infinity')
for file in files:
    if not file.is_dir():
        if any(x in file.get_path() for x in subject_id) :
            file_names.append (file.get_path() + '/' + file.get_name())

print(files)