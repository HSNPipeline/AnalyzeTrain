import os
from os import path as ospath
import numpy as np
import pandas as pd

# get all session paths
def get_all_session_paths(base_path):
    session_paths = []
    items = os.listdir(base_path)
    subj_files = [item for item in items if item != '.DS_Store']
    for i,subj in enumerate (subj_files):
        subj_path = os.path.join(base_path, subj)
        session_paths.append(subj_path)
    return session_paths