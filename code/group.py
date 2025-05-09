import os


# get all session paths
def get_all_session_paths(base_path):
    """
    Get all session paths from a base directory.
    
    Parameters
    ----------
    base_path : str
        Path to the base directory containing subject folders.
        
    Returns
    -------
    session_paths : list
        List of paths to individual subject sessions, excluding hidden files like .DS_Store.
    """
    session_paths = []
    items = os.listdir(base_path)
    subj_files = [item for item in items if item != '.DS_Store']
    for i,subj in enumerate (subj_files):
        subj_path = os.path.join(base_path, subj)
        session_paths.append(subj_path)
    return session_paths