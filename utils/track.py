import pandas as pd

def snap_to_trackpy(defect_pos, frame):
    x = defect_pos[:,0]
    y = defect_pos[:,1]
    
    return pd.DataFrame(dict(x = x, y = y, frame = frame))

def prep_traj_trackpy(traj):
    frames = []
    for i,snap in enumerate(traj):
        frames.append(snap_to_trackpy(snap.position, i))
        
    return frames