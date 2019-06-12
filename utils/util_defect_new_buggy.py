import sys
sys.path.append("/Volumes/ALI/postdoc_project/data_analysis/freud")
import numpy as np
import math
import freud
import pandas as pd
import random
import matplotlib.pyplot as plt

def compute_density(t,flist, N, r_cut):
    box = freud.box.Box.square(L=t[0].configuration.box[0])
    L = t[0].configuration.box[0]
    ref_points = get_grid(L,N)
    
    density = freud.density.LocalDensity(r_cut,0.5,0.5)

    dens = np.array([])
    for f in flist:
        snap = t[f]
        #d = density.compute(box, ref_points,snap.particles.position).density
        d = density.compute(box, ref_points,snap.particles.position).num_neighbors
        dens = np.concatenate((dens,d),axis=0)
        
        return dens;
    
    
def get_grid(L,N):
    dist = L/(N+1.0)
    x = np.linspace(-1.0*L/2.0+dist, L/2.0-dist, int(N))
    y = np.linspace(-1.0*L/2.0+dist, L/2.0-dist, int(N))
    z = np.zeros(1)
    
    X,Y,Z = np.meshgrid(x,y,z)
    p = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    p = np.ravel(p,order="F")
    p = p.reshape(int(N**2),3)
    
    return p

def compute_orientation(snap):
    N = snap.particles.N;
    orientation=np.zeros([N,3]);
    for tag in range(N):
        orient = np.array([1,0,0]);
        if(snap.particles.typeid[tag] != 2):
            orient = snap.particles.position[tag+1] - snap.particles.position[tag];
        else:
            orient = snap.particles.position[tag] -snap.particles.position[tag-1];
        orientation[tag] = orient / np.linalg.norm(orient);
    return orientation;


def find_position_angles(center_position, neigh_positions):
    alphas = []
    for pos in neigh_positions:
        r = pos-center_position[0]
        alpha = math.atan2(r[1],r[0])
        if (alpha < 0):
            alpha = alpha + 2*np.pi
        alphas.append(alpha)
    return alphas

def compute_theta(oct_orient):
    thetas = []
    theta = 0;
    for i in range(8):
        j = i+1
        or_curr = oct_orient[i]
        if(i==7):
            j=0;
        or_next = oct_orient[j]
        dtheta = math.atan2(or_curr[0]*or_next[1]-or_curr[1]*or_next[0], or_curr[0]*or_next[0]+or_curr[1]*or_next[1])
        if (dtheta>np.pi/2.):
            dtheta = dtheta - np.pi
        elif(dtheta<-np.pi/2.):
            dtheta = dtheta + np.pi
        theta+=dtheta

        thetas.append(theta)
    thetas.insert(0,0)
    thetas = np.array(thetas)
    thetas = thetas + math.atan(oct_orient[0,1]/oct_orient[0,0])

    return thetas

def detect_defect_single(center_position, neigh_positions,neigh_orientations, slice_no):
   
    alphas = find_position_angles(center_position,neigh_positions)
    octant_angles = np.linspace(0,2*np.pi,9,endpoint=True)
    alphas_octant_indices = np.digitize(alphas,octant_angles)
        
    octant_orientation = []
    for i in range(8):
        ind = i+1
        orient_oct = neigh_orientations[np.where(alphas_octant_indices==ind)]
        octant_orientation.append(np.array(np.mean(orient_oct, axis=0))) 
    octant_orientation = np.array(octant_orientation)
            
    return compute_theta(octant_orientation)    

def detect_defect_pipeline(positions,orientations,index_i, index_j, random_points,slice_no):
    charges = []
    for i,center_tag in enumerate(random_points):
        print("point {} of {}".format(i, len(random_points)))
        neigh_tags = index_j[np.where(index_i==center_tag)]
        neigh_positions = positions[neigh_tags]
        neigh_orientations = orientations[neigh_tags]
        center_position = positions[neigh_tags]

        result = detect_defect_single(center_position, neigh_positions, neigh_orientations, slice_no)
        defect_rot = result[-1]-result[0]
        if not (math.isnan(defect_rot)):
            out_tuple = (int(center_tag),round(defect_rot/3.1415926535)/2)
            charges.append(out_tuple)
    return charges
        