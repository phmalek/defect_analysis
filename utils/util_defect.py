from scipy import interpolate
from scipy.optimize import bisect
import numpy as np
import sys
sys.path.append("/Volumes/ALI/Postdoc_Project/data_analysis/freud")
sys.path.append("../")
import freud
import os
import gsd.hoomd
import math
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import util



def find_point_in_cell(lc,point_no):
    rnd_tag_in_cells=[]
    for cell in range(lc.num_cells):
        tags  = [int(x) for x in lc.itercell(cell)]
        if (len(tags)==0):
            rnd_tag_in_cells.append([0]*point_no)
        else:
            center_tag = np.random.choice([int(x) for x in lc.itercell(cell)],point_no)
            rnd_tag_in_cells.append(center_tag)
    return np.array(rnd_tag_in_cells)


#def find_position_angles(snap,tags,center_tag):
#    alphas = []

#    for tag in tags:
#        r = snap.particles.position[tag]-snap.particles.position[center_tag]
#        alpha = math.atan2(r[1],r[0])
#        if (alpha < 0):
#            alpha = alpha + 2*np.pi
#        alphas.append(alpha)
#    return alphas

def find_position_angles(position,tags,center_tag):
    alphas = []

    for tag in tags:
        r = position[tag]-position[center_tag]
        alpha = math.atan2(r[1],r[0])
        if (alpha < 0):
            alpha = alpha + 2*np.pi
        alphas.append(alpha)
    return alphas

def find_alphas_thetas(oct_orient):
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
    
    alphas_octant = np.linspace(np.pi/8,2*np.pi+np.pi/8,9)
    
    
    return np.column_stack((np.array(alphas_octant),np.array(thetas)))


def find_defect_dir(alphas_thetas):
        
    alphas_thetas_interp = interpolate.interp1d(alphas_thetas[:-1,0],alphas_thetas[:-1,1])
    
    nl = math.floor((alphas_thetas[0,1]-alphas_thetas[0,0])/np.pi)
    nr = int((alphas_thetas[-2,1]-alphas_thetas[-2,0])/np.pi)#the reason for -2 is that thetas[-1] is just thetas[0]+2pi


    roots_x = []
    for n in range(nr,nl+1):
        f = lambda x:x+n*np.pi
        difs = lambda x:f(x)-alphas_thetas_interp(x)
        root = bisect(difs,alphas_thetas[0,0],alphas_thetas[-2,0])
        roots_x.append(root)

    roots_x=np.array(roots_x)
    roots_y=alphas_thetas_interp(roots_x)
    roots_xy = np.column_stack((roots_x,roots_y))
    return roots_xy


def modify_orientations(orientations_x):
    orientations = orientations_x.copy()
    for i,o in enumerate(orientations):
        if o[0]<0:
            orientations[i]=-1*o
    return orientations
            
            
#def find_point_in_cell(lc):
#    rnd_tag_in_cells=[]
#    for cell in range(lc.num_cells):
#        center_tag = np.random.choice([int(x) for x in lc.itercell(cell)],5)
#        rnd_tag_in_cells.append(center_tag)
#    return np.array(rnd_tag_in_cells)


def detect_defect(snap,orientations,nlist,center_tag,r):
    
    #orientations = util.compute_orientation(snap)
    #modify_orientations(orientations)
    
    #nlist = lc.nlist.filter_r(box,snap.particles.position[:],snap.particles.position[:],rmax=r)
    neigh_tags = nlist.index_j[np.where(nlist.index_i==center_tag)]
    print("neigh_tags: Done")
    
    alphas = find_position_angles(snap.particles.position,neigh_tags,center_tag)
    print("alphas: Done")

    octant_angles = np.linspace(0,2*np.pi,9,endpoint=True)
    print("octant_angles: Done")

    
    alphas_octant_indices = np.digitize(alphas,octant_angles)
    print("alphas_octant_indices: Done")
    
    
    orient_neigh = orientations[neigh_tags,:]
    pos_neigh = snap.particles.position[neigh_tags,:]
    octant_orientation = []
    for i in range(8):
        ind = i+1
        orient_oct = orient_neigh[np.where(alphas_octant_indices==ind)]
        octant_orientation.append(np.array(np.mean(orient_oct, axis=0))) 
    octant_orientation = np.array(octant_orientation)
    
    alphas_thetas = find_alphas_thetas(octant_orientation)
    print("alphas_thetas: Done")
    
    
    defect_charge = round((alphas_thetas[-1,1]-alphas_thetas[0,1])/np.pi)/2
    
    #defect_direction = find_defect_dir(alphas_thetas)
    
    return defect_charge, alphas_thetas
    #return defect_charge