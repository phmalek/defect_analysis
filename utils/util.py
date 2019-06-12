import sys
sys.path.append("/Volumes/ALI/postdoc_project/data_analysis/freud")
import numpy as np
import math
import freud
import pandas as pd
import random
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../defect_xtensor/")
from xdefect import detection_pipeline as dpc

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

def modify_orientations(orientations_x):
    orientations = orientations_x.copy()
    for i,o in enumerate(orientations):
        if o[0]<0:
            orientations[i]=-1*o
    return orientations


def find_point_in_cell(lc,point_no):
    rnd_tag_in_cells=[]
    for cell in range(lc.num_cells):
        tags  = np.array([int(x) for x in lc.itercell(cell)])
        if (len(tags)!=0):
            #p_in_cell = [int(x) for x in lc.itercell(cell)]
            p = min(len(tags), point_no)
            #print(p)
            center_tag = np.random.choice(tags,p, replace=False)
            rnd_tag_in_cells.extend(center_tag)
    return np.array(rnd_tag_in_cells)


#def find_point_in_cell_det(lc, point_no):
#    rnd_tag_in_cells = []
#    for cell in range(lc.num_cells):
#        tags  = [int(x) for x in lc.itercell(cell)]
#        if (len(tags)!=0):
#            center_tag = np.arange(lc.itercell())
#            center_tag = np.random.choice([int(x) for x in lc.itercell(cell)],point_no)
#            rnd_tag_in_cells.extend(center_tag)
#    return np.array(rnd_tag_in_cells)


def filter_big_clusters(cl,min_number):
    big_clusters = []
    big_clusters_tags = []
    for c in range(cl.num_clusters):
        cl_points = np.where(cl.cluster_idx==c)
        if(len(cl_points[0])>=min_number):
            big_clusters.append(c)
            big_clusters_tags.append(cl_points)
    return big_clusters, big_clusters_tags



class defect:
    def __init__(self, snap, cell_size, r_cl,slice_no, min_angle):
        self.box = freud.box.Box.from_box(snap.configuration.box, dimensions=2)
        self.lc = freud.locality.LinkCell(self.box,cell_size)
        self.cl = freud.cluster.Cluster(self.box,r_cl)
        self.clp = freud.cluster.ClusterProperties(box=self.box)
        self.min_angle = min_angle
        self.slice_no = slice_no

        
    def snap_find_tags(self,snap, r_nlist, p_per_cell, count=0):
        orientations = compute_orientation(snap)
        orientations = modify_orientations(orientations)
        
        self.lc.compute(self.box, snap.particles.position)

        nlist = self.lc.nlist
        nlist.filter_r(self.box, snap.particles.position[:], snap.particles.position[:],rmax = r_nlist)
        print("AND?")
        random_points = find_point_in_cell(self.lc, int(p_per_cell))
        print("before:")
        print(len(random_points))

        if count!=0:
            dens_tags = nlist.neighbor_counts[random_points]>count
            random_points = random_points[dens_tags]    
            print("after")
            print(len(random_points))
       
        defect_list = np.array(dpc(snap.particles.position[:,[0,1]], orientations[:,[0,1]], nlist.index_i, nlist.index_j, nlist.segments, random_points,self.slice_no, self.min_angle))
        
        tags_p = random_points[defect_list==0.5]
        tags_m = random_points[defect_list==-0.5]
        

        return tags_p, tags_m
    
    def defect_position(self, snap,tags, min_cl):
        self.cl.computeClusters(snap.particles.position[tags])
        self.clp.computeProperties(snap.particles.position[tags], self.cl.cluster_idx)
                
        bc, bc_tags = filter_big_clusters(self.cl,min_cl)
        
        coord_com = self.clp.cluster_COM[bc]
        return coord_com
    
    def dump_nlist(self,snap,r_nlist, p_per_cell,i):
        orientations = compute_orientation(snap)
        orientations = modify_orientations(orientations)
        self.lc.compute(self.box, snap.particles.position)
        nlist = self.lc.nlist
        nlist.filter_r(self.box, snap.particles.position[:], snap.particles.position[:],rmax = r_nlist)
        random_points = find_point_in_cell(self.lc, p_per_cell)

        np.save("test_data/orientations_{}.npy".format(i), orientations[:,[0,1]].astype(np.float32))
        np.save("test_data/positions_{}.npy".format(i), snap.particles.position[:,[0,1]].astype(np.float32))
        np.save("test_data/index_i_{}.npy".format(i), nlist.index_i.astype(np.int32))
        np.save("test_data/index_j_{}.npy".format(i), nlist.index_j.astype(np.int32))
        np.save("test_data/segments_{}.npy".format(i), nlist.segments.astype(np.int32))
        np.save("test_data/random_points_{}.npy".format(i), random_points.astype(np.int32))

        