from anchor_pro.elements.wall_backing import WallBackingProps, WallBackingElement
from anchor_pro.elements.elastic_bolt_group import ElasticBoltGroupProps

# from anchor_pro.equipment import WallBackingElement as OldElement
from anchor_pro.elements.wall_backing import WallBackingElement, WallBackingProps
from anchor_pro.elements.elastic_bolt_group import ElasticBoltGroupProps, ElasticBoltGroupResults, calculate_bolt_group_forces
import numpy as np

w = 6
h = 6
sx = 4
sy = 4
xy_anchors = np.array([[-sx/2,-sy/2],
                       [sx/2,-sy/2],
                       [sx/2,sy/2],
                       [-sx/2,sy/2]])

# Create old element
bg = ElasticBoltGroupProps(w, h, xy_anchors,4,np.mean(xy_anchors,axis=0))
N = np.array([10])
Vx = np.array([5])
Vy = np.array([5])
Mx = np.array([20])
My = np.array([0])
T = np.array([0])
res = calculate_bolt_group_forces(N,Vx,Vy,Mx,My,T,bg.n_anchors,bg.inert_props_cent)