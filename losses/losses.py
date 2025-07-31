"""
losses/domain.py (formerly losses.py):
General-purpose domain, collocation, and boundary condition classes for PINN frameworks.
Contains ASTPN and general boundary condition utilities (e.g., periodic_BC).
"""

from dependency_codes.data_generator import col_gen
from .base import diff, mse_loss
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from dependency_codes.config import cs, const, G, rho_o

class ASTPN(col_gen):

    def __init__(self, rmin=[0,0,0,0], rmax=[1,1,1,1], N_0 = 1000, N_b=1000, N_r=3000, dimension=1):
        super().__init__(rmin,rmax, N_0,N_b,N_r, dimension)

        self.coord_Lx, self.coord_Rx = self.geo_time_coord(option="BC",coordinate=1)

        if dimension == 2:
            self.coord_Ly, self.coord_Ry = self.geo_time_coord(option="BC",coordinate=2)

        if dimension == 3:
            self.coord_Ly, self.coord_Ry = self.geo_time_coord(option="BC",coordinate=2)
            self.coord_Lz, self.coord_Rz = self.geo_time_coord(option="BC",coordinate=3)

    def periodic_BC(self, net, alpha, alpha_size, coordinate=1, derivative_order=0, component=0):

        if coordinate==1:
            coord_L, coord_R = self.coord_Lx, self.coord_Rx

        if coordinate==2:
            coord_L, coord_R = self.coord_Ly, self.coord_Ry

        if coordinate==3:
            coord_L, coord_R = self.coord_Lz, self.coord_Rz

        coord_L_mod = [tensor.repeat_interleave(alpha_size, dim=0) for tensor in coord_L]
        coord_R_mod = [tensor.repeat_interleave(alpha_size, dim=0) for tensor in coord_R]
        alpha_repeated = alpha.repeat_interleave(coord_L[0].size(0), dim=0)

        coord_L_in = [t.clone() for t in coord_L_mod]
        coord_R_in = [t.clone() for t in coord_R_mod]
        coord_L_in.append(alpha_repeated)
        coord_R_in.append(alpha_repeated)
        variable_l = net(coord_L_in)[:,component:component+1]
        variable_r = net(coord_R_in)[:,component:component+1]

        if derivative_order == 0:
            return torch.mean((variable_l - variable_r)**2)
        
        elif derivative_order == 1:        
            der_l = diff(variable_l,coord_L_in[coordinate-1],order=derivative_order)
            der_r = diff(variable_r,coord_R_in[coordinate-1],order=derivative_order)
            
            return torch.mean((der_l-der_r)**2)
