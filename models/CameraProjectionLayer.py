import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_tensor(theta, phi, psi, b=1, device="gpu"):
    """
    Takes theta, phi, and psi and generates the
    3x3 rotation matrix. Works for batched ops
    As well, returning a Bx3x3 matrix.
    """
    one = torch.ones(b, 1, 1).to(device)
    zero = torch.zeros(b, 1, 1).to(device)
    rot_x = torch.cat((
                torch.cat((one, zero, zero), 1),
                torch.cat((zero, theta.cos(), theta.sin()), 1),
                torch.cat((zero, -theta.sin(), theta.cos()), 1),
            ), 2)
    rot_y = torch.cat((
                torch.cat((phi.cos(), zero, -phi.sin()), 1),
                torch.cat((zero, one, zero), 1),
                torch.cat((phi.sin(), zero, phi.cos()), 1),
            ), 2)
    rot_z = torch.cat((
                torch.cat((psi.cos(), -psi.sin(), zero), 1),
                torch.cat((psi.sin(), psi.cos(), zero), 1),
                torch.cat((zero, zero, one), 1)
            ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))


class CameraProps(nn.Module):
    """
    Generates the extrinsic rotation and translation matrix
    For the current camera. Takes some feature as input, then
    Returns the rotation matrix (3x3) and translation (3x1)
    """
    def __init__(self, channels):
        super(CameraProps, self).__init__()
        self.cam = nn.Conv2d(channels, 128, 3)
        self.cam2 = nn.Linear(128, 32)
        self.rot = nn.Linear(32, 3)
        self.trans = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.cam(x))
        # averages x over space,time
        # then provides 3x3 rot and 3-dim trans
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = F.relu(self.cam2(x))
        b = x.size(0)
        r = self.rot(x)
        return rotation_tensor(r[:, 0], r[:, 1], r[:, 2], b), self.trans(x).view(b, 3, 1, 1)


class CameraProjection(nn.Module):
    """
    Does the camera transforms and multi-view projection
    Described in the paper.
    """
    def __init__(self, num_cameras):
        super(CameraProjection, self).__init__()
        self.cameras = nn.ParameterList()
        self.cam_rot = nn.ParameterList()
        for c in range(num_cameras):
            self.cameras.append(nn.Parameter(torch.rand(4)*2-1))
            self.cam_rot.append(nn.Parameter(torch.rand(3)*np.pi))

    def forward(self, x, rot, trans):
        # X is a list of [F, x,y,z] feature maps
        # or X is a [C, W, H] feature map
        # rot, trans are the extensic camera parameters
        if isinstance(x, list):
            # if it is a list, process each feature map
            # resulting in a [C, W, H] as input
            output = [self.forward(f, rot, trans) for f in x]
            return torch.cat(output, dim=1) # channels is dim1
        # x is now a [F, x,y,z] input where F is the feature
        fts = x[:, :-3] # get feature value, a B x F x H x W tensor
        pt = x[:, -3:] # get 3D point locations, a B x 3 x H x W tensor
        # rot is a 3x3 matrix
        # pw is 3x3 matrix applied along dim
        pw = torch.einsum('bphw,bpq->bqhw', pt, rot)
        pw += trans # add 3D translation
        # pw is now world coordinates at each feature map location
        # we do 2d projection next
        views = []
        for r, c in zip(self.cam_rot, self.cameras):
            rot = rotation_tensor(r[0].view(1,1,1), r[1].view(1,1,1), r[2].view(1,1,1))
            cam_pt = torch.einsum('bphw,pq->bqhw', pw, rot.squeeze(0))
            proj = torch.stack([(cam_pt[:, 0]*c[0] + c[2]), (cam_pt[:, 1]*c[1] + c[3])], dim=-1)
            proj = torch.tanh(proj) # apply tanh to get values in [-1,1]
            views.append(F.grid_sample(fts, proj))
        return torch.cat(views, dim=1)
