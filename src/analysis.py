import sys, os, pathlib
import numpy as np

stat_test_props = dict(interval_pre=[-1.5,0],
                       interval_post=[0.5,2],
                       test='ttest',
                       positive=True)

response_significance_threshold = 0.01

def selectivity_index(angles, resp):
    """
    computes the selectivity index: (Pref-Orth)/(Pref+Orth)
    clipped in [0,1]
    """
    imax = np.argmax(resp)
    iop = np.argmin(((angles[imax]+90)%(180)-angles)**2)
    if (resp[imax]>0):
        return min([1,max([0,(resp[imax]-resp[iop])/(resp[imax]+resp[iop])])])
    else:
        return 0

def shift_orientation_according_to_pref(angle, 
                                        pref_angle=0, 
                                        start_angle=-45, 
                                        angle_range=360):
    new_angle = (angle-pref_angle)%angle_range
    if new_angle>=angle_range+start_angle:
        return new_angle-angle_range
    else:
        return new_angle
