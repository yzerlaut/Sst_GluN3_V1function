# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plot Morphologies

# %%
import sys, os

import pandas
import numpy as np

sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz

sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt

# %% [markdown]
# ## 1) Load and build dataset

# %%
datafolder = '../data/SchneiderMizell_et_al_2023'
DATASET = {}

for key in os.listdir(datafolder):
    if 'csv' in key:
        DATASET[key.replace('.csv', '')] = pandas.read_csv('../data/SchneiderMizell_et_al_2023/%s' % key)
        
# ID of the two cell types of interest:
for cType in ['MC', 'BC']:
    DATASET['%s_id' % cType] = DATASET['cell_types']['pt_root_id'][DATASET['cell_types']['cell_type_manual']==cType]
    
# need to correct the swc files bur thei use in brian2
def swc_correction(ID):
    """
    a small correction to replace the 'dendrite' label
    for the soma in the allen swc files (3 is replaced by 1)
    """
    fixed = ''
    with open(os.path.join(datafolder, 'skeletons' , 'swc', '%s.swc'%ID), "r") as file:
        for i, line in enumerate(file):
            if i==0:
                fixed += '0 1 '+line[4:]
            else:
                fixed += line
    with open(os.path.join(datafolder, 'skeletons' , '_fixed', '%s.swc'%ID), 'w') as f:
        f.write(fixed)
        
for cType in ['MC', 'BC']:
    for i, ID in enumerate(DATASET['%s_id' % cType]):
        swc_correction(ID)


# %% [markdown]
# ## 2) Plot all morphologies

# %%
def plot_all_morphologies(cType, Nx = 4,
                          dendrite_color='k',
                          axon_color='tab:blue'):
    
    Ny = int(len(DATASET['%s_id' % cType]) / Nx)
    
    fig, AX = pt.plt.subplots(Ny+1, Nx, figsize=(1.7*Nx, 1.4*Ny))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    
    for i, ID in enumerate(DATASET['%s_id' % cType]):
        morpho = nrn.Morphology.from_swc_file(os.path.join(datafolder,
                                                           'skeletons', '_fixed', '%s.swc'%ID))
        SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)
        vis = nrnvyz(SEGMENTS)
        if axon_color is not None:
            vis.plot_segments(cond=(SEGMENTS['comp_type']=='axon'),
                              color=axon_color,
                              bar_scale_args={'Ybar':100, 'Xbar':1e-9,
                                              'Ybar_label':'100$\mu$m ', 'fontsize':6},
                              ax=AX[int(i/Nx)][i%Nx])
        vis.plot_segments(cond=(SEGMENTS['comp_type']!='axon'),
                          color=dendrite_color,
                          bar_scale_args=None,
                          ax=AX[int(i/Nx)][i%Nx])

        AX[int(i/Nx)][i%Nx].set_title('%i ) %s' % (i+1, ID), fontsize=6)

    while i<(Nx*(Ny+1)-1):
        i+=1
        AX[int(i/Nx)][i%Nx].axis('off')
    AX[-1][-1].annotate('dendrite', (0,1), xycoords='axes fraction', va='top', color=dendrite_color)
    AX[-1][-1].annotate('\naxon', (0,1), xycoords='axes fraction', va='top', color=axon_color)
    
    return fig, AX


# %%
fig, AX = plot_all_morphologies('MC')#, axon_color=None)

# %%
fig, AX = plot_all_morphologies('BC')#, axon_color=None)

# %% [markdown]
# # Visualize Morphology with synapses

# %%
np.flatnonzero(np.array(DATASET['BC_id'],dtype=int)==864691135100167712)

# %%
DATASET['BC_id']

# %%
df = '../data/SchneiderMizell_et_al_2023/skeletons/swc/864691134885028602.swc'

def correction(df):
    fixed = ''
    with open(df, "r") as file:
        for i, line in enumerate(file):
            if i==0:
                fixed += '0 1 '+line[4:]
                print(fixed)
            else:
                n = len('%i '%i)
                fixed += '%i '%(i)+line[n:]
    with open(df.replace('swc', '_fixed'), 'w') as f:
        f.write(fixed)


# %%
morpho = nrn.Morphology.from_swc_file(df.replace('swc', '_fixed'))

# %%
SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)
vis = nrnvyz(SEGMENTS)
vis.plot_segments(cond=(SEGMENTS['comp_type']!='axon'))


# %%
