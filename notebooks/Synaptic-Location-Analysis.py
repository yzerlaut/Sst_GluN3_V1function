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
# # Analyze Synaptic Locations in Martinotti and Basket Cells

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
    DATASET['%s_id' % cType] = DATASET['cell_types']['pt_root_id'][DATASET['cell_types']['cell_type_manual']=='MC']
    

# %% [markdown]
# ## 2) Plot some morphologies

# %%
cType = 'MC'

Nx = 4
Ny = int(len(DATASET['%s_id' % cType]) / Nx)
fig, AX = pt.plt.subplots(Ny+1, Nx, figsize=(1.4*Nx, 1.2*Ny))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

def correction(df):
    fixed = ''
    with open(df, "r") as file:
        for i, line in enumerate(file):
            if i==0:
                fixed += '0 1 '+line[4:]
            else:
                fixed += line
    with open(df.replace('swc', '_fixed'), 'w') as f:
        f.write(fixed)
        
for i, ID in enumerate(DATASET['%s_id' % cType]):
    df = datafolder+'/skeletons/swc/%s.swc'%ID
    correction(df)
    morpho = nrn.Morphology.from_swc_file(df.replace('swc', '_fixed')) 
    SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)
    vis = nrnvyz(SEGMENTS)
    vis.plot_segments(cond=(SEGMENTS['comp_type']!='axon'), 
                      ax=AX[int(i/N)][i%N])
    vis.plot_segments(cond=(SEGMENTS['comp_type']=='axon'),
                      color='tab:blue',                                           
                      bar_scale_args=None,
                      ax=AX[int(i/N)][i%N])
    
    AX[int(i/N)][i%N].set_title('%i ) %s' % (i+1, ID), fontsize=6)

while i<(Nx*(Ny+1)-1):
    i+=1
    AX[int(i/N)][i%N].axis('off')

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
