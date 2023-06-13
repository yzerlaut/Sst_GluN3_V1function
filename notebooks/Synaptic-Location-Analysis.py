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
from scipy import stats

# packages from Allen Institute:
from meshparty import meshwork # version 1.16.4
import pcg_skel # version 0.3.0 
from caveclient import CAVEclient # version 4.16.2

sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt


colors = {'BC':'tab:red', 'MC':'tab:orange'}

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
    DATASET['%s_id' % cType] = DATASET['cell_types']['pt_root_id'][\
                                    DATASET['cell_types']['cell_type_manual']==cType]


# %%
datastack_name = 'minnie65_public_v343'
client = CAVEclient(datastack_name)
client.materialize.version = 343
client.materialize.get_tables() # to explore the available data

# %%
nucleus_table = client.materialize.query_table('nucleus_detection_v0')

# %%
cell_table = client.materialize.query_table('allen_v1_column_types_slanted')


# %% [markdown]
# ## 2) Analyze presynaptic locations

# %%
def load_cell(ID):
    """
    we translate everything in terms of skeleton indices ! (mesh properties)
    """
    nrn = meshwork.load_meshwork(os.path.join(datafolder, 'skeletons', 'meshwork', '%s.h5'%ID))
    nrn.root_id = ID

    # 1) splitting axon and dendrites !
    axon_inds, Q = meshwork.algorithms.split_axon_by_annotation(nrn, 
                                                                pre_anno='pre_syn',
                                                                post_anno='post_syn')
    if Q>0.5:
        nrn.axon_inds = nrn.skeleton.mesh_to_skel_map[axon_inds]
        nrn.is_axon = np.array([(m in nrn.axon_inds) for m in nrn.skeleton_indices], dtype=bool)
        # dendritic cover paths
        nrn.dend_cover_paths = []
        for cover_path in nrn.skeleton.cover_paths:
            path = [c for c in cover_path if c not in nrn.axon_inds]
            if len(path)>0:
                nrn.dend_cover_paths.append(path)
    else:
        print('axon splitting not trusted...')
        
    # 2) get synapses:
    nrn.post_syn_sites = nrn.skeleton.mesh_to_skel_map[nrn.anno['post_syn']['mesh_ind']]
    
    # 3) check that they do belong to the dendrite !
    syn_in_dend_cond = np.array([(i not in nrn.skeleton_indices[nrn.is_axon]) for i in nrn.post_syn_sites])
    nrn.fraction_synapses_excluded = np.sum(~syn_in_dend_cond)/len(nrn.post_syn_sites)
    # and restrict to those
    nrn.post_syn_sites = nrn.post_syn_sites[syn_in_dend_cond]
    
    # 4) compute the path distance to soma
    nrn.syn_dist_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in nrn.post_syn_sites]
    
    return nrn

for cType in ['MC', 'BC']:
    DATASET[cType+'_nrn'] = []
    DATASET[cType+'_fraction_synapses_excluded'] = []
    for i, ID in enumerate(DATASET['%s_id' % cType]):
        DATASET[cType+'_nrn'].append(load_cell(ID))

# %%
fig1, AX = plt.subplots(1, 2, figsize=(3.5,1.2))
plt.subplots_adjust(wspace=0.1)
fig2, AX2= plt.subplots(1, 5, figsize=(5,1))
plt.subplots_adjust(wspace=2)


x = np.linspace(0, 270, 25)

def peak_func(values, x=x):
    hist, _ = np.histogram(values, bins=x)
    return x[1:][np.argmax(hist)]

for c, ax, cType in zip(range(2), AX, ['BC', 'MC']):
    HIST, NSYNS = [] ,[]
    for i, nrn in enumerate(DATASET['%s_nrn' % cType]):
        hist, _ = np.histogram(nrn.syn_dist_to_soma, bins=x, density=False)
        ax.plot(x[:-1], hist, lw=0.05, color=colors[cType])
        HIST.append(hist)
        NSYNS.append(len(nrn.post_syn_sites))
    
    pt.plot(x[1:-1], np.mean(HIST, axis=0)[1:], sy=np.std(HIST, axis=0)[1:], ax=ax, color=colors[cType])
    pt.scatter([0], [np.mean(HIST, axis=0)[0]], sy=[np.std(HIST, axis=0)[0]], ax=ax, color=colors[cType], ms=2)
    pt.annotate(ax, '%i$\pm$%i syn. / cell \n n=%i cells ' % (np.mean(NSYNS),
                                                             np.std(NSYNS),
                                                             len(DATASET[cType+'_id'])), 
                (1,1), ha='right', va='top', fontsize=7, color=colors[cType])
    
for ax, func, label in zip(AX2,
                           [np.mean, np.median, peak_func, np.std, stats.skew],
                           ['mean dist.($\mu$m)', 'median dist. ($\mu$m)',
                            'distr. peak ($\mu$m)', 'distr. s.d.', 'distr. skewness']):
    
    for c, cType in zip(range(2), ['BC', 'MC']):
        values = [func(nrn.syn_dist_to_soma) for nrn in DATASET['%s_nrn' % cType]]
        ax.bar([c], [np.mean(values)], yerr=[stats.sem(values)], color=colors[cType])
        pt.annotate(ax, '%.2f$\pm$%.2f' % (np.mean(values), np.std(values)) + (1+c)*'\n', 
                    (0.5,1), ha='center', fontsize=5, color=colors[cType])
    pt.annotate(ax, 'p=%.2e' % stats.mannwhitneyu(*[\
                                                   [func(nrn.syn_dist_to_soma) for nrn in DATASET['%s_nrn' % cType]]\
                                                   for cType in ['BC', 'MC']]).pvalue, 
                    (0.5,1), ha='center', fontsize=5, color='grey')
        
    pt.set_plot(ax, ylabel=label, xticks=[])
    
pt.set_plot(AX[0], xticks=np.arange(3)*100, 
            ylabel='syn. count', xlabel=38*' '+'path dist. to soma ($\mu$m)')
pt.set_plot(AX[1], yticks_labels=[], xticks=np.arange(3)*100)

pt.set_common_ylims(AX)
pt.set_plot(ax3, ylabel='cum. proba', xlabel='path dist. to soma ($\mu$m)')


# %% [markdown]
# ## Linear Density

# %%
def compute_single_cell(nrn, 
                        bins = np.linspace(30, 350, 20),
                        verbose=True, with_fig=False):
    
    if with_fig:
        fig, AX = plt.subplots(1, 2, figsize=(12,4))

    DENSITY_HIST, Ntot_syn = [], 0

    for i, path in enumerate(nrn.dend_cover_paths):
        
        # we plot all paths with a different color
        if with_fig:
            AX[0].plot(nrn.skeleton.vertices[path,0]/1e3, 
                       nrn.skeleton.vertices[path,1]/1e3)
    
        path_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in path]

        count_along_path = np.zeros(len(path))
        for i, p in enumerate(path):
            count_along_path[i] = np.sum(nrn.post_syn_sites==p)

        binned_dist = np.digitize(path_to_soma, bins=bins)
        density_hist = np.ones(len(bins))*np.nan # nan by default
        for b in range(1, len(bins)-1):
            if np.sum(binned_dist==b)>0:
                # we sum all synapses that fall into this bin and we divide by the bin length
                density_hist[b] = np.sum(count_along_path[binned_dist==b])/(bins[1]-bins[0])

        DENSITY_HIST.append(density_hist)
        Ntot_syn += count_along_path.sum()
    
    # add
    nrn.linear_density = np.nanmean(DENSITY_HIST, axis=0)
    nrn.linear_density_bins = bins
    
    if verbose:
        print('synapses counted: %i/%i' % (Ntot_syn, len(nrn.anno['post_syn']['mesh_ind'])))

    if with_fig:
        AX[1].plot(bins[1:], np.nanmean(DENSITY_HIST, axis=0)[1:]) # only non-infinite values contributing

        AX[0].set_title('looping over dendritic paths')
        AX[0].axis('equal')
        
        AX[1].set_xlabel('path dist. to soma ($\mu$m)'); 
        AX[1].set_ylabel('linear density (syn./$\mu$m)')
    else:
        fig = None
        
compute_single_cell(DATASET['MC_nrn'][0], with_fig=True)

# %%
bins = np.linspace(25, 300, 25)
for c, ax, cType in zip(range(2), AX, ['MC', 'BC']):
    for i, nrn in enumerate(DATASET['%s_nrn' % cType]):
        compute_single_cell(nrn, bins=bins, verbose=False)

# %%
fig1, AX = plt.subplots(1, 2, figsize=(3,1))
plt.subplots_adjust(wspace=0.1)
fig2, AX2= plt.subplots(1, 3, figsize=(3,1))
plt.subplots_adjust(wspace=2)
fig3, ax3 = plt.subplots(1, figsize=(1.4,1))

for c, ax, cType in zip(range(2), AX, ['BC', 'MC']):
    HIST, SLOPES = [], []
    for i, nrn in enumerate(DATASET['%s_nrn' % cType]):
        ax.plot(nrn.linear_density_bins, nrn.linear_density, lw=0.1, color=colors[cType])
        HIST.append(nrn.linear_density)
        # now slopes
        cond = np.isfinite(nrn.linear_density)
        pol = np.polyfit(nrn.linear_density_bins[cond],
                         100*nrn.linear_density[cond]/np.max(nrn.linear_density[cond]), 1)
        SLOPES.append(100.*pol[0])
        
    pt.plot(bins, np.nanmean(HIST, axis=0), sy=np.nanstd(HIST, axis=0), ax=ax, color=colors[cType])
    pt.annotate(ax, 'n=%i' % len(HIST), (1,1), va='top', ha='right', color=colors[cType], fontsize=7)
    means = [np.nanmean(nrn.syn_dist_to_soma) for nrn in DATASET['%s_nrn' % cType]]
    medians = [np.median(nrn.syn_dist_to_soma) for nrn in DATASET['%s_nrn' % cType]]
    
    AX2[0].bar([c], [np.mean(SLOPES)], yerr=[stats.sem(SLOPES)], color=colors[cType])
    maxs = bins[1:][np.array([np.argmax(hist[1:]) for hist in HIST])]
    #AX2[1].bar([c], [np.mean(medians)], yerr=[stats.sem(medians)], color=colors[cType])
    AX2[2].bar([c], [np.mean(maxs)], yerr=[stats.sem(maxs)], color=colors[cType])

pt.set_plot(AX2[0], ylabel='lin. density slope\n (% var. / 100 $\mu$m)', xticks=[])
    
pt.set_plot(AX[0], xticks=50+np.arange(3)*100,
            ylabel='lin. density \n (syn. / $\mu$m)', xlabel=35*' '+'path dist. to soma ($\mu$m)')
pt.set_plot(AX[1], xticks=50+np.arange(3)*100, yticks_labels=[])
pt.set_common_ylims(AX)
pt.set_plot(ax3, ylabel='cum. proba', xlabel='path dist. to soma ($\mu$m)')

# %%
bins = [25, 75, 150, 400]
for c, ax, cType in zip(range(2), AX, ['MC', 'BC']):
    for i, nrn in enumerate(DATASET['%s_nrn' % cType]):
        compute_single_cell(nrn, bins=bins, verbose=False)

# %%
fig1, AX = plt.subplots(1, 2, figsize=(3,1))
plt.subplots_adjust(wspace=0.1)
fig2, AX2= plt.subplots(1, 3, figsize=(3,1))
plt.subplots_adjust(wspace=2)
fig3, ax3 = plt.subplots(1, figsize=(1.4,1))

for c, ax, cType in zip(range(2), AX, ['BC', 'MC']):
        
    ax.bar(bins[1:], np.nanmean(HIST, axis=0), sy=np.nanstd(HIST, axis=0), ax=ax, color=colors[cType])
    pt.annotate(ax, 'n=%i' % len(HIST), (1,1), va='top', ha='right', color=colors[cType], fontsize=7)
    means = [np.nanmean(nrn.syn_dist_to_soma) for nrn in DATASET['%s_nrn' % cType]]
    medians = [np.median(nrn.syn_dist_to_soma) for nrn in DATASET['%s_nrn' % cType]]
    
    AX2[0].bar([c], [np.mean(SLOPES)], yerr=[stats.sem(SLOPES)], color=colors[cType])
    maxs = bins[1:][np.array([np.argmax(hist[1:]) for hist in HIST])]
    #AX2[1].bar([c], [np.mean(medians)], yerr=[stats.sem(medians)], color=colors[cType])
    AX2[2].bar([c], [np.mean(maxs)], yerr=[stats.sem(maxs)], color=colors[cType])

pt.set_plot(AX2[0], ylabel='lin. density slope\n (% var. / 100 $\mu$m)', xticks=[])
    
pt.set_plot(AX[0], xticks=50+np.arange(3)*100,
            ylabel='lin. density \n (syn. / $\mu$m)', xlabel=30*' '+'path dist. to soma ($\mu$m)')
pt.set_plot(AX[1], xticks=50+np.arange(3)*100, yticks_labels=[])
pt.set_common_ylims(AX)
pt.set_plot(ax3, ylabel='cum. proba', xlabel='path dist. to soma ($\mu$m)')


# %%
def compute_single_cell(nrn, 
                        bins = np.linspace(0, 400, 50),
                        with_fig=False):
    
    if with_fig:
        fig, AX = plt.subplots(1, 2, figsize=(12,4))

    DENSITY_HIST, Ntot_syn = [], 0

    for i, path in enumerate(nrn.dend_cover_paths):
        
        # we plot all paths with a different color
        if with_fig:
            AX[0].plot(nrn.skeleton.vertices[path,0]/1e3, 
                       nrn.skeleton.vertices[path,1]/1e3)
    
        path_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in path]

        count_along_path = np.zeros(len(path))
        for i, p in enumerate(path):
            count_along_path[i] = np.sum(nrn.post_syn_sites==p)

        binned_dist = np.digitize(path_to_soma, bins=bins)
        density_hist = np.ones(len(bins))*np.nan # nan by default
        for b in range(len(bins)):
            if np.sum(binned_dist==b)>0:
                # we sum all synapses that fall into this bin and we divide by the bin length
                density_hist[b] = np.sum(count_along_path[binned_dist==b])/(bins[1]-bins[0])

        DENSITY_HIST.append(density_hist)
        Ntot_syn += count_along_path.sum()
    
    print('synapses counted: %i/%i' % (Ntot_syn, len(nrn.anno.post_syn.df['post_pt_mesh_ind'])))

    if with_fig:
        AX[1].plot(bins, np.nanmean(DENSITY_HIST, axis=0)) # only non-infinite values contributing

        AX[0].set_title('looping over dendritic paths')
        AX[0].axis('equal')
        
        AX[1].set_xlabel('path dist. to soma ($\mu$m)'); 
        AX[1].set_ylabel('linear density (syn./$\mu$m)')
    else:
        fig = None
        
    return bins, np.nanmean(DENSITY_HIST, axis=0), fig



# %%
DATASET['inhibitory_synapses_onto_column']

# %%
DATASET.keys()

# %%
fig, AX = plot_all_morphologies('MC', axon_color=None)

# %%
fig, AX = plot_all_morphologies('BC', axon_color=None)

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
