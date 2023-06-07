# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# general python modules
import sys, os, pprint, pandas
import numpy as np
import matplotlib.pylab as plt
from scipy import stats

sys.path.append('../src')
from analysis import compute_tuning_response_per_cells
sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
sys.path.append('../')
import plot_tools as pt

folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'SST-WT-NR1-GluN3-2023')

# %%
import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)

DATASET = scan_folder_for_NWBfiles(folder,
                                   verbose=False)


# %%
# -------------------------------------------------- #
# ----    Pick the session datafiles and sort ------ #
# ----      them according to genotype ------------- #
# -------------------------------------------------- #

def init_summary(DATASET):
    
    SUMMARY = {'WT':{'FILES':[], 'subjects':[]}, 
               'GluN1':{'FILES':[], 'subjects':[]}, 
               'GluN3':{'FILES':[], 'subjects':[]}}

    for i, protocols in enumerate(DATASET['protocols']):

        # select the sessions
        if ('size-tuning' in protocols[0]):

            # sort the sessions according to the mouse genotype
            if ('NR1' in DATASET['subjects'][i]) or ('GluN1' in DATASET['subjects'][i]):
                SUMMARY['GluN1']['FILES'].append(DATASET['files'][i])
                SUMMARY['GluN1']['subjects'].append(DATASET['subjects'][i])
            elif 'GluN3' in DATASET['subjects'][i]:
                SUMMARY['GluN3']['FILES'].append(DATASET['files'][i])
                SUMMARY['GluN3']['subjects'].append(DATASET['subjects'][i])
            else:
                SUMMARY['WT']['FILES'].append(DATASET['files'][i])
                SUMMARY['WT']['subjects'].append(DATASET['subjects'][i])
                
    return SUMMARY


# %%
from physion.analysis.protocols.size_tuning import center_and_compute_size_tuning

def run_dataset_analysis(DATASET,
                         quantity='dFoF',
                         roi_to_neuropil_fluo_inclusion_factor=1.2,
                         neuropil_correction_factor = 0.8,
                         method_for_F0 = 'sliding_percentile',
                         percentile=5., # percent
                         sliding_window = 180.0, # seconds
                         Nmax=999, # max datafiles (for debugging)
                         verbose=True):

    SUMMARY = init_summary(DATASET)
    
    SUMMARY['quantity'] = quantity
    SUMMARY['quantity_args'] = dict(roi_to_neuropil_fluo_inclusion_factor=\
                                        roi_to_neuropil_fluo_inclusion_factor,
                                    method_for_F0=method_for_F0,
                                    percentile=percentile,
                                    sliding_window=sliding_window,
                                    neuropil_correction_factor=neuropil_correction_factor)
    
    for key in ['WT', 'GluN1']:

        for k in ['RESPONSES', 'CENTERED_ROIS', 'PREF_ANGLES']:
            SUMMARY[key][k] = [] 

        for f in SUMMARY[key]['FILES'][:Nmax]:

            print('analyzing "%s" [...] ' % f)
            data = Data(f, verbose=False)

            if quantity=='dFoF':
                data.build_dFoF(roi_to_neuropil_fluo_inclusion_factor=\
                                        roi_to_neuropil_fluo_inclusion_factor,
                                method_for_F0=method_for_F0,
                                percentile=percentile,
                                sliding_window=sliding_window,
                                neuropil_correction_factor=neuropil_correction_factor,
                                verbose=False)
            elif quantity=='rawFluo':
                data.build_rawFluo(verbose=verbose)
            elif quantity=='neuropil':
                data.build_neuropil(verbose=verbose)            
            else:
                print('quantity not recognized !!')

            #print('-->', data.vNrois)
            radii, size_resps, rois, pref_angles = center_and_compute_size_tuning(data,
                                                                                  imaging_quantity=quantity,
                                                                                  with_rois_and_angles=True,
                                                                                  verbose=False)
            if len(size_resps)>0:
                for k, q in zip(['RESPONSES', 'CENTERED_ROIS', 'PREF_ANGLES'],
                                [size_resps, rois, pref_angles]):
                    SUMMARY[key][k].append(q)

            if len(radii)>0:
                SUMMARY['radii'] = radii
                
    return SUMMARY


# %%
for quantity in ['rawFluo', 'neuropil', 'dFoF']:
    SUMMARY = run_dataset_analysis(DATASET, quantity=quantity, verbose=False)
    np.save('data/%s-summary.npy' % quantity, SUMMARY)
    
for neuropil_correction_factor in [0.7, 0.8, 0.9, 1.]:
    # rawFluo
    SUMMARY = run_dataset_analysis(DATASET, quantity='dFoF', 
                                   neuropil_correction_factor=neuropil_correction_factor,
                                   verbose=False)
    np.save('data/factor-neuropil-%.1f-summary.npy' % neuropil_correction_factor, SUMMARY)
    
for roi_to_neuropil_fluo_inclusion_factor in [1.1, 1.15, 1.2, 1.25, 1.3]:
    # rawFluo
    SUMMARY = run_dataset_analysis(DATASET, 
                                   quantity='dFoF', 
                                   roi_to_neuropil_fluo_inclusion_factor=roi_to_neuropil_fluo_inclusion_factor,
                                   verbose=False)
    np.save('data/inclusion-factor-neuropil-%.1f-summary.npy' % roi_to_neuropil_fluo_inclusion_factor, SUMMARY)


# %%

def plot_summary(SUMMARY,
                 average_by='sessions',
                 ms=2):
    
    fig, AX = pt.plt.subplots(1, 2, figsize=(3.5,1.))
    #fig, AX = pt.plt.subplots(1, 3, figsize=(7,1.5))
    AX[0].annotate('average\nover\n%s' % average_by,
                   (-0.8, 0.5), va='center', ha='center', xycoords='axes fraction')
    plt.subplots_adjust(wspace=0.7)

    center_index = 2

    for i, key, color in zip(range(2), ['WT', 'GluN1'], ['k', 'tab:blue']):

        if (len(SUMMARY[key]['RESPONSES'])>0) and (len(SUMMARY[key]['RESPONSES'][0])>0):

            if average_by=='sessions':
                resp = np.array([np.mean(r, axis=0) for r in SUMMARY[key]['RESPONSES']])
            else:
                resp = np.concatenate([r for r in SUMMARY[key]['RESPONSES']])
                #resp = np.concatenate([r/np.max(r) for r in SUMMARY[key]['RESPONSES'] if np.max(r)>0])

            #AX[0].plot(SUMMARY['radii'][1:], np.mean(resp, axis=0)[1:], 'o', ms=2, color=color)
            pt.scatter(SUMMARY['radii'][1:]+i*.3, np.mean(resp, axis=0)[1:],
                       sy=stats.sem(resp, axis=0)[1:],
                       ax=AX[i], color=color, ms=ms)

            """
            center_norm_resp = np.array([r/rn for r, rn in zip(resp, resp[:,center_index])])
            pt.scatter(SUMMARY['radii'], np.mean(center_norm_resp, axis=0),
                       sy=stats.sem(center_norm_resp, axis=0),
                       ax=AX[1], color=color, ms=ms)

            ff_norm_resp = np.array([r/rn for r, rn in zip(resp, resp[:,-1])])
            pt.scatter(SUMMARY['radii'], np.mean(ff_norm_resp, axis=0),
                       sy=stats.sem(ff_norm_resp, axis=0),
                       ax=AX[2], color=color, ms=ms)
            """

        AX[i].annotate('n=%i%s' % (len(resp), average_by), (1,0),
                        ha='right', color=color, xycoords='axes fraction')
        AX[i].set_title(key, color=color)

    #AX[1].plot([SUMMARY['radii'][center_index], SUMMARY['radii'][center_index], 0],
    #           [0,1,1], 'k:', lw=0.5)
    #AX[2].plot([SUMMARY['radii'][-1], SUMMARY['radii'][-1], 0], [0,1,1], 'k:', lw=0.5)

    for ax in AX:
        pt.set_plot(ax, xlabel='stim. size ($^o$)',
                    #xscale='log', xticks=[10,100], xticks_labels=['10', '100'], xlim=[9,101],
                    ylabel='$\delta$ %s' % SUMMARY['quantity'].replace('dFoF', '$\Delta$F/F'))

    return fig

SUMMARY = np.load('data/dFoF-summary.npy', allow_pickle=True).item()
fig = plot_summary(SUMMARY, average_by='ROIs')
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'final.svg'))

# %%
for quantity in ['rawFluo', 'neuropil', 'dFoF']:
    SUMMARY = np.load('data/%s-summary.npy' % quantity, allow_pickle=True).item()
    fig = plot_summary(SUMMARY)

# %%
for neuropil_correction_factor in [0.7, 0.8, 0.9, 1.]:
    SUMMARY = np.load('data/factor-neuropil-%.1f-summary.npy' % neuropil_correction_factor,
                      allow_pickle=True).item()
    fig = plot_summary(SUMMARY)
    fig.suptitle('Neuropil-factor for substraction: %.1f' % neuropil_correction_factor)

# %%
for roi_to_neuropil_fluo_inclusion_factor in [1.1, 1.15, 1.2, 1.25, 1.3]:
    SUMMARY = np.load('data/inclusion-factor-neuropil-%.1f-summary.npy' % roi_to_neuropil_fluo_inclusion_factor,
                      allow_pickle=True).item()
    fig = plot_summary(SUMMARY)
    fig.suptitle('roiFluo/Neuropil inclusion-factor: %.2f' % roi_to_neuropil_fluo_inclusion_factor)

# %%
from physion.analysis.protocols.size_tuning import center_and_compute_size_tuning

for key in ['WT', 'GluN1']:

    for k in ['RESPONSES', 'CENTERED_ROIS', 'PREF_ANGLES']:
        SUMMARY[key][k] = [] 

    for f in SUMMARY[key]['FILES']:
        
        print('analyzing "%s" [...] ' % f)
        data = Data(f, verbose=False)
        data.build_dFoF(method_for_F0='sliding_percentile',
                        percentile=10,
                        sliding_window=180)
        print('-->', data.vNrois)
        
        radii, size_resps, rois, pref_angles = center_and_compute_size_tuning(data, 
                                                                              with_rois_and_angles=True,
                                                                              verbose=False)
        
        if len(size_resps)>0:
            for k, q in zip(['RESPONSES', 'CENTERED_ROIS', 'PREF_ANGLES'],
                            [size_resps, rois, pref_angles]):
                SUMMARY[key][k].append(q)
        
        if len(radii)>0:
            SUMMARY['radii'] = radii

# %%
fig, AX = pt.plt.subplots(1, 3, figsize=(7,1.5))
AX[0].annotate('average\nover\nsessions', (-0.8, 0.5), va='center', ha='center', xycoords='axes fraction')
plt.subplots_adjust(wspace=0.5)

center_index = 2

for i, key, color in zip(range(3), ['WT', 'GluN1'], ['k', 'tab:blue']):
    
    if (len(SUMMARY[key]['RESPONSES'])>0) and (len(SUMMARY[key]['RESPONSES'][0])>0):
        
        resp = np.array([np.mean(r, axis=0) for r in SUMMARY[key]['RESPONSES']])
        
        AX[0].plot(SUMMARY['radii'][1:], np.mean(resp, axis=0)[1:], 'o', ms=2, color=color)
        pt.plot(SUMMARY['radii'], np.mean(resp, axis=0), sy=np.std(resp, axis=0),
                ax=AX[0], color=color)
        
        center_norm_resp = np.array([r/rn for r, rn in zip(resp, resp[:,center_index])])
        pt.plot(SUMMARY['radii'], np.mean(center_norm_resp, axis=0), sy=np.std(center_norm_resp, axis=0),
                ax=AX[1], color=color)

        ff_norm_resp = np.array([r/rn for r, rn in zip(resp, resp[:,-1])])
        pt.plot(SUMMARY['radii'], np.mean(ff_norm_resp, axis=0), sy=np.std(ff_norm_resp, axis=0),
                ax=AX[2], color=color)
        
    AX[-1].annotate(i*'\n'+'%s, N=%i sessions' % (key, len(resp)), (1,1),
                    va='top', color=color, xycoords='axes fraction')

AX[1].plot([SUMMARY['radii'][center_index], SUMMARY['radii'][center_index], 0],
           [0,1,1], 'k:', lw=0.5)
AX[2].plot([SUMMARY['radii'][-1], SUMMARY['radii'][-1], 0], [0,1,1], 'k:', lw=0.5)

for ax, ylabel in zip(AX,
                     ['$\delta$ $\Delta$F/F', 'center norm. $\delta$ $\Delta$F/F', 'F.F. norm. $\delta$ $\Delta$F/F']):
    pt.set_plot(ax, xlabel='size ($^o$)', ylabel=ylabel)    

#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'final.svg'))

# %%
fig, AX = pt.plt.subplots(1, 3, figsize=(7,1.5))
AX[0].annotate('average\nover\nsessions', (-0.8, 0.5), va='center', ha='center', xycoords='axes fraction')
plt.subplots_adjust(wspace=0.5)

center_index = 2

for i, key, color in zip(range(3), ['WT', 'GluN1'], ['k', 'tab:blue']):
    
    if (len(SUMMARY[key]['RESPONSES'])>0) and (len(SUMMARY[key]['RESPONSES'][0])>0):
        
        resp = np.array([np.mean(r, axis=0) for r in SUMMARY[key]['RESPONSES']])

        AX[0].plot(SUMMARY['radii'][1:], np.mean(resp, axis=0)[1:], 'o', ms=2, color=color)
        pt.plot(SUMMARY['radii'][1:], np.mean(resp, axis=0)[1:], sy=np.std(resp, axis=0)[1:],
                ax=AX[0], color=color)
        
        center_norm_resp = np.array([r/rn for r, rn in zip(resp, resp[:,center_index])])
        pt.plot(SUMMARY['radii'][1:], np.mean(center_norm_resp, axis=0)[1:], sy=np.std(center_norm_resp, axis=0)[1:],
                ax=AX[1], color=color)

        ff_norm_resp = np.array([r/rn for r, rn in zip(resp, resp[:,-1])])
        pt.plot(SUMMARY['radii'][1:], np.mean(ff_norm_resp, axis=0)[1:], sy=np.std(ff_norm_resp, axis=0)[1:],
                ax=AX[2], color=color)
        
    AX[-1].annotate(i*'\n'+'%s, N=%i sessions' % (key, len(resp)), (1,1),
                    va='top', color=color, xycoords='axes fraction')

AX[1].plot([SUMMARY['radii'][center_index], SUMMARY['radii'][center_index], 0],
           [0,1,1], 'k:', lw=0.5)
AX[2].plot([SUMMARY['radii'][-1], SUMMARY['radii'][-1], 0], [0,1,1], 'k:', lw=0.5)

for ax, ylabel in zip(AX,
                     ['$\delta$ $\Delta$F/F', 'center norm. $\delta$ $\Delta$F/F', 'F.F. norm. $\delta$ $\Delta$F/F']):
    pt.set_plot(ax, xlabel='size ($^o$)', ylabel=ylabel, xscale='log', xlim=[9,110], 
                xticks=[10, 100], xticks_labels=['10', '100'])    

#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'final.svg'))

# %%
fig, AX = pt.plt.subplots(1, 3, figsize=(7,1.5))
AX[0].annotate('average\nover\nrois', (-0.8, 0.5), va='center', ha='center', xycoords='axes fraction')
plt.subplots_adjust(wspace=0.5)

center_index = 2

for i, key, color in zip(range(3), ['WT', 'GluN1'], ['k', 'tab:blue']):
    
    if (len(SUMMARY[key]['RESPONSES'])>0) and (len(SUMMARY[key]['RESPONSES'][0])>0):
        
        resp = np.concatenate([r for r in SUMMARY[key]['RESPONSES']])

        #for r in SUMMARY[key]['RESPONSES']:
        #    AX[0].plot(SUMMARY['radii'], np.mean(r, axis=0), lw=0.1, color=color)
        
        pt.plot(SUMMARY['radii'], np.mean(resp, axis=0), sy=np.std(resp, axis=0),
                ax=AX[0], color=color)
        
        center_norm_resp = np.array([r/r[center_index] for r in resp])
        pt.plot(SUMMARY['radii'], np.mean(center_norm_resp, axis=0), sy=0*np.std(center_norm_resp, axis=0),
                ax=AX[1], color=color)

        ff_norm_resp = np.array([r/r[-1] for r in resp])
        pt.plot(SUMMARY['radii'], np.mean(ff_norm_resp, axis=0), sy=0*np.std(ff_norm_resp, axis=0),
                ax=AX[2], color=color)
        
    AX[-1].annotate(i*'\n'+'%s, n=%i rois' % (key, len(resp)), (1,1),
                    va='top', color=color, xycoords='axes fraction')

AX[1].plot([SUMMARY['radii'][center_index], SUMMARY['radii'][center_index], 0],
           [0,1,1], 'k:', lw=0.5)
AX[2].plot([SUMMARY['radii'][-1], SUMMARY['radii'][-1], 0], [0,1,1], 'k:', lw=0.5)

for ax, ylabel in zip(AX,
                     ['$\delta$ $\Delta$F/F', 'center norm. $\delta$ $\Delta$F/F', 'F.F. norm. $\delta$ $\Delta$F/F']):
    pt.set_plot(ax, xlabel='size ($^o$)', ylabel=ylabel)    

#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'final.svg'))

# %%
fig, AX = pt.plt.subplots(1, 3, figsize=(7,1.5))
plt.subplots_adjust(wspace=0.5)

for i, key, color in zip(range(3), ['WT', 'GluN1'], ['k', 'tab:blue']):
    
    if (len(SUMMARY[key]['RESPONSES'])>0) and (len(SUMMARY[key]['RESPONSES'][0])>0):
        
        resp = [np.mean(r, axis=0) for r in SUMMARY[key]['RESPONSES']]

        for r in SUMMARY[key]['RESPONSES']:
            ax.plot(SUMMARY['radii'][1:], np.mean(r, axis=0)[1:], lw=0.1, color=color)
        
        pt.plot(SUMMARY['radii'][1:], np.mean(resp, axis=0)[1:], sy=np.std(resp, axis=0)[1:],
                ax=AX[0], color=color)
        
        pt.set_plot(AX[0], xscale='log', xlim=[9,101])
        
    AX[-1].annotate(i*'\n'+'%s, N=%i sessions' % (key, len(SUMMARY[key]['RESPONSES'])), (1,1),
                    va='top', color=color, xycoords='axes fraction')

AX[0].set_ylabel('$\delta$ $\Delta$F/F')
for ax in AX:
    ax.set_xlabel('size ($^o$)')    

#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'final.svg'))

# %% [markdown]
# # Visualizing some evoked response in single ROI

# %%
data = Data(SUMMARY['WT']['FILES'][0])
data.protocols
data.get_protocol_id('size-tuning-protocol-loc')

# %%
import sys, os
import numpy as np
import matplotlib.pylab as plt
sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.utils import plot_tools as pt
from physion.dataviz.episodes.trial_average import plot_trial_average
sys.path.append('../')
import plot_tools as pt

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)


def cell_tuning_example_fig(filename,
                            contrast=1.0,
                            stat_test_props = dict(interval_pre=[-1,0], 
                                                   interval_post=[1,2],
                                                   test='ttest',
                                                   positive=True),
                            response_significance_threshold = 0.01,
                            Nsamples = 10, # how many cells we show
                            seed=10):
    
    np.random.seed(seed)
    
    data = Data(filename)
    
    EPISODES = EpisodeData(data,
                           quantities=['dFoF'],
                           protocol_id=np.flatnonzero(['8orientation' in p for p in data.protocols]),
                           with_visual_stim=True,
                           verbose=True)
    
    fig, AX = pt.plt.subplots(Nsamples, len(EPISODES.varied_parameters['angle']), 
                          figsize=(7,7))
    plt.subplots_adjust(right=0.75, left=0.1, top=0.97, bottom=0.05, wspace=0.1, hspace=0.8)
    
    for Ax in AX:
        for ax in Ax:
            ax.axis('off')

    for i, r in enumerate(np.random.choice(np.arange(data.vNrois), 
                                           min([Nsamples, data.vNrois]), replace=False)):

        # SHOW trial-average
        plot_trial_average(EPISODES,
                           condition=(EPISODES.contrast==contrast),
                           column_key='angle',
                           #color_key='contrast',
                           #color=['lightgrey', 'k'],
                           quantity='dFoF',
                           ybar=1., ybarlabel='1dF/F',
                           xbar=1., xbarlabel='1s',
                           roiIndex=r,
                           with_stat_test=True,
                           stat_test_props=stat_test_props,
                           with_screen_inset=True,
                           AX=[AX[i]], no_set=False)
        AX[i][0].annotate('roi #%i  ' % (r+1), (0,0), ha='right', xycoords='axes fraction')

        # SHOW summary angle dependence
        inset = pt.inset(AX[i][-1], (2.2, 0.2, 1.2, 0.8))

        angles, y, sy, responsive_angles = [], [], [], []
        responsive = False

        for a, angle in enumerate(EPISODES.varied_parameters['angle']):

            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=\
                                            EPISODES.find_episode_cond(key=['angle', 'contrast'],
                                                                       value=[angle, contrast]),
                                                            response_args=dict(quantity='dFoF', roiIndex=r),
                                                            **stat_test_props)

            angles.append(angle)
            y.append(np.mean(stats.y-stats.x))    # means "post-pre"
            sy.append(np.std(stats.y-stats.x))    # std "post-pre"

            if stats.significant(threshold=response_significance_threshold):
                responsive = True
                responsive_angles.append(angle)

        pt.plot(angles, np.array(y), sy=np.array(sy), ax=inset)
        inset.plot(angles, 0*np.array(angles), 'k:', lw=0.5)
        inset.set_ylabel('$\delta$ $\Delta$F/F     ', fontsize=7)
        inset.set_xticks(angles)
        inset.set_xticklabels(['%i'%a if (i%2==0) else '' for i, a in enumerate(angles)], fontsize=7)
        if i==(Nsamples-1):
            inset.set_xlabel('angle ($^{o}$)', fontsize=7)

        #SI = selectivity_index(angles, y)
        #inset.annotate('SI=%.2f ' % SI, (0, 1), ha='right', weight='bold', fontsize=8,
        #               color=('k' if responsive else 'lightgray'), xycoords='axes fraction')
        inset.annotate(('responsive' if responsive else 'unresponsive'), (1, 1), ha='right',
                        weight='bold', fontsize=6, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)),
                        xycoords='axes fraction')
        
    return fig

fig = cell_tuning_example_fig('/home/yann/CURATED/SST-WT-NR1-GluN3-2023/2023_02_15-13-30-47.nwb',
                             contrast=1)

# %%
fig = cell_tuning_example_fig(SUMMARY['GluN1']['FILES'][0])

# %% [markdown]
# # Visualizing some raw population data

# %%
import sys, os
import numpy as np
import matplotlib.pylab as plt
sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data
from physion.analysis.process_NWB import EpisodeData
from physion.utils import plot_tools as pt
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings

sys.path.append('../')
import plot_tools as pt

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)

data = Data('/home/yann/CURATED/SST-WT-NR1-GluN3-2023/2023_02_15-13-30-47.nwb',
            with_visual_stim=True)
data.init_visual_stim()

tlim = [984,1100]

settings = {'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
            'FaceMotion': {'fig_fraction': 1, 'subsampling': 1, 'color': 'purple'},
            'Pupil': {'fig_fraction': 2, 'subsampling': 1, 'color': '#d62728'},
             'CaImaging': {'fig_fraction': 4,
              'subsampling': 1,
              'subquantity': 'dF/F',
              'color': '#2ca02c',
              'roiIndices': np.random.choice(np.arange(data.nROIs), 5, replace=False)},
             'CaImagingRaster': {'fig_fraction': 3,
              'subsampling': 1,
              'roiIndices': 'all',
              'normalization': 'per-line',
              'subquantity': 'dF/F'},
             'VisualStim': {'fig_fraction': 0.5, 'color': 'black', 'with_screen_inset':True}}

plot_raw(data, tlim=tlim, settings=settings)

# %%
settings = {'Locomotion': {'fig_fraction': 1, 'subsampling': 2, 'color': '#1f77b4'},
            'FaceMotion': {'fig_fraction': 1, 'subsampling': 2, 'color': 'purple'},
            'Pupil': {'fig_fraction': 2, 'subsampling': 1, 'color': '#d62728'},
             'CaImaging': {'fig_fraction': 4,
              'subsampling': 1,
              'subquantity': 'dF/F',
              'color': '#2ca02c',
              'roiIndices': np.random.choice(np.arange(data.nROIs), 5, replace=False)},
             'CaImagingRaster': {'fig_fraction': 3,
              'subsampling': 1,
              'roiIndices': 'all',
              'normalization': 'per-line',
              'subquantity': 'dF/F'}}

tlim = [900, 1300]
plot_raw(data, tlim=tlim, settings=settings)

# %%
from physion.dataviz.imaging import show_CaImaging_FOV
show_CaImaging_FOV(data, key='max_proj', NL=3, roiIndices='all')

# %%
show_CaImaging_FOV(data, key='max_proj', NL=3)
