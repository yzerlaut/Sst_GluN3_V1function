# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)

DATASET = scan_folder_for_NWBfiles(folder,
                                   verbose=False)

# %%
# -------------------------------------------------- #
# ----    Pick the session datafiles and sort ------ #
# ----      them according to genotype ------------- #
# -------------------------------------------------- #

SUMMARY = {'WT':{'FILES':[], 'subjects':[]}, 
           'GluN1':{'FILES':[], 'subjects':[]}, 
           'GluN3':{'FILES':[], 'subjects':[]},
           # add a summary for half contrast
           'WT_c=0.5':{'FILES':[]},
           'GluN1_c=0.5':{'FILES':[]},
           'GluN3_c=0.5':{'FILES':[]}}

for i, protocols in enumerate(DATASET['protocols']):
    
    # select the sessions with different 
    if ('ff-gratings-8orientation-2contrasts-15repeats' in protocols) or\
        ('ff-gratings-8orientation-2contrasts-10repeats' in protocols):
        
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


# %%
# -------------------------------------------------- #
# ----   Loop over datafiles to compute    --------- #
# ----           the evoked responses      --------- #
# -------------------------------------------------- #

def orientation_selectivity_index(resp_pref, resp_90):
    """                                                                         
     computes the selectivity index: (Pref-Orth)/(Pref+Orth)                     
     clipped in [0,1] --> because resp_90 can be negative    
    """
    return np.clip((resp_pref-np.clip(resp_90, 0, np.inf))/(resp_pref+resp_90), 0, 1)

stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       positive=True) 
    
def compute_summary_responses(stat_test_props=dict(interval_pre=[-1.,0],                                   
                                                   interval_post=[1.,2.],                                   
                                                   test='anova',                                            
                                                   positive=True),
                              response_significance_threshold=5e-2):
    
    for key in ['WT', 'GluN1', 'GluN3']:

        SUMMARY[key]['RESPONSES'], SUMMARY[key]['OSI'], SUMMARY[key]['FRAC_RESP'] = [], [], []
        SUMMARY[key+'_c=0.5']['RESPONSES'], SUMMARY[key+'_c=0.5']['OSI'], SUMMARY[key+'_c=0.5']['FRAC_RESP'] = [], [], []

        for f in SUMMARY[key]['FILES']:

            data = Data(f, verbose=False)
            protocol = 'ff-gratings-8orientation-2contrasts-15repeats' if\
                        ('ff-gratings-8orientation-2contrasts-15repeats' in data.protocols) else\
                        'ff-gratings-8orientation-2contrasts-10repeats'

            # at full contrast
            responses, frac_resp, shifted_angle = compute_tuning_response_per_cells(data,
                                                                                    contrast=1,
                                                                                    protocol_name=protocol,
                                                                                    stat_test_props=stat_test_props,
                                                                                    response_significance_threshold=response_significance_threshold,
                                                                                    verbose=False)
            SUMMARY[key]['RESPONSES'].append(responses)
            SUMMARY[key]['OSI'].append([orientation_selectivity_index(r[1], r[5]) for r in responses])
            SUMMARY[key]['FRAC_RESP'].append(frac_resp)

            # for those two genotypes (not run for the GluN3-KO), we add:
            if key in ['WT', 'GluN1']:
                # at half contrast
                responses, frac_resp, shifted_angle = compute_tuning_response_per_cells(data,
                                                                                        contrast=0.5,
                                                                                        protocol_name=protocol,
                                                                                        stat_test_props=stat_test_props,
                                                                                        response_significance_threshold=response_significance_threshold,
                                                                                        verbose=False)
                SUMMARY[key+'_c=0.5']['RESPONSES'].append(responses)
                SUMMARY[key+'_c=0.5']['OSI'].append([orientation_selectivity_index(r[1], r[5]) for r in responses])
                SUMMARY[key+'_c=0.5']['FRAC_RESP'].append(frac_resp)
                
    SUMMARY['shifted_angle'] = shifted_angle
    
    return SUMMARY


# %%
SUMMARY = compute_summary_responses()


# %%
def plot_tunning_summary(shifted_angle,
                         frac_resp,
                         responses,
                         OSIs):
    """
    """
    fig, AX = pt.plt.subplots(1, 4, figsize=(8,1.5))
    pt.plt.subplots_adjust(wspace=0.8, top=0.7, bottom=0.3)

    RESPONSES = [np.mean(responses, axis=0) for responses in responses]
    
    # raw
    pt.plot(shifted_angle, np.mean(RESPONSES, axis=0),
            sy=stats.sem(RESPONSES, axis=0), ax=AX[0])
    
    
    AX[0].set_ylabel('evoked $\Delta$F/F')
    AX[0].set_title('raw resp.')

    for ax in AX[:2]:
        ax.set_xlabel('angle ($^o$)')
        ax.annotate('N=%i sessions'%len(responses), (1,1), fontsize=6,
                    va='top', ha='right', xycoords='axes fraction')

    # peak normalized
    N_RESP = [resp/resp[1] for resp in RESPONSES]
    pt.plot(shifted_angle, np.mean(N_RESP, axis=0),
            sy=stats.sem(N_RESP, axis=0), ax=AX[1])

    AX[1].set_yticks([0, 0.5, 1])
    AX[1].set_ylabel('n. $\Delta$F/F')
    AX[1].set_title('peak normalized')

    # orientation selectivity index
    AX[2].hist(np.concatenate([osi for osi in OSIs]), color='grey', bins=20, density=True)
    AX[2].set_xlabel('OSI')
    
    # fraction responsive
    pt.pie([np.mean(frac_resp), 1-np.mean(frac_resp)],
           pie_labels=['%.1f%%' % (100.*np.mean(frac_resp)),
                       '     %.1f%%' % (100.*(1-np.mean(frac_resp)))],
           COLORS=[pt.plt.cm.tab10(2), pt.plt.cm.tab10(1)], ax=AX[3])
    
    NTOTs = [int(len(resp)/fr) for fr, resp in zip(frac_resp, responses)]
    Ns = [len(resp) for resp in responses]
    AX[3].annotate('responsive ROIS :\nN=%i sessions\n n= %i$\pm$%i / %i$\pm$%i ROIs  ' % (len(responses),
                                                                        np.mean(Ns), np.std(Ns),
                                                                        np.mean(NTOTs), np.std(NTOTs)),
                   (0.5, 0), va='top', ha='center',
                   xycoords='axes fraction')

    return fig, AX

# %%
for key in ['WT', 'GluN1']:
    fig, AX = plot_tunning_summary(SUMMARY['shifted_angle'],
                                   SUMMARY[key]['FRAC_RESP'], 
                                   SUMMARY[key]['RESPONSES'],
                                   SUMMARY[key]['OSI'])
    AX[3].set_title(key+' mice');


# %%
def generate_comparison_figs(SUMMARY, case1, case2,
                             color1='k', color2='tab:blue'):
    
    fig1, AX1 = plt.subplots(1, 4, figsize=(6, 1))
    plt.subplots_adjust(wspace=0.7, top=0.9, bottom=0.2, right=0.95)
    AX1[0].annotate('average\nover\nsessions', (-1.2, 0.5), va='center', ha='center', xycoords='axes fraction')
    

    fig2, AX2 = plt.subplots(1, 4, figsize=(6, 1))
    plt.subplots_adjust(wspace=0.7, top=0.9, bottom=0.2, right=0.95)
    AX2[0].annotate('average\nover\nrois', (-1.2, 0.5), va='center', ha='center', xycoords='axes fraction')
    
    fig3, AX3 = pt.plt.subplots(1, 4, figsize=(6,1))
    plt.subplots_adjust(wspace=0.7, top=0.9, bottom=0.2, right=0.95)
    AX3[0].annotate('%s\n\n' % case1, (-1.2, 0.5), va='center', ha='center', xycoords='axes fraction', color=color1)
    AX3[0].annotate('\n\n%s' % case2, (-1.2, 0.5), va='center', ha='center', xycoords='axes fraction', color=color2)

    for i, key, color in zip(range(2),
                             [case1, case2],
                             [color1, color2]):

        # sessions vs ROIs -- averages
        for AX, RESPONSES, label in zip([AX1, AX2],
                                 [[np.mean(responses, axis=0) for responses in SUMMARY[key]['RESPONSES']],
                                   np.concatenate([responses for responses in SUMMARY[key]['RESPONSES']])],
                                 ['N=%i', 'n=%i']):
            
            # mean +/- sem vs mean +/- std plots
            for ax, func in zip([AX[0], AX[2]], [stats.sem, np.std]):
                pt.plot(SUMMARY['shifted_angle'], np.mean(RESPONSES, axis=0),
                        func(RESPONSES, axis=0),
                        ax=ax, color=color)
                
            for ax in AX:
                ax.annotate(i*'\n'+label%len(RESPONSES), (1,1), xycoords='axes fraction',
                            ha='right', va='top', color=color, fontsize=7)

            # responses normalized to prefered orientation
            N_RESP = [resp/resp[1] for resp in RESPONSES]
            # mean +/- sem vs mean +/- std plots
            for ax, func in zip([AX[1], AX[3]], [stats.sem, np.std]):
                pt.plot(SUMMARY['shifted_angle'], np.mean(N_RESP, axis=0),
                        func(N_RESP, axis=0),
                        ax=ax, color=color)
                
        # responsiveness
        AX3[0].bar([i],
                  [100*np.mean(SUMMARY[key]['FRAC_RESP'])],
                  yerr=[100*np.std(SUMMARY[key]['FRAC_RESP'])],
                  color=color)
        AX3[0].annotate(i*'\n\n'+'%.1f $\pm$ %.0f %%\n N=%i' % (100*np.mean(SUMMARY[key]['FRAC_RESP']),
                                                                         100*np.std(SUMMARY[key]['FRAC_RESP']),
                                                                        len(SUMMARY[key]['RESPONSES'])),
                        (1,1), xycoords='axes fraction', ha='right', va='top', color=color, fontsize=6)
        
        
        # violin plot of orientation selectivity index
        OSIs = np.concatenate([osis for osis in SUMMARY[key]['OSI']])
        pt.violin(OSIs, X=[i], ax=AX3[1], COLORS=[color])
        AX3[1].annotate(i*'\n\n'+'%.2f $\pm$ %.2f \n n=%i rois' % (np.mean(OSIs),
                                                                   np.std(OSIs),
                                                                   len(OSIs)),
                        (1,1), xycoords='axes fraction', ha='right', va='top', color=color, fontsize=6)
        
        # 
        hist, be = np.histogram(OSIs, bins=20, density=True)
        AX3[2].plot(.5*(be[:-1]+be[1:]), hist, color=color, lw=1)
        
        x = np.cumsum(hist)
        AX3[3].plot(.5*(be[:-1]+be[1:]), x/x[-1], color=color)

    for i, AX, label in zip(range(2),
                            [AX1, AX2],
                            ['N=%i sessions', 'n=%i ROIs']):
        for ax in AX:
            pt.set_plot(ax, xlabel='angle from pref. ($^o$)',
                xticks=SUMMARY['shifted_angle'],
                xticks_labels=['%.0f'%s if (i%2==1) else '' for i,s in enumerate(SUMMARY['shifted_angle'])])
            
        for ax in [AX[0], AX[2]]:
            ax.set_ylabel('evoked $\Delta$F/F')
        for ax in [AX[1], AX[3]]:
            ax.set_ylabel('norm. $\Delta$F/F')
        for ax in AX[:2]:
            ax.set_title('mean$\pm$s.e.m.', fontsize=6)
        for ax in AX[2:]:
            ax.set_title('mean$\pm$s.d.', fontsize=6)

    # statistical analysis
    pval = stats.mannwhitneyu([np.mean(responses) for responses in SUMMARY[case1]['RESPONSES']],
                              [np.mean(responses) for responses in SUMMARY[case2]['RESPONSES']]).pvalue
    AX3[0].set_title('Mann-Whitney:\np=%.1e' % pval, fontsize=6)
    pval = stats.mannwhitneyu(np.concatenate([osis for osis in SUMMARY[case1]['OSI']]),
                              np.concatenate([osis for osis in SUMMARY[case2]['OSI']])).pvalue
    AX3[1].set_title('Mann-Whitney:\np=%.1e' % pval, fontsize=6)
    
    pt.set_plot(AX3[0], ylabel='fraction (%)\nresponsive', xticks=[], yticks=[0,50,100], xlim=[-1, 6])
    pt.set_plot(AX3[1], ylabel='OSI', xticks=[], xlim=[-1, 6])
    pt.set_plot(AX3[2], ylabel='density', xlabel='OSI', xticks=[0, 0.5, 1])
    pt.set_plot(AX3[3], ylabel='cum. frac.', xlabel='OSI', yticks=[0,0.5,1], xticks=[0, 0.5, 1])

    return fig1, fig2, fig3

FIGS = generate_comparison_figs(SUMMARY, 'WT', 'GluN1',
                                color1='k', color2='tab:blue')

# %%
FIGS = generate_comparison_figs(SUMMARY, 'WT', 'WT_c=0.5',
                                color1='k', color2='grey')

# %%
FIGS = generate_comparison_figs(SUMMARY, 'GluN1', 'GluN1_c=0.5',
                                color1='tab:blue', color2='tab:cyan')

# %% [markdown]
# ## Testing different "visual-responsiveness" criteria

# %%
# most permissive
SUMMARY = compute_summary_responses(stat_test_props=dict(interval_pre=[-1.,0],                                   
                                                         interval_post=[1.,2.],                                   
                                                         test='ttest',                                            
                                                         positive=True),
                                    response_significance_threshold=5e-2)
FIGS = generate_comparison_figs(SUMMARY, 'WT', 'GluN1',
                                color1='k', color2='tab:blue')

# %%
# most strict
SUMMARY = compute_summary_responses(stat_test_props=dict(interval_pre=[-1.5,0],                                   
                                                         interval_post=[1.,2.5],                                   
                                                         test='anova',                                            
                                                         positive=True),
                                    response_significance_threshold=1e-3)
FIGS = generate_comparison_figs(SUMMARY, 'WT', 'GluN1',
                                color1='k', color2='tab:blue')

# %% [markdown]
# # Visualizing some evoked response in single ROI

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
