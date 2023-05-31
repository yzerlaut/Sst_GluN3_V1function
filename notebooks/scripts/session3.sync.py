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

# %% [markdown]
# # Session 3: Analysis
#
# N.B. now one generates a pdf per recording for the different protocols:
#
# - For the luminosity and orientation selectivity protocol:
#     ```
#     python src/pdf_lum_with_tuning.py your-datafile-file.nwb
#     ```
# - For the surround suppression protocol (analysis included in `physion`)
#     ```
#     python physion/src/physion/analysis/protocols/size-tuning.py your-datafile-file.nwb
#     ```

# %%
# general python modules
import sys, os, pprint, pandas
import numpy as np
import matplotlib.pylab as plt

sys.path.append('../../src')
from analysis import * # with physion path
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles


# %%
DATASET = scan_folder_for_NWBfiles('/home/yann.zerlaut/CURATED/SST-GluN3KO-January-2023/', 
                                   verbose=True)

# %%
DATASET['files'] = [os.path.basename(f) for f in DATASET['files']]
pandas.DataFrame(DATASET)

# %% [markdown]
# # Dataset description

# %%
from physion.analysis.behavior import population_analysis
fig, ax = population_analysis(DATASET['files'],
                            running_speed_threshold=0.1,
                            ax=None)
#ge.save_on_desktop(fig, 'fig.svg')

# %% [markdown]
# # Visual stimulation protocols

# %% [markdown]
# ## Luminosity changes

# %%
# find the protocols with the drifting gratings
FILES = []
for i, protocols in enumerate(DATASET['protocols']):
    if (('luminosity' in protocols) or ('black-5min' in protocols)) and (DATASET['dates'][i]!='2023_01_19'):
        # 19th Ketamine exp
        FILES.append(DATASET['files'][i])

# %%
from scipy.stats import skew

def get_luminosity_summary(data):                                                                                                  
                                                                                                                                                               
    summary = {}                                                                                                                                               
    for lum in ['dark' ,'grey', 'black']:                                                                                                                      
        summary[lum] = {}                                                                                                                                      
        for key in ['mean', 'std', 'skewness']:                                                                                                                
            summary[lum][key] = []                                                                                                                             
                                                                                                                                                               
    if 'BlankFirst' in data.metadata['protocol']:                                                                                                              
        tstarts = data.nwbfile.stimulus['time_start_realigned'].data[:3]                                                                                       
        tstops = data.nwbfile.stimulus['time_stop_realigned'].data[:3]                                                                                         
        values = ['dark' ,'black', 'grey']                                                                                                                     
    elif 'BlankLast' in data.metadata['protocol']:                                                                                                             
        tstarts = data.nwbfile.stimulus['time_start_realigned'].data[-3:]                                                                                      
        tstops = data.nwbfile.stimulus['time_stop_realigned'].data[-3:]                                                                                        
        values = ['black' ,'grey', 'dark']                                                                                                                     
    else:                                                                                                                                                      
        tstarts = data.nwbfile.stimulus['time_start_realigned'].data[:3]                                                                                      
        tstops = data.nwbfile.stimulus['time_stop_realigned'].data[:3]                                                                                        
        values = ['black' ,'grey', 'dark']                                                                                                                     
                                                                                                                                                               
    for tstart, tstop, lum in zip(tstarts, tstops, values):                                                                                                    
        t_cond = (data.t_dFoF>tstart) & (data.t_dFoF<tstop)                                                                                                    
        for roi in range(data.nROIs):                                                                                                                          
            for key, func in zip(['mean', 'std', 'skewness'], [np.mean, np.std, skew]):                                                                        
                summary[lum][key].append(func(data.dFoF[roi,t_cond]))                                                                                          
                                                                                                                                                               
    return summary                                                                                                                                             



# %%
SUMMARY = []
for f in FILES:
    data = Data(f, verbose=False)
    data.build_dFoF(verbose=False)
    SUMMARY.append(get_luminosity_summary(data))


# %%
def generate_lum_response_fig(SUMMARY):

    fig, AX = pt.plt.subplots(1, 3, figsize=(7,2.4))
    fig.subplots_adjust(wspace=0.6, bottom=.4)

    values = ['dark' ,'black', 'grey']

    for i, key in enumerate(['mean', 'std', 'skewness']):

        for summary in SUMMARY:
            AX[i].plot(np.arange(1,4),
                [np.mean(summary[lum][key]) for lum in values], 'k-', lw=0.2)
        data = [[np.mean(summary[lum][key]) for summary in SUMMARY] for lum in values]

        pt.violin(data, ax=AX[i],
                  labels=values)
        
        AX[i].set_title('$\Delta$F/F %s  (N=%i sessions)' % (key, len(SUMMARY)))
        
    AX[0].set_ylabel('$\Delta$F/F')
    AX[1].set_ylabel('$\Delta$F/F')

    return fig
                                                                                                                                                               
fig = generate_lum_response_fig(SUMMARY)

# %% [markdown]
# ## Orientation tuning

# %%
# find the protocols with the drifting gratings
FILES = []
for i, protocols in enumerate(DATASET['protocols']):
    if 'ff-gratings-8orientation-2contrasts-10repeats' in protocols:
        FILES.append(DATASET['files'][i])

# %%
FILES = ['/home/yann.zerlaut/CURATED/SST-Glun3KO-January-2023/2023_01_12-18-47-40.nwb',
         '/home/yann.zerlaut/CURATED/SST-Glun3KO-January-2023/2023_01_12-21-01-10.nwb',
         '/home/yann.zerlaut/CURATED/SST-Glun3KO-January-2023/2023_01_12-21-51-21.nwb']

# %%
stat_test_props = dict(interval_pre=[-1.5,0],                                                                                                                  
                       interval_post=[1,2.5],
                       test='ttest',
                       positive=True)
                                                                                                                                                               
response_significance_threshold = 0.01                                                                                                                         

RESPONSES, FRAC_RESP = [], []
for f in FILES:
    data = Data(f, verbose=False)
    responses, frac_resp, shifted_angle = compute_tuning_response_per_cells(data, verbose=False)
    RESPONSES.append(responses)
    FRAC_RESP.append(frac_resp)

# %%
from physion.utils import plot_tools as pt

def plot_tunning_summary(shifted_angle, frac_resp, responses):
    """
    """
    fig, AX = pt.plt.subplots(1, 3, figsize=(6,1))
    pt.plt.subplots_adjust(wspace=0.8)

    RESPONSES = [np.mean(responses, axis=0) for responses in responses]
    
    # raw
    pt.plot(shifted_angle, np.mean(RESPONSES, axis=0),
            sy=np.std(RESPONSES, axis=0), ax=AX[0])
    
    
    AX[0].set_ylabel('$\Delta$F/F')
    AX[0].set_title('raw resp.')

    for ax in AX[:2]:
        ax.set_xlabel('angle ($^o$)')
        ax.annotate('N=%i sessions'%len(responses), (1,1), fontsize=6,
                    va='top', ha='right', xycoords='axes fraction')

    # peak normalized
    N_RESP = [resp/resp[1] for resp in RESPONSES]
    pt.plot(shifted_angle, np.mean(N_RESP, axis=0),
            sy=np.std(N_RESP, axis=0), ax=AX[1])

    AX[1].set_yticks([0, 0.5, 1])
    AX[1].set_ylabel('n. $\Delta$F/F')
    AX[1].set_title('peak normalized')

    pt.pie([np.mean(frac_resp), 1-np.mean(frac_resp)],
           pie_labels=['%.1f%%' % (100.*np.mean(frac_resp)),
                       '     %.1f%%' % (100.*(1-np.mean(frac_resp)))],
           COLORS=[pt.plt.cm.tab10(2), pt.plt.cm.tab10(1)], ax=AX[2])
    
    NTOTs = [len(resp)/fr for fr, resp in zip(frac_resp, responses)]
    Ns = [len(resp) for resp in responses]
    AX[2].annotate('responsive ROIS :\nN=%i sessions\n n= %i$\pm$%i / %i$\pm$%i ROIs  ' % (len(responses),
                                                                        np.mean(Ns), np.std(Ns),
                                                                        np.mean(NTOTs), np.std(NTOTs)),
                   (0.5, 0), va='top', ha='center',
                   xycoords='axes fraction')
    #ge.save_on_desktop(fig, 'fig.png', dpi=300)
    return fig, AX
    
fig, AX = plot_tunning_summary(shifted_angle, FRAC_RESP, RESPONSES)


# %% [markdown]
# ## Looking at individual recordings

# %%
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



# %%
from physion.analysis.process_NWB import EpisodeData
from physion.utils import plot_tools as pt
from physion.dataviz.episodes.trial_average import plot_trial_average


stat_test_props = dict(interval_pre=[-1.5,0], 
                       interval_post=[0.5,2],
                       test='ttest',
                       positive=True)

response_significance_threshold = 0.01

def cell_tuning_example_fig(data,
                            Nsamples = 10, # how many cells we show
                            seed=10):
    np.random.seed(seed)
    
    EPISODES = EpisodeData(data,
                           quantities=['dFoF'],
                           protocol_id=protocol_id,
                           verbose=True)
    
    fig, AX = pt.plt.subplots(Nsamples, len(EPISODES.varied_parameters['angle']), 
                          figsize=(7,7))
    plt.subplots_adjust(right=0.75, left=0.1, top=0.97, bottom=0.05, wspace=0.1, hspace=0.8)
    
    for Ax in AX:
        for ax in Ax:
            ax.axis('off')

    for i, r in enumerate(np.random.choice(np.arange(data.nROIs), 
                                           min([Nsamples, data.nROIs]), replace=False)):

        # SHOW trial-average
        plot_trial_average(EPISODES,
                           column_key='angle',
                           color_key='contrast',
                           quantity='dFoF',
                           ybar=1., ybarlabel='1dF/F',
                           xbar=1., xbarlabel='1s',
                           roiIndex=r,
                           color=['khaki', 'k'],
                           with_stat_test=True,
                           AX=[AX[i]], no_set=False)
        AX[i][0].annotate('roi #%i  ' % (r+1), (0,0), ha='right', xycoords='axes fraction')

        # SHOW summary angle dependence
        inset = pt.inset(AX[i][-1], (2.2, 0.2, 1.2, 0.8))

        angles, y, sy, responsive_angles = [], [], [], []
        responsive = False

        for a, angle in enumerate(EPISODES.varied_parameters['angle']):

            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=\
                                            EPISODES.find_episode_cond(['angle', 'contrast'], [a,1]),
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
        inset.set_ylabel('$\delta$dF/F     ')
        if i==(Nsamples-1):
            inset.set_xlabel('angle ($^{o}$)')

        SI = selectivity_index(angles, y)
        inset.annotate('SI=%.2f ' % SI, (0, 1), ha='right', weight='bold', fontsize=8,
                       color=('k' if responsive else 'lightgray'), xycoords='axes fraction')
        inset.annotate(('responsive' if responsive else 'unresponsive'), (1, 1), ha='right',
                        weight='bold', fontsize=6, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)),
                        xycoords='axes fraction')
        
    return fig

fig = cell_tuning_example_fig(FILES[-1])
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'), dpi=150)

