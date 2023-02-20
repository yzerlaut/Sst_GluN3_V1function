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
print('keep this line')

# %%
# general python modules
import sys, os, pprint
import numpy as np
import matplotlib.pylab as plt

# *_-= physion =-_*
physion_folder = os.path.join(os.path.expanduser('~'), 'work', 'SST_GluN3_V1function', 'physion') # UPDATE to your folder location
# -- physion core
sys.path.append(os.path.join(physion_folder, 'physion'))
from analysis.read_NWB import Data, scan_folder_for_NWBfiles
from dataviz.show_data import MultimodalData, EpisodeResponse
# -- physion data visualization
sys.path.append(os.path.join(physion_folder, 'dataviz', 'datavyz'))
from datavyz import ge


# %%
FILES, DATES, SUBJECTS = scan_folder_for_NWBfiles('/home/yann.zerlaut/DATA/taddy_GluN3KO/', 
                                                  verbose=True)

FILES = FILES[np.argsort(DATES)]

# %% [markdown]
# # Dataset description

# %%
from analysis.behavior import population_analysis
fig, ax = ge.figure(figsize=(1.5,.7), top=2.5, left=0.2, right=1.1, bottom=0.4)
_, ax = population_analysis(FILES,
                            running_speed_threshold=0.1,
                            ax=ax)
ge.save_on_desktop(fig, 'fig.svg')

# %%
for i, f in enumerate(FILES):
    data = MultimodalData(FILES[i])
    print('session #%i' % (i+1), ' --> %s (%s)' % (f.split(os.path.sep)[-1], data.metadata['subject_ID']))
    print('  ', data.protocols)
    data.io.close()

# %% [markdown]
# ## FOV

# %%
session = 0

fig, ax = ge.figure(figsize=(2.6,6), 
                    left=0.2, right=0, 
                    top=.1, bottom=0.05)

data = MultimodalData(FILES[session])
data.show_CaImaging_FOV(key='meanImg', NL=3, cmap=ge.viridis,
                        ax=ax, roiIndex=None, with_roi_zoom=False)
ge.title(ax, 'meanImg     ---  '+data.metadata['notes'])
ge.annotate(ax, 'session #%i, %s ' % (session, data.metadata['subject_ID']), (0,.0), 
            rotation=90, size='x-small')
    
ge.save_on_desktop(fig, 'fig.png')

# %%
data = MultimodalData(FILES[0])
fig, AX = ge.figure(axes=(3,1), 
                    figsize=(2,3), left=0.1, right=0, top=.5, bottom=0.2, hspace=0.1, wspace=0.05)
for key, ax, nl in zip(['meanImg', 'meanImgE', 'max_proj'], AX, [2,1,2]):
    data.show_CaImaging_FOV(key=key, NL=nl, cmap=ge.viridis, #ge.get_linear_colormap('k', 'green'), 
                                ax=ax, roiIndices='all', with_roi_zoom=False)
    ge.annotate(AX[0], 'GCamp6s', (.98,.99), ha='right', va='top', color='w', size='small')
    ge.annotate(AX[0], 'n=%i ROIs ' % data.iscell.sum(), (1,.0), ha='right', color='w', size='small')
ge.save_on_desktop(fig, 'fig.png')

# %% [markdown]
# # Visual stimulation protocols

# %% [markdown]
# ## Spatial mapping

# %%
indices = [0,1,2] # indices of spatial mapping files
for i, f in enumerate(FILES[indices]):
    data = Data(FILES[i])
    print('session #%i' % (i+1), ' --> %s (%s)' % (f.split(os.path.sep)[-1], data.metadata['subject_ID']))
    print('  ', data.protocols)
    data.io.close()

# %%
index = 0
np.random.seed(10)
data = MultimodalData(FILES[index])
fig, ax = data.plot_raw_data(tlim=[1, 30], 
                  settings={'Photodiode':dict(fig_fraction=0.5, subsampling=10, color=ge.grey),
                            'Locomotion':dict(fig_fraction=2, subsampling=1, color=ge.blue),
                            'CaImagingRaster':dict(fig_fraction=3, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dFoF'),
                            'CaImaging':dict(fig_fraction=4, subsampling=1,
                                                   roiIndices=np.sort(np.random.choice(np.arange(data.iscell.sum()),10, 
                                                                               replace=False)),
                                                   subquantity='dFoF'),
                           'VisualStim':dict(fig_fraction=1)},
                   Tbar=1, figsize=(3.5,6));
ge.annotate(ax, 'session #%i, %s, %s ' % (index+1, data.metadata['subject_ID'], FILES[index].split('/')[-1]),
            (0.5,0), ha='center', size='small')
ge.save_on_desktop(fig, 'fig.png', dpi=300)

# %%
for index in indices:
    data = MultimodalData(FILES[index], verbose=False)
    fig, ax = data.plot_raw_data(tlim=[1, 200], 
                      settings={'Photodiode':dict(fig_fraction=0.5, subsampling=10, color=ge.grey),
                                'Locomotion':dict(fig_fraction=2, subsampling=1, color=ge.blue),
                                'CaImagingRaster':dict(fig_fraction=3, subsampling=1,
                                                       roiIndices='all',
                                                       normalization='per-line',
                                                       subquantity='dFoF'),
                                'CaImaging':dict(fig_fraction=4, subsampling=1,
                                                       roiIndices=np.sort(np.random.choice(np.arange(data.iscell.sum()),5, 
                                                                                   replace=False)),
                                                       subquantity='dFoF'),
                               'VisualStim':dict(fig_fraction=1)},
                       Tbar=10, figsize=(3.5,6));
    ge.annotate(ax, 'session #%i, %s, %s --- %s' % (index+1, data.metadata['subject_ID'], 
                                              FILES[index].split('/')[-1], data.metadata['notes']),
                (0.5,0), ha='center', size='small')
    ge.save_on_desktop(fig, 'fig-%i.png' % index, dpi=300)

# %%
stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2],
                     test='ttest', positive=True)
response_significance_threshold = 0.01

for index in indices:
    episodes = EpisodeResponse(FILES[index],
                               protocol_id=data.get_protocol_id('spatial-mapping'),
                               quantities=['dFoF'],
                               prestim_duration=2,
                               verbose=True)

    responsive_cells = []
    for i, r in enumerate(np.arange(episodes.data.iscell.sum())):

        cell_resp = episodes.compute_summary_data(stat_test_props,
                                                  response_significance_threshold=response_significance_threshold,
                                                  response_args=dict(quantity='dFoF', roiIndex=r))

        if np.sum(cell_resp['significant'])>0:
            responsive_cells.append(r) # we just need one responsive angle to turn this True

    fig, _ = episodes.plot_trial_average(column_key='x-center', row_key='y-center',
                                roiIndices=responsive_cells,
                                ybar=.1, ybarlabel='0.1dF/F',
                                xbar=1., xbarlabel='1s',
                                with_screen_inset=True,
                                with_std_over_rois=True,
                                #color_key='contrast',
                                with_annotation=True,
                                with_std=False);
    ge.annotate(fig, 'session #%i, %s, %s --- %s' % (index+1, episodes.data.metadata['subject_ID'], 
                                                  FILES[index].split('/')[-1], episodes.data.metadata['notes']),
                (0.5,0), ha='center', size='small')
    ge.save_on_desktop(fig, 'fig-%i.png' % index, dpi=300)

# %% [markdown]
# ## Luminosity changes

# %%
indices = [5,6,7,8] # indices of spatial mapping files
for index in indices:
    data = Data(FILES[index])
    print('session #%i'%(index+1),' --> %s (%s)' % (FILES[index].split(os.path.sep)[-1],data.metadata['subject_ID']))
    print('  ', data.protocols)
    data.io.close()

# %%
for index in [6,7]:
    np.random.seed(10)
    data = MultimodalData(FILES[index])
    fig, ax = data.plot_raw_data(tlim=[1, 800], 
                      settings={'Photodiode':dict(fig_fraction=0.5, subsampling=10, color=ge.grey),
                                'Locomotion':dict(fig_fraction=2, subsampling=1, color=ge.blue),
                                'FaceMotion':dict(fig_fraction=2, subsampling=1, color=ge.purple),
                                'Pupil':dict(fig_fraction=2, subsampling=1, color=ge.red),
                                'CaImagingRaster':dict(fig_fraction=3, subsampling=1,
                                                       roiIndices='all',
                                                       normalization='per-line',
                                                       subquantity='dFoF'),
                                'CaImaging':dict(fig_fraction=4, subsampling=1,
                                                       roiIndices=np.sort(np.random.choice(np.arange(data.iscell.sum()),5, 
                                                                                   replace=False)),
                                                       subquantity='dFoF'),
                               'VisualStim':dict(fig_fraction=1)},
                       Tbar=1, figsize=(3.5,6));
    ge.annotate(ax, 'session #%i, %s, %s -- %s' % (index+1, data.metadata['subject_ID'], 
                                              FILES[index].split('/')[-1], data.metadata['notes']),
                (0.5,0), ha='center', size='small')
    ge.save_on_desktop(fig, 'fig-%i.png' % index, dpi=300)

# %%
for index in [5,8]:
    np.random.seed(10)
    data = MultimodalData(FILES[index])
    fig, ax = data.plot_raw_data(tlim=[900,data.tlim[-1]], 
                      settings={'Photodiode':dict(fig_fraction=0.5, subsampling=10, color=ge.grey),
                                'Locomotion':dict(fig_fraction=2, subsampling=1, color=ge.blue),
                                'FaceMotion':dict(fig_fraction=2, subsampling=1, color=ge.purple),
                                'Pupil':dict(fig_fraction=2, subsampling=1, color=ge.red),
                                'CaImagingRaster':dict(fig_fraction=3, subsampling=1,
                                                       roiIndices='all',
                                                       normalization='per-line',
                                                       subquantity='dFoF'),
                                'CaImaging':dict(fig_fraction=4, subsampling=1,
                                                       roiIndices=np.sort(np.random.choice(np.arange(data.iscell.sum()),5, 
                                                                                   replace=False)),
                                                       subquantity='dFoF'),
                               'VisualStim':dict(fig_fraction=1)},
                       Tbar=1, figsize=(3.5,6));
    ge.annotate(ax, 'session #%i, %s, %s -- %s' % (index+1, data.metadata['subject_ID'], 
                                              FILES[index].split('/')[-1], data.metadata['notes']),
                (0.5,0), ha='center', size='small')
    ge.save_on_desktop(fig, 'fig-%i.png' % index, dpi=300)

# %% [markdown]
# ### analysis

# %%
summary = {'dark':[],
           'grey':[],
           'black':[]}

for index in [6,7]:
    data = Data(FILES[index])
    data.build_dFoF()
    for tstart, tstop, label in zip(data.nwbfile.stimulus['time_start_realigned'].data[:3],
                                    data.nwbfile.stimulus['time_stop_realigned'].data[:3],
                                    ['dark', 'black', 'grey']):
        t_cond = (data.t_dFoF>tstart) & (data.t_dFoF<tstop)
        for roi in range(data.nROIs):
            summary[label].append(data.dFoF[roi, t_cond].mean())
            
for index in [5,8]:
    data = Data(FILES[index])
    data.build_dFoF()
    for tstart, tstop, label in zip(data.nwbfile.stimulus['time_start_realigned'].data[-3:],
                                    data.nwbfile.stimulus['time_stop_realigned'].data[-3:],
                                    ['dark', 'black', 'grey']):
        t_cond = (data.t_dFoF>tstart) & (data.t_dFoF<tstop)
        for roi in range(data.nROIs):
            summary[label].append(data.dFoF[roi, t_cond].mean())

# %%
fig, ax = ge.figure()
ge.bar([np.mean(summary['dark']), np.mean(summary['black']), np.mean(summary['grey'])],
       [np.std(summary['dark']), np.std(summary['black']), np.std(summary['grey'])],
       COLORS=['k', 'dimgrey', 'lightgrey'], ax=ax, ylabel='mean $\Delta$F/F     ',
       axes_args={'xticks':range(3),
                  'xticks_rotation':60,
                 'xticks_labels':['dark', 'black', 'grey']})
ge.save_on_desktop(fig, 'fig.png', dpi=300)

# %% [markdown]
# ## Orientation tuning

# %%
indices = np.arange(3,9) # indices of spatial mapping files
for index in indices:
    data = Data(FILES[index])
    print('session #%i'%(index+1),' --> %s (%s)' % (FILES[index].split(os.path.sep)[-1],data.metadata['subject_ID']))
    print('  ', data.protocols)
    data.io.close()


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
Nsamples = 10

stat_test_props=dict(interval_pre=[-1.5,0], interval_post=[0.5,2],
                     test='ttest', positive=True)
response_significance_threshold = 0.01

np.random.seed(10)

for index, protocol_id in zip(indices, [0,0,0,2,2,0]):
    
    EPISODES = EpisodeResponse(FILES[index],
                               quantities=['dFoF'],
                               protocol_id=protocol_id,
                               verbose=True)

    fig, AX = ge.figure((len(EPISODES.varied_parameters['angle']), Nsamples),
                        figsize=(.8,.9), right=15)


    for i, r in enumerate(np.random.choice(np.arange(EPISODES.data.iscell.sum()), 
                                           Nsamples, replace=False)):

        # SHOW trial-average
        EPISODES.plot_trial_average(column_key='angle',
                                    quantity='dFoF',
                                    ybar=1., ybarlabel='1dF/F',
                                    xbar=1., xbarlabel='1s',
                                    roiIndex=r,
                                    AX=[AX[i]], no_set=False)
        ge.annotate(AX[i][0], 'roi #%i  ' % (r+1), (0,0), ha='right')

        # SHOW summary angle dependence
        inset = ge.inset(AX[i][-1], (2, 0.2, 1.2, 0.8))

        angles, y, sy, responsive_angles = [], [], [], []
        responsive = False

        for a, angle in enumerate(EPISODES.varied_parameters['angle']):

            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a),
                                                            response_args=dict(quantity='dFoF', roiIndex=r),
                                                            **stat_test_props)

            angles.append(angle)
            y.append(np.mean(stats.y-stats.x))    # means "post"
            sy.append(np.std(stats.y-stats.x))    # std "post"

            if stats.significant(threshold=response_significance_threshold):
                responsive = True
                responsive_angles.append(angle)

        ge.plot(angles, np.array(y), sy=np.array(sy), ax=inset,
                axes_args=dict(ylabel='<$\Delta$ dF/F>         ', xlabel='angle ($^{o}$)',
                               size='small'), 
                m='o', ms=2, lw=1)

        SI = selectivity_index(angles, y)
        ge.annotate(inset, 'SI=%.2f ' % SI, (0, 1), ha='right', weight='bold', fontsize=8,
                    color=('k' if responsive else 'lightgray'))
        ge.annotate(inset, ('responsive' if responsive else 'unresponsive'), (1, 1), ha='right',
                    weight='bold', fontsize=6, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))
        
        ge.save_on_desktop(fig, 'fig-%i.png' % index, dpi=300)


# %%
Nsamples = 10

stat_test_props=dict(interval_pre=[-1.5,0], interval_post=[0.5,2],
                     test='ttest', positive=True)
response_significance_threshold = 0.01

np.random.seed(10)

RESPONSES = []
Ntot, Nresp = 0, 0

#for index, protocol_id in zip([3,7], [0,2]):
for index, protocol_id in zip(indices, [0,0,0,2,2,0]):
    
    EPISODES = EpisodeResponse(FILES[index],
                               quantities=['dFoF'],
                               protocol_id=protocol_id,
                               verbose=True)
                               
    shifted_angle = EPISODES.varied_parameters['angle']-EPISODES.varied_parameters['angle'][1]
    Ntot += EPISODES.data.iscell.sum()

    for roi in np.arange(EPISODES.data.iscell.sum()):

        cell_resp = EPISODES.compute_summary_data(response_significance_threshold=response_significance_threshold,
                                                  response_args=dict(quantity='dFoF', roiIndex=roi),
                                                  stat_test_props=stat_test_props)

        condition = np.ones(len(cell_resp['angle']), dtype=bool) # no condition

        if np.sum(cell_resp['significant'][condition]):

            ipref = np.argmax(cell_resp['value'][condition])
            prefered_angle = cell_resp['angle'][condition][ipref]

            RESPONSES.append(np.zeros(len(shifted_angle)))
            Nresp+=1
            
            for a, angle in enumerate(cell_resp['angle'][condition]):

                new_angle = shift_orientation_according_to_pref(angle, 
                                                                pref_angle=prefered_angle, 
                                                                start_angle=shifted_angle[0], 
                                                                angle_range=180)
                iangle = np.argwhere(shifted_angle==new_angle)[0][0]
                RESPONSES[-1][iangle] = cell_resp['value'][a] 
                               
                               
                

# %%
fig, AX = ge.figure(axes=(3,1), wspace=2)
# raw
ge.scatter(shifted_angle, np.mean(RESPONSES, axis=0), 
           sy=np.std(RESPONSES, axis=0), ms=4, lw=1, ax=AX[0],
           xlabel='angle ($^o$)', ylabel='$\Delta$F/F', title='raw resp.')
# peak normalized
N_RESP = [resp/np.max(resp) for resp in RESPONSES]
ge.scatter(shifted_angle, np.mean(N_RESP, axis=0), sy=np.std(N_RESP, axis=0),
           ms=4, lw=1, ax=AX[1], axes_args={'yticks':[0,1]},
           xlabel='angle ($^o$)', ylabel='n. $\Delta$F/F', title='peak normalized')

Nresp = len(RESPONSES)
ge.pie([resp, non_resp], 
       ext_labels=['responsive ROIS :   \nn=%i / %i        ' % (Nresp, Ntot), ''],
       pie_labels=['%.1f%%' % (100.*Nresp/Ntot), '%.1f%%' % (100.*(1-Nresp/Ntot))],
       COLORS=[ge.green, ge.orange], ax=AX[2])
ge.save_on_desktop(fig, 'fig.png', dpi=300)

# %% [markdown]
# ## Behavioral modulation of orientation tuning

# %%
np.random.seed(10)
for index in [8]:
    data = MultimodalData(FILES[index])
    fig, ax = data.plot_raw_data(tlim=[650, 910], 
                      settings={'Photodiode':dict(fig_fraction=0.5, subsampling=10, color=ge.grey),
                                'Locomotion':dict(fig_fraction=2, subsampling=1, color=ge.blue),
                                'FaceMotion':dict(fig_fraction=2, subsampling=1, color=ge.purple),
                                'Pupil':dict(fig_fraction=2, subsampling=1, color=ge.red),
                                'CaImagingRaster':dict(fig_fraction=3, subsampling=1,
                                                       roiIndices='all',
                                                       normalization='per-line',
                                                       subquantity='dFoF'),
                                'CaImaging':dict(fig_fraction=4, subsampling=1,
                                                       roiIndices=np.sort(np.random.choice(np.arange(data.iscell.sum()),5, 
                                                                                   replace=False)),
                                                       subquantity='dFoF'),
                               'VisualStim':dict(fig_fraction=1)},
                       Tbar=1, figsize=(3.5,6));
    ge.annotate(ax, 'session #%i, %s, %s -- %s' % (index+1, data.metadata['subject_ID'], 
                                              FILES[index].split('/')[-1], data.metadata['notes']),
                (0.5,0), ha='center', size='small')
    ge.save_on_desktop(fig, 'fig-%i.png' % index, dpi=300)


# %%

def behav_split_fig(EPISODES, threshold=0.1, ax=None):
    """
    threshold = 0.2 # cm/s
    """
    if ax is None:
        fig, ax = ge.figure(figsize=(1.2,1.5))
    else:
        fig = None

    running = np.mean(EPISODES.RunningSpeed, axis=1)>threshold

    ge.scatter(np.mean(EPISODES.pupilSize, axis=1)[running], 
               np.mean(EPISODES.RunningSpeed, axis=1)[running],
               ax=ax, no_set=True, color=ge.blue, ms=5)
    ge.scatter(np.mean(EPISODES.pupilSize, axis=1)[~running], 
               np.mean(EPISODES.RunningSpeed, axis=1)[~running],
               ax=ax, no_set=True, color=ge.orange, ms=5)
    ge.set_plot(ax, xlabel='pupil size (mm)', ylabel='run. speed (cm/s)')
    ax.plot(ax.get_xlim(), threshold*np.ones(2), 'k--', lw=0.5)
    ge.annotate(ax, 'n=%i' % np.sum(running), (0,1), va='top', color=ge.blue)
    ge.annotate(ax, '\nn=%i' % np.sum(~running), (0,1), va='top', color=ge.orange)
    return fig, ax


# %%

fig, AX = ge.figure(axes=(4,1), wspace=2)

for ax, index, protocol_id in zip(AX, np.arange(5, 9), [0,2,2,0]):
    
    EPISODES = EpisodeResponse(FILES[index],
                               quantities=['dFoF', 'Pupil', 'Running-Speed'],
                               protocol_id=protocol_id,
                               verbose=False)
    Nep = EPISODES.dFoF.shape[0]
    behav_split_fig(EPISODES, ax=ax)
    ge.title(ax, FILES[index].split('/')[-1])
ge.save_on_desktop(fig, 'fig.png')

# %%
Nsamples = 10 #EPISODES.data.nROIs
np.random.seed(10)
index = 8

EPISODES = EpisodeResponse(FILES[index],
                           quantities=['dFoF', 'Pupil', 'Running-Speed'],
                           protocol_id=0,
                           verbose=True)
Nep = EPISODES.dFoF.shape[0]

threshold = 0.1 # cm/s
running = np.mean(EPISODES.RunningSpeed, axis=1)>threshold


fig, AX = ge.figure((len(EPISODES.varied_parameters['angle']), Nsamples),
                    figsize=(.8,.9), right=10)
ge.annotate(fig, FILES[index].split('/')[-1], (0.5,0), ha='center')
            
stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2],
                     test='ttest', positive=True, verbose=False)
response_significance_threshold = 0.01


responsive_cells = []
for i, r in enumerate(np.arange(EPISODES.data.iscell.sum())):
    # decide whether a cell is visually responsive on all episodes
    responsive = False
    for a, angle in enumerate(EPISODES.varied_parameters['angle']):
        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a),
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)
        if r not in responsive_cells and \
                stats.significant(threshold=response_significance_threshold):
            responsive_cells.append(r) # we just need one responsive angle to turn this True
    
    
for i, r in enumerate(np.random.choice(np.arange(EPISODES.data.iscell.sum()), 
                                       Nsamples, replace=False)):
    
    # STILL trial-average
    EPISODES.plot_trial_average(column_key='angle',
                                condition=EPISODES.find_episode_cond(key='contrast', value=1.) & ~running,
                                quantity='dFoF',
                                ybar=1., ybarlabel='1dF/F',
                                xbar=1., xbarlabel='1s',
                                roiIndex=r,
                                color=ge.orange,
                                AX=[AX[i]], no_set=True, verbose=False)
    
    # RUNNING trial-average
    EPISODES.plot_trial_average(column_key='angle',
                                condition=EPISODES.find_episode_cond(key='contrast', value=1.) & running,
                                quantity='dFoF',
                                ybar=1., ybarlabel='1dF/F',
                                xbar=1., xbarlabel='1s',
                                roiIndex=r,
                                color=ge.blue,
                                AX=[AX[i]], no_set=False,verbose=False)


    ge.annotate(AX[i][0], 'roi #%i  ' % (r+1), (0,0), ha='right')
    
    # SHOW summary angle dependence
    inset = ge.inset(AX[i][-1], (2, 0.2, 1.2, 0.8))
    
    ########################################################################
    ###### RUNNING 
    ########################################################################
    
    angles, y, sy = [], [], []
    
    for a, angle in enumerate(EPISODES.varied_parameters['angle']):

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a) & running,
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)

        if stats.r!=0:
            angles.append(angle)
            y.append(np.mean(stats.y-stats.x))    
            sy.append(np.std(stats.y-stats.x))  
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=inset,
            m='o', ms=2, lw=1, color=ge.blue, no_set=True)
    

    ########################################################################
    ###### STILL
    ########################################################################

    angles, y, sy = [], [], []

    for a, angle in enumerate(EPISODES.varied_parameters['angle']):

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a) & ~running,
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)
        
        if stats.r!=0:
            angles.append(angle)
            y.append(np.mean(stats.y-stats.x))    # means "delta"
            sy.append(np.std(stats.y-stats.x))    # std "delta"
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=inset,
            m='o', ms=2, lw=1, color=ge.orange, no_set=True)


    ge.set_plot(inset, 
                ylabel='< $\Delta$ dF/F>  ', xlabel='angle ($^{o}$)',
                #xticks=angles, 
                size='small')
ge.save_on_desktop(fig, 'fig.png')

# %% [markdown]
# ### summary analysis

# %%
stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2],
                     test='ttest', positive=True)
response_significance_threshold = 0.01


fig, AX = ge.figure(axes=(4,1), wspace=2)

for ax, index, protocol_id in zip(AX, np.arange(5, 9), [0,2,2,0]):

    EPISODES = EpisodeResponse(FILES[index],
                               quantities=['dFoF', 'Pupil', 'Running-Speed'],
                               protocol_id=protocol_id,
                               verbose=True)
    Nep = EPISODES.dFoF.shape[0]

    threshold = 0.1 # cm/s
    running = np.mean(EPISODES.RunningSpeed, axis=1)>threshold

    responsive_cells = []
    for i, r in enumerate(np.arange(EPISODES.data.iscell.sum())):
        # decide whether a cell is visually responsive on all episodes
        responsive = False
        for a, angle in enumerate(EPISODES.varied_parameters['angle']):
            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a),
                                                            response_args=dict(quantity='dFoF', roiIndex=r),
                                                            **stat_test_props)
            if r not in responsive_cells and \
                    stats.significant(threshold=response_significance_threshold):
                responsive_cells.append(r) # we just need one responsive angle to turn this True


    RUN_RESPONSES, STILL_RESPONSES, RESPONSES = [], [], []
    shifted_angle = EPISODES.varied_parameters['angle']-EPISODES.varied_parameters['angle'][1]

    for roi in responsive_cells:

        # ALL    
        # ----------------------------------------
        cell_resp = EPISODES.compute_summary_data(stat_test_props,
                                                  response_significance_threshold=response_significance_threshold,
                                                  response_args=dict(quantity='dFoF', roiIndex=roi))

        condition = np.ones(len(cell_resp['angle']), dtype=bool) # no condition
        # condition = (cell_resp['contrast']==1.) # if specific condition

        ipref = np.argmax(cell_resp['value'][condition])
        prefered_angle = cell_resp['angle'][condition][ipref]

        RESPONSES.append(np.zeros(len(shifted_angle)))

        for a, angle in enumerate(cell_resp['angle'][condition]):

            new_angle = shift_orientation_according_to_pref(angle, 
                                                            pref_angle=prefered_angle, 
                                                            start_angle=shifted_angle[0], 
                                                            angle_range=180)
            iangle = np.argwhere(shifted_angle==new_angle)[0][0]
            RESPONSES[-1][iangle] = cell_resp['value'][a]


        # RUNNING
        # ----------------------------------------
        cell_resp = EPISODES.compute_summary_data(stat_test_props,
                                                  episode_cond=running,
                                                  response_significance_threshold=response_significance_threshold,
                                                  response_args=dict(quantity='dFoF', roiIndex=roi),
                                                  verbose=False)

        RUN_RESPONSES.append(np.zeros(len(shifted_angle)))

        for a, angle in enumerate(cell_resp['angle'][condition]):

            new_angle = shift_orientation_according_to_pref(angle, 
                                                            pref_angle=prefered_angle, 
                                                            start_angle=shifted_angle[0], 
                                                            angle_range=180)
            iangle = np.argwhere(shifted_angle==new_angle)[0][0]
            RUN_RESPONSES[-1][iangle] = cell_resp['value'][a]

        # STILL
        # ----------------------------------------
        cell_resp = EPISODES.compute_summary_data(stat_test_props,
                                                  episode_cond=~running,
                                                  response_significance_threshold=response_significance_threshold,
                                                  response_args=dict(quantity='dFoF', roiIndex=roi),
                                                  verbose=False)

        STILL_RESPONSES.append(np.zeros(len(shifted_angle)))

        for a, angle in enumerate(cell_resp['angle'][condition]):

            new_angle = shift_orientation_according_to_pref(angle, 
                                                            pref_angle=prefered_angle, 
                                                            start_angle=shifted_angle[0], 
                                                            angle_range=180)
            iangle = np.argwhere(shifted_angle==new_angle)[0][0]
            STILL_RESPONSES[-1][iangle] = cell_resp['value'][a]
            
    plot_single_session(shifted_angle, RUN_RESPONSES, STILL_RESPONSES, RESPONSES, ax=ax)
    ge.title(ax, FILES[index].split('/')[-1])

ge.save_on_desktop(fig, 'fig.png')

# %%
ge.save_on_desktop(fig, 'fig.png')


# %%
def plot_single_session(shifted_angle, RUN_RESPONSES, STILL_RESPONSES, RESPONSES,
                        ax=None):
    if ax is None:
        fig, ax = ge.figure()
        
    ge.scatter(shifted_angle, np.mean(RESPONSES, axis=0), 
               sy=np.std(RESPONSES, axis=0), 
               ms=2, lw=0.5, ax=ax,
               color=ge.grey, no_set=True)
    ge.scatter(shifted_angle, np.mean(STILL_RESPONSES, axis=0), 
               sy=np.std(STILL_RESPONSES, axis=0), 
               ms=3, lw=1, ax=ax,
               color=ge.orange, no_set=True)
    ge.scatter(shifted_angle, np.mean(RUN_RESPONSES, axis=0), 
               sy=np.std(RUN_RESPONSES, axis=0),
               ms=3, lw=1, ax=ax,
               xlabel='angle from pref. ($^o$)', ylabel='raw $\Delta$F/F', color=ge.blue)
    
plot_single_session(shifted_angle, RUN_RESPONSES, STILL_RESPONSES, RESPONSES)


# %% [markdown]
# # Spontaneous activity

# %%
index =0
np.random.seed(10)
data = MultimodalData(FILES[index])
fig, ax = data.plot_raw_data(tlim=[350, 550], 
                  settings={'Locomotion':dict(fig_fraction=2, subsampling=1, color=ge.blue),
                            #'FaceMotion':dict(fig_fraction=2, subsampling=1, color=ge.purple),
                            #'Pupil':dict(fig_fraction=2, subsampling=1, color=ge.red),
                            'CaImagingRaster':dict(fig_fraction=3, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dFoF'),
                            'CaImaging':dict(fig_fraction=4, subsampling=1,
                                                   roiIndices=np.sort(np.random.choice(np.arange(data.iscell.sum()),5, 
                                                                               replace=False)),
                                                   subquantity='dFoF'),
                           'VisualStim':dict(fig_fraction=1)},
                   Tbar=2, figsize=(3.5,6));
ge.annotate(ax, 'session #%i, %s, %s ' % (index+1, data.metadata['subject_ID'], FILES[index].split('/')[-1]),
            (0.5,0), ha='center', size='small')
ge.save_on_desktop(fig, 'fig.png', dpi=300)

# %%
index = 5
data = MultimodalData(FILES[index])
#fig, ax = ge.figure(axes=(1,1), figsize=(2,2), left=0.2, bottom=0.04, top=0.1)
fig, ax = data.plot_raw_data(tlim=[10, 2510], 
                  settings={'Locomotion':dict(fig_fraction=2, subsampling=30, color=ge.blue),
                            'FaceMotion':dict(fig_fraction=2, subsampling=30, color=ge.purple),
                            'Pupil':dict(fig_fraction=2, subsampling=10, color=ge.red),
                            'CaImagingRaster':dict(fig_fraction=5, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   quantity='CaImaging', subquantity='Fluorescence')},
                   Tbar=120, figsize=(3.5,5));
ge.annotate(ax, 'session #%i, %s, %s ' % (index+1, data.metadata['subject_ID'], FILES[index].split('/')[-1]),
            (0.5,1), ha='center', size='small')
ge.save_on_desktop(fig, 'fig.png', dpi=300)

# %%
index =5
data = MultimodalData(FILES[index])
#fig, ax = ge.figure(axes=(1,1), figsize=(2,2), left=0.2, bottom=0.04, top=0.1)
fig, ax = data.plot_raw_data(tlim=[510, 1010], 
                  settings={'Locomotion':dict(fig_fraction=2, subsampling=1, color=ge.blue),
                            'FaceMotion':dict(fig_fraction=2, subsampling=1, color=ge.purple),
                            'Pupil':dict(fig_fraction=2, subsampling=1, color=ge.red),
                            'CaImagingRaster':dict(fig_fraction=5, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   quantity='CaImaging', subquantity='Fluorescence')},
                   Tbar=10, figsize=(3.5,5));
ge.annotate(ax, 'session #%i, %s, %s ' % (index+1, data.metadata['subject_ID'], FILES[index].split('/')[-1]),
            (0.5,1), ha='center', size='small')
ge.save_on_desktop(fig, 'fig.png', dpi=300)

# %% [markdown]
# # Visually-evoked activity

# %%


stat_test_props= dict(interval_pre=[-1,0], interval_post=[1,2],
                      test='ttest', positive=True)

index, iprotocol = 5, 0

data = MultimodalData(FILES[index])

EPISODES = EpisodeResponse(FILES[index],
                           protocol_id=0,
                           verbose=True)
Nexample = 8
fig, AX = ge.figure(axes=(8,Nexample), figsize=(.8,.9), right=10, bottom=0.3, top=1.5,
                    reshape_axes=False, wspace=0.5, hspace=1.2)

for n, i in enumerate(np.random.choice(np.arange(data.iscell.sum()), Nexample)):
    
    EPISODES.plot_trial_average(roiIndex = n,
                                column_key='angle',
                                color_key='contrast',
                                          ybar=1., ybarlabel='1dF/F',
                                          xbar=1., xbarlabel='1s',
                                          fig=fig, AX=[AX[n]], no_set=False,
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=False)

    ge.annotate(AX[n][0], 'roi #%i\n' % (i+1), (0,-.2), ha='right', rotation=90, size='small', bold=True)

ge.annotate(fig, 'session #%i, %s, %s ' % (index+1, data.metadata['subject_ID'], FILES[index].split('/')[-1]),
            (0.5,1), ha='center', va='top', size='small', xycoords='figure fraction')

#ge.save_on_desktop(fig, 'fig.png', dpi=300)

# %%
stat_test_props= dict(interval_pre=[-1,0], interval_post=[1,2],
                      test='ttest', positive=True)

summary = EPISODES.compute_summary_data(stat_test_props,
                                        response_args={'quantity':'dFoF', 'roiIndex':5})

summary

# %% [markdown]
# ## modulation of visually-evoked responses by behavioral state

# %%
from physion.analysis.behavioral_modulation import tuning_modulation_fig, compute_behavior_mod_population_tuning, population_tuning_fig, compute_population_resp

options = dict(subquantity='d(F-0.7*Fneu)',
               dt_sampling=1, prestim_duration=3, 
               verbose=False)

# %%
FILES = ['/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_12/2021_10_12-15-34-26.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_12/2021_10_12-16-44-18.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_14/2021_10_14-15-34-45.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_20/2021_10_20-13-12-35.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_27/2021_10_27-16-42-16.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_27/2021_10_27-17-51-55.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_27/2021_10_27-16-05-01.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_14/2021_10_14-16-54-21.nwb', # PB HERE
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_20/2021_10_20-14-30-52.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_20/2021_10_20-16-22-39.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_22/2021_10_22-17-28-02.nwb',
         '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Batch2-Sept_2021/2021_10_22/2021_10_22-16-10-05.nwb']

# %%
from physion.analysis.behavioral_modulation import compute_DS_population_resp

options = dict(subquantity='d(F-0.7*Fneu)',
               dt_sampling=1, prestim_duration=2.5, 
               verbose=False)

FULL_RESPS = []
for i in range(len(FILES)):
    FULL_RESPS.append(compute_DS_population_resp(FILES[i], options, protocol_id=0))


# %%
from physion.analysis.behavioral_modulation import tuning_modulation_fig, compute_behavior_mod_population_tuning, population_tuning_fig, compute_DS_population_resp

full_resp = FULL_RESPS[0]
fig = population_tuning_fig(full_resp)
curves = compute_behavior_mod_population_tuning(full_resp,
                                                running_speed_threshold = 0.2,
                                                pupil_threshold=2.3)
fig1, fig2 = tuning_modulation_fig(curves, full_resp=full_resp)


# %%
fig = population_tuning_fig(full_resp)
curves = compute_behavior_mod_population_tuning(full_resp,
                                                running_speed_threshold = 0.2,
                                                pupil_threshold=2.7)
fig1, fig2 = tuning_modulation_fig(curves, full_resp=full_resp)

# %%
fig = population_tuning_fig(full_resp)
curves = compute_behavior_mod_population_tuning(full_resp,
                                                running_speed_threshold = 0.2,
                                                pupil_threshold=2.1)
fig1, fig2 = tuning_modulation_fig(curves, full_resp=full_resp)

# %%
FULL_RESPS = []
for i in range(len(FILES)):
    FULL_RESPS.append(compute_population_resp(FILES[i], options, protocol_id=0))

# %%
CURVES = []
thresholds = [2.9, 2.9, 2.1, 2.4, 2.6, 2.6, 2.7, None, None, 2.6, 3.1, 2.9, 2.5]

for i in range(len(FILES)):
    if thresholds[i] is not None:
        fig = population_tuning_fig(FULL_RESPS[i])
        ge.annotate(fig, 'session #%i: %s' % (i+1, FILES[i].split(os.path.sep)[-1]), (.5,1.),
                    xycoords='figure fraction', ha='center', va='top', size='xxx-small')
        curves = compute_behavior_mod_population_tuning(FULL_RESPS[i],
                                                        running_speed_threshold = 0.1,
                                                        pupil_threshold=thresholds[i])
        CURVES.append(curves)
        fig1, fig2 = tuning_modulation_fig(curves, full_resp=FULL_RESPS[i])
        ge.save_on_desktop(fig, 'figs/full-%i.png' % (i+1))
        ge.save_on_desktop(fig1, 'figs/running-%i.png' % (i+1))
        ge.save_on_desktop(fig2, 'figs/pupil-%i.png' % (i+1))
        for f in [fig, fig1, fig2]:
            plt.close(fig)


# %%
from datavyz import ge
# sessions average
SA = {}
for key in ['all', 'running', 'still', 'constricted', 'dilated']:
    SA[key] = np.zeros((len(CURVES), len(CURVES[0]['angles'])))

for i in range(len(CURVES)):
    norm = CURVES[i]['all'].max()
    SA['all'][i,:] = CURVES[i]['all']/norm
    for key in ['running', 'still', 'constricted', 'dilated']:
        SA[key][i,:] = CURVES[i]['%s_mean' % key]/norm
                  

# all episodes
fig, AX = ge.figure(axes=(3,1), figsize=(1.5,1.5))
ge.plot(CURVES[0]['angles'], SA['all'].mean(axis=0), sy=SA['all'].std(axis=0), no_set=True, ax=AX[0], lw=1)
ge.annotate(AX[0], 'all episodes', (1,1), va='top', ha='right')
ge.set_plot(AX[0], xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='norm. evoked resp.',
               xticks=[0,90,180,270], num_yticks=4)

# running vs still
ge.plot(CURVES[0]['angles'], SA['all'].mean(axis=0), no_set=True, ax=AX[1], lw=1, label='all')
ge.scatter(CURVES[0]['angles'], SA['running'].mean(axis=0), sy=SA['running'].std(axis=0),
        no_set=True, ax=AX[1], lw=1, color=ge.orange, label='running')
ge.scatter(CURVES[0]['angles'], SA['still'].mean(axis=0), sy=SA['still'].std(axis=0),
        no_set=True, ax=AX[1], lw=1, color=ge.blue, label='still')
ge.set_plot(AX[1], xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='norm. evoked resp.',
               xticks=[0,90,180,270], num_yticks=4)
ge.legend(AX[1], ncol=3, size='small')

# constricted vs dilated 
#ge.plot(CURVES[0]['angles'], SA['all'].mean(axis=0), no_set=True, ax=AX[2], lw=0.5, label='all')
ge.scatter(CURVES[0]['angles'], SA['constricted'].mean(axis=0), sy=SA['constricted'].std(axis=0),
        no_set=True, ax=AX[2], lw=1, color=ge.green, label='constricted')
ge.scatter(CURVES[0]['angles'], SA['dilated'].mean(axis=0), sy=SA['dilated'].std(axis=0),
        no_set=True, ax=AX[2], lw=1, color=ge.purple, label='dilated')
ge.set_plot(AX[2], xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='norm. evoked resp.',
               xticks=[0,90,180,270], num_yticks=4)
ge.legend(AX[2], ncol=2, size='small', loc='upper right')


for ax in AX:
    ge.annotate(ax, 'n=%i sessions' % len(CURVES), (0,1))
    
ge.save_on_desktop(fig, 'figs/final.png', dpi=300)

# %%
