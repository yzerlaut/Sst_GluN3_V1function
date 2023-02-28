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

sys.path.append('../../src')
from analysis import compute_tuning_response_per_cells
sys.path.append('../../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
import physion.utils.plot_tools as pt

# %% [markdown]
# ## Orientation tuning

# %%
SUMMARY = {'WT':{}, 'GluN3':{}, 'NR1':{}}

SUMMARY['GluN3']['FILES'] = [\
               '/home/yann.zerlaut/CURATED/SST-GluN3KO-January-2023/2023_01_12-18-47-40.nwb',
               '/home/yann.zerlaut/CURATED/SST-GluN3KO-January-2023/2023_01_12-21-01-10.nwb',
               '/home/yann.zerlaut/CURATED/SST-GluN3KO-January-2023/2023_01_12-21-51-21.nwb']


SUMMARY['NR1']['FILES'] = [\
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-11-53-39.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-12-41-21.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-16-40-50.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-17-14-56.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-18-05-25.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-18-52-59.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_17-13-48-50.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_17-14-35-39.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_17-18-47-20.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_17-19-21-51.nwb']


SUMMARY['WT']['FILES'] = [\
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-13-30-47.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-14-05-01.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-15-10-04.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-15-48-06.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_17-15-30-46.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_17-16-15-09.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_17-17-02-46.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_17-17-39-12.nwb']

for key in ['WT', 'GluN3', 'NR1']:

    SUMMARY[key]['RESPONSES'], SUMMARY[key]['FRAC_RESP'] = [], []

    for f in SUMMARY[key]['FILES']:
        
        data = Data(f, verbose=False)
        responses, frac_resp, shifted_angle = compute_tuning_response_per_cells(data,
                                                                                verbose=False)
        SUMMARY[key]['RESPONSES'].append(responses)
        SUMMARY[key]['FRAC_RESP'].append(frac_resp)

# %%
from physion.analysis.behavior import population_analysis
fig, ax = population_analysis(np.concatenate([SUMMARY[key]['FILES'] for key in ['WT', 'GluN3', 'NR1']]),
                              running_speed_threshold=0.1,
                            ax=None)


# %%
def plot_tunning_summary(shifted_angle, frac_resp, responses):
    """
    """
    fig, AX = pt.plt.subplots(1, 3, figsize=(6,1.5))
    pt.plt.subplots_adjust(wspace=0.8, top=0.7, bottom=0.3)

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
    
    NTOTs = [int(len(resp)/fr) for fr, resp in zip(frac_resp, responses)]
    Ns = [len(resp) for resp in responses]
    AX[2].annotate('responsive ROIS :\nN=%i sessions\n n= %i$\pm$%i / %i$\pm$%i ROIs  ' % (len(responses),
                                                                        np.mean(Ns), np.std(Ns),
                                                                        np.mean(NTOTs), np.std(NTOTs)),
                   (0.5, 0), va='top', ha='center',
                   xycoords='axes fraction')
    #ge.save_on_desktop(fig, 'fig.png', dpi=300)
    return fig, AX
 
for key in ['WT', 'GluN3', 'NR1']:

    fig, AX = plot_tunning_summary(shifted_angle,
                                   SUMMARY[key]['FRAC_RESP'], SUMMARY[key]['RESPONSES'])
    AX[2].set_title(key+' mice');
    fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', key+'.svg'))


# %%
fig, AX = pt.plt.subplots(3, 1, figsize=(1.8,4))
pt.plt.subplots_adjust(hspace=0.7, top=0.95, bottom=0.2, left=0.33)

for i, key, color in zip(range(3), ['WT', 'GluN3', 'NR1'], ['k', 'tab:blue', 'tab:green']):
    
    RESPONSES = [np.mean(responses, axis=0) for responses in SUMMARY[key]['RESPONSES']]
    pt.plot(shifted_angle, np.mean(RESPONSES, axis=0),
            0*np.std(RESPONSES, axis=0),
            ax=AX[0], color=color)
    
    N_RESP = [resp/resp[1] for resp in RESPONSES]
    pt.plot(shifted_angle, np.mean(N_RESP, axis=0),
            sy=0*np.std(N_RESP, axis=0), ax=AX[1], color=color)
    
    AX[2].bar([i],
              [100*np.mean(SUMMARY[key]['FRAC_RESP'])],
              yerr=[100*np.std(SUMMARY[key]['FRAC_RESP'])],
              color=color)

AX[0].set_ylabel('$\Delta$F/F')
AX[0].set_xlabel('angle ($^o$)')
AX[1].set_ylabel('n. $\Delta$F/F')
AX[1].set_xlabel('angle ($^o$)')

AX[2].set_ylabel('fraction (%)\nresponsive')
AX[2].set_xticks([0,1,2])
AX[2].set_xticklabels(['WT', 'GluN3', 'NR1'], rotation=70)
AX[2].set_xlim([-2,4])

fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'final.svg'))

# %% [markdown]
# ## Size tuning

# %%
DATASET = scan_folder_for_NWBfiles('/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023')

SUMMARY = {'WT':{'FILES':[]}, 'GLUN3':{'FILES':[]}, 'NR1':{'FILES':[]}}

for f, s, p in zip(DATASET['files'], DATASET['subjects'], DATASET['protocols']):
    if ('size-tuning' in p[0]):
        if 'GluN1' in s:
            SUMMARY['NR1']['FILES'].append(f)
        else:
            SUMMARY['WT']['FILES'].append(f)

SUMMARY

# %%
from physion.analysis.protocols.size_tuning import center_and_compute_size_tuning

for key in ['WT', 'GLUN3', 'NR1']:

    SUMMARY[key]['RESPONSES'] = []

    for f in SUMMARY[key]['FILES']:
        
        data = Data(f, verbose=False)

        radii, size_resps = center_and_compute_size_tuning(data, verbose=False)
        
        SUMMARY[key]['RESPONSES'].append(size_resps)
        
SUMMARY

# %%
fig, ax = pt.plt.subplots(1, figsize=(3,2))

for i, key, color in zip(range(3), ['WT', 'GLUN3', 'NR1'], ['k', 'tab:blue', 'tab:green']):
    
    if len(SUMMARY[key]['RESPONSES'])>0:
        
        resp = [np.mean(r, axis=0) for r in SUMMARY[key]['RESPONSES']]
        for r in SUMMARY[key]['RESPONSES']:
            ax.plot(radii, np.mean(r, axis=0), lw=0.5, color=color)
        
        pt.plot(radii, np.mean(resp, axis=0), sy=np.std(resp, axis=0),
                ax=ax, color=color)
        
        ax.annotate(i*'\n'+'%s, N=%i sessions' % (key, len(SUMMARY[key]['RESPONSES'])), (1,1),
                    va='top', color=color, xycoords='axes fraction')

ax.set_ylabel('$\delta$ $\Delta$F/F')                                                                      
ax.set_xlabel('size ($^o$)')    

#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'final.svg'))


# %%
