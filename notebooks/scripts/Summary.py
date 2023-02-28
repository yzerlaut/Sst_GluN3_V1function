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

sys.path.append('../../../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
import physion.utils.plot_tools as pt
sys.path.append('../../src')
from analysis import * # with physion path

# %% [markdown]
# ## Orientation tuning

# %%
stat_test_props = dict(interval_pre=[-1.5,0],                                                                                                                  
                       interval_post=[1,2.5],
                       test='ttest',
                       positive=True)
                                                                                                                                                               
response_significance_threshold = 0.01                                                                                                                         

SUMMARY = {'WT':{}, 'GLUN3':{}, 'NR1':{}}

SUMMARY['GLUN3']['FILES'] = [\
               '/home/yann.zerlaut/CURATED/SST-Glun3KO-January-2023/2023_01_12-18-47-40.nwb',
               '/home/yann.zerlaut/CURATED/SST-Glun3KO-January-2023/2023_01_12-21-01-10.nwb',
               '/home/yann.zerlaut/CURATED/SST-Glun3KO-January-2023/2023_01_12-21-51-21.nwb']


SUMMARY['NR1']['FILES'] = [\
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-11-53-39.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-12-41-21.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-16-40-50.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-17-14-56.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-18-05-25.nwb',
             '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-18-52-59.nwb']


SUMMARY['WT']['FILES'] = [\
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-13-30-47.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-14-05-01.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-15-10-04.nwb',
            '/home/yann.zerlaut/CURATED/SST-GluN3KO-February-2023/2023_02_15-15-48-06.nwb']

for key in ['WT', 'GLUN3', 'NR1']:

    SUMMARY[key]['RESPONSES'], SUMMARY[key]['FRAC_RESP'] = [], []

    for f in SUMMARY[key]['FILES']:
        
        data = Data(f, verbose=False)
        responses, frac_resp, shifted_angle = compute_tuning_response_per_cells(data,
                                                                                verbose=False)
        SUMMARY[key]['RESPONSES'].append(responses)
        WT_FRAC_RESP.append(frac_resp)

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
        pt.plot(radii, np.mean(resp, axis=0), sy=np.std(resp, axis=0),
                ax=ax, color=color)
        
        ax.annotate(i*'\n'+'%s, N=%i sessions' % (key, len(SUMMARY[key]['RESPONSES'])), (1,1),
                    va='top', color=color, xycoords='axes fraction')

ax.set_ylabel('$\delta$ $\Delta$F/F')                                                                      
ax.set_xlabel('size ($^o$)')    


# %%
