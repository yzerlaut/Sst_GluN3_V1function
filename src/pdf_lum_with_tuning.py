import os, tempfile, subprocess, sys, pathlib
import numpy as np
from scipy.stats import skew
from PIL import Image

sys.path.append('./src')
from analysis import *

physion_folder = os.path.join(pathlib.Path(__file__).resolve().parent,
                              '..', 'physion', 'src')
sys.path.append(os.path.join(physion_folder))

import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data
from physion.analysis.summary_pdf import summary_pdf_folder,\
        metadata_fig, generate_FOV_fig, generate_raw_data_figs, join_pdf
from physion.dataviz.tools import format_key_value
from physion.dataviz.episodes.trial_average import plot_trial_average
from physion.analysis.process_NWB import EpisodeData
from physion.utils.plot_tools import pie

tempfile.gettempdir()

stat_test_props = dict(interval_pre=[-1.5,0.0],
                       interval_post=[0.5,2.0],
                       test='ttest',
                       positive=True)

response_significance_threshold = 0.01

def generate_pdf(args,
                 subject='Mouse'):

    pdf_file= os.path.join(summary_pdf_folder(args.datafile), 'Summary.pdf')

    PAGES  = [os.path.join(tempfile.tempdir, 'session-summary-1-%i.pdf' % args.unique_run_ID),
              os.path.join(tempfile.tempdir, 'session-summary-2-%i.pdf' % args.unique_run_ID)]

    rois = generate_figs(args)

    width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi : (2481, 3510)

    ### Page 1 - Raw Data

    # let's create the A4 page
    page = Image.new('RGB', (width, height), 'white')

    KEYS = ['metadata',
            'raw-full', 'lum-resp', 'raw-0',
            'FOV']

    LOCS = [(200, 130),
            (150, 650), (150, 1600), (150, 2400),
            (900, 130)]

    for key, loc in zip(KEYS, LOCS):
        
        fig = Image.open(os.path.join(tempfile.tempdir, '%s-%i.png' % (key, args.unique_run_ID)))
        page.paste(fig, box=loc)
        fig.close()

    page.save(PAGES[0])

    ### Page 2 - Analysis

    page = Image.new('RGB', (width, height), 'white')

    KEYS = ['tuning-summary', 'tuning-examples']

    LOCS = [(300, 150), (200, 700)]

    for key, loc in zip(KEYS, LOCS):
        
        if os.path.isfile(os.path.join(tempfile.tempdir, '%s-%i.png' % (key, args.unique_run_ID))):

            fig = Image.open(os.path.join(tempfile.tempdir, '%s-%i.png' % (key, args.unique_run_ID)))
            page.paste(fig, box=loc)
            fig.close()

    page.save(PAGES[1])

    join_pdf(PAGES, pdf_file)


def generate_lum_response_fig(results, data, args):

    fig, AX = pt.plt.subplots(1, 3, figsize=(7,3))
    fig.subplots_adjust(wspace=0.6, bottom=.4)


    values = ['dark' ,'black', 'grey']

    for i, key in enumerate(['mean', 'std', 'skewness']):

        data = [results[lum][key] for lum in values]

        pt.violin(data, ax=AX[i],
                  labels=values)
        AX[i].set_title('$\Delta$F/F '+key)
    AX[0].set_ylabel('$\Delta$F/F')
    AX[1].set_ylabel('$\Delta$F/F')

    return fig

def annotate_luminosity_and_get_summary(data, args, ax=None):
    
    summary = {}
    for lum in ['dark' ,'grey', 'black']:
        summary[lum] = {}
        for key in ['mean', 'std', 'skewness']:
            summary[lum][key] = []
            
    if 'BlankFirst' in data.metadata['protocol']:
        tstarts = data.nwbfile.stimulus['time_start_realigned'].data[:3]
        tstops = data.nwbfile.stimulus['time_stop_realigned'].data[:3]
        values = ['dark' ,'grey', 'black']
    elif 'BlankLast' in data.metadata['protocol']:
        tstarts = data.nwbfile.stimulus['time_start_realigned'].data[-3:]
        tstops = data.nwbfile.stimulus['time_stop_realigned'].data[-3:]
        values = ['black' ,'grey', 'dark']
    else:
        print(' Protocol not recognized !!  ')
        
    for tstart, tstop, lum in zip(tstarts, tstops, values):
        t_cond = (data.t_dFoF>tstart) & (data.t_dFoF<tstop)
        if ax is not None:
            ax.annotate(lum, (.5*(tstart+tstop), 0), va='top', ha='center')
            ax.fill_between([tstart, tstop], np.zeros(2), np.ones(2), lw=0, 
                            alpha=.2, color='k')
        for roi in range(data.nROIs):
            for key, func in zip(['mean', 'std', 'skewness'], [np.mean, np.std, skew]):
                summary[lum][key].append(func(data.dFoF[roi,t_cond]))
                
    return summary

def compute_response_per_cells(data):
    
    RESPONSES = []

    protocol_id = data.get_protocol_id(protocol_name='ff-gratings-8orientation-2contrasts-10repeats')

    EPISODES = EpisodeData(data,
                           quantities=['dFoF'],
                           protocol_id=protocol_id,
                           verbose=True)
                               
    shifted_angle = EPISODES.varied_parameters['angle']-EPISODES.varied_parameters['angle'][1]
    
    for roi in np.arange(data.nROIs)[:10]:

        cell_resp = EPISODES.compute_summary_data(response_significance_threshold=\
                                                          response_significance_threshold,
                                                  response_args=dict(quantity='dFoF', roiIndex=roi),
                                                  stat_test_props=stat_test_props)

        #condition = np.ones(len(cell_resp['angle']), dtype=bool) # no condition
        condition = cell_resp['contrast']==1 # RESTRICT TO FULL CONTRAST
        
        if np.sum(cell_resp['significant'][condition]):
            
            ipref = np.argmax(cell_resp['value'][condition])
            prefered_angle = cell_resp['angle'][condition][ipref]

            RESPONSES.append(np.zeros(len(shifted_angle)))

            for angle, value in zip(cell_resp['angle'][condition],
                                    cell_resp['value'][condition]):

                new_angle = shift_orientation_according_to_pref(angle, 
                                                                pref_angle=prefered_angle, 
                                                                start_angle=-22.5, 
                                                                angle_range=180)
                iangle = np.flatnonzero(shifted_angle==new_angle)[0]

                RESPONSES[-1][iangle] = value
                
    return RESPONSES, shifted_angle

def cell_tuning_example_fig(data,
                            Nsamples = 15, # how many cells we show
                            seed=10):
    np.random.seed(seed)
    
    protocol_id = data.get_protocol_id(protocol_name='ff-gratings-8orientation-2contrasts-10repeats')

    EPISODES = EpisodeData(data,
                           quantities=['dFoF'],
                           protocol_id=protocol_id,
                           verbose=True)
    
    fig, AX = pt.plt.subplots(Nsamples, len(EPISODES.varied_parameters['angle']), 
                          figsize=(7.5,9))
    pt.plt.subplots_adjust(right=0.7, left=0.1, top=0.97, bottom=0.05,
                            wspace=0.1, hspace=0.8)
    
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
        inset = pt.inset(AX[i][-1], (2.6, 0.2, 1.2, 0.8))

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
                        weight='bold', fontsize=6, color=(pt.plt.cm.tab10(2) if responsive else pt.plt.cm.tab10(3)),
                        xycoords='axes fraction')
        
    return fig


def plot_tunning_summary(data, shifted_angle, RESPONSES):
    """
    """
    fig, AX = pt.plt.subplots(1, 3, figsize=(6,1.7))
    pt.plt.subplots_adjust(wspace=0.8, bottom=.25, top=.9)

    # raw
    pt.plot(shifted_angle, np.mean(RESPONSES, axis=0),
            sy=np.std(RESPONSES, axis=0), ax=AX[0])
    AX[0].set_ylabel('$\Delta$F/F')
    AX[0].set_title('raw resp.')

    for ax in AX[:2]:
        ax.set_xlabel('angle ($^o$)')

    # peak normalized
    N_RESP = [resp/resp[1] for resp in RESPONSES]
    pt.plot(shifted_angle, np.mean(N_RESP, axis=0),
            sy=np.std(N_RESP, axis=0), ax=AX[1])

    AX[1].set_yticks([0, 0.5, 1])
    AX[1].set_ylabel('n. $\Delta$F/F')
    AX[1].set_title('peak normalized')

    pt.pie([len(RESPONSES)/data.nROIs, 1-len(RESPONSES)/data.nROIs],
           pie_labels=['%.1f%%' % (100.*len(RESPONSES)/data.nROIs),
                       '     %.1f%%' % (100.*(1-len(RESPONSES)/data.nROIs))],
           COLORS=[pt.plt.cm.tab10(2), pt.plt.cm.tab10(1)], ax=AX[2])
    AX[2].annotate('responsive ROIS :\nn=%i / %i   ' % (len(RESPONSES), data.nROIs),
                   (0.5, 0), va='top', ha='center',
                   xycoords='axes fraction')
    #ge.save_on_desktop(fig, 'fig.png', dpi=300)
    return fig, AX
 

def generate_figs(args,
                  Nexample=2):


    pdf_folder = summary_pdf_folder(args.datafile)

    data = Data(args.datafile)
    if args.imaging_quantity=='dFoF':
        data.build_dFoF()
    else:
        data.build_rawFluo()

    # ## --- METADATA  ---
    fig = metadata_fig(data, short=True)
    fig.savefig(os.path.join(tempfile.tempdir, 'metadata-%i.png' % args.unique_run_ID), dpi=300)

    # ##  --- FOVs ---
    fig = generate_FOV_fig(data, args)
    fig.savefig(os.path.join(tempfile.tempdir, 'FOV-%i.png' % args.unique_run_ID), dpi=300)

    # ## --- FULL RECORDING VIEW --- 
    args.raw_figsize=(7, 3.2)
    figs, axs = generate_raw_data_figs(data, args,
                                      TLIMS = [(15, 65)],
                                      return_figs=True)
    figs[0].subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.9)

    results = annotate_luminosity_and_get_summary(data, args, ax=axs[0])
    figs[0].savefig(os.path.join(tempfile.tempdir,
                    'raw-full-%i.png' % args.unique_run_ID), dpi=300)

    fig = generate_lum_response_fig(results, data, args)
    fig.savefig(os.path.join(tempfile.tempdir,
        'lum-resp-%i.png' % args.unique_run_ID), dpi=300)


    # ## --- EPISODES AVERAGE -- 

    fig = cell_tuning_example_fig(data)
    fig.savefig(os.path.join(tempfile.tempdir,
        'tuning-examples-%i.png' % args.unique_run_ID), dpi=300)

    RESPONSES, shifted_angle = compute_response_per_cells(data)

    fig, AX = plot_tunning_summary(data, shifted_angle, RESPONSES)
    fig.savefig(os.path.join(tempfile.tempdir,
        'tuning-summary-%i.png' % args.unique_run_ID), dpi=300)


if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument("datafile", type=str)

    parser.add_argument("--iprotocol", type=int, default=0,
        help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument("--imaging_quantity", default='dFoF')
    parser.add_argument("--nROIs", type=int, default=5)
    parser.add_argument("--show_all_ROIs", action='store_true')
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    args.unique_run_ID = np.random.randint(10000)
    print('unique run ID', args.unique_run_ID)

    if '.nwb' in args.datafile:
        if args.debug:
            generate_figs(args)
            pt.plt.show()
        else:
            generate_pdf(args)

    else:
        print('/!\ Need to provide a NWB datafile as argument ')

