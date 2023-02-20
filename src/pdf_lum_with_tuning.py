import os, tempfile, subprocess, sys, pathlib
import numpy as np
from scipy.stats import skew
from PIL import Image

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

    """
    KEYS = ['resp-fraction', 'TA-all']

    LOCS = [(300, 150), (200, 700)]

    for i in range(2):

        KEYS.append('TA-%i'%i)
        LOCS.append((300, 1750+800*i))

    for key, loc in zip(KEYS, LOCS):
        
        if os.path.isfile(os.path.join(tempfile.tempdir, '%s-%i.png' % (key, args.unique_run_ID))):

            fig = Image.open(os.path.join(tempfile.tempdir, '%s-%i.png' % (key, args.unique_run_ID)))
            page.paste(fig, box=loc)
            fig.close()

    """
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

    episodes = EpisodeData(data,
                           protocol_id=0,
                           quantities=[args.imaging_quantity],
                           prestim_duration=3,
                           with_visual_stim=True,
                           verbose=True)






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

