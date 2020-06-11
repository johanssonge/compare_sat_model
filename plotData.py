'''
Created on 29 Aug 2018

@author: a001721
'''
from optparse import OptionParser
import numpy as np
from matplotlib import pyplot as plt
import warnings
import pdb
import os
from calcSat import all_months_comb
from scipy.stats import ttest_ind, mannwhitneyu
from matplotlib.patches import Ellipse, Rectangle


def getLonLatMask(ll, llmin, llmax):
    llf = ll[0]
    lls = ll[-1] + 1
    #: over edge
    if llmin > llmax:
        llmin1 = llmin
        llmin2 = llf
        llmax1 = lls
        llmax2 = llmax
    else:
        llmin1 = llmin
        llmin2 = llmin
        llmax1 = llmax
        llmax2 = llmax
    mask = ( ( (ll >= llmin1) & (ll < llmax1) ) | 
             ( (ll >= llmin2) & (ll < llmax2) ) )
    return mask



def calculateLeapingMean2D(data_list,nr):
    nrstep=int(nr/2)
    size=data_list.shape[1]
    retv=np.zeros(data_list.shape)
    for i in range(nrstep):        
        retv[:, i] = np.nanmean(data_list[:, 0:(i+nrstep+1)], axis=1)  # @UndefinedVariable
    for ii in range(size-(2*nrstep)):
        j=ii+nrstep
        retv[:, j] = np.nanmean(data_list[:, (j-nrstep):(j+nrstep+1)], axis=1)  # @UndefinedVariable
    for iii in range(nrstep):
        jj=size-(nrstep-iii)
        retv[:, jj]= np.nanmean(data_list[:, (jj-nrstep):], axis=1)  # @UndefinedVariable
    return retv


def calculateVertMean(e_res, e_res_stat, namn, i, p, lat, lon, use_datline_center):
    if use_datline_center:
        lonmin = {'nino4': 160, 'nino3': 210}
        lonmax = {'nino4': 210, 'nino3': 270}
    else:
        lonmin = {'nino4': 160, 'nino3': -150}
        lonmax = {'nino4': -150, 'nino3': -90}
    if p == 0:
        #: Hela tropicerna
        latmask = np.ones(e_res_stat[namn].shape[1]).astype(bool)
        lonmask = np.ones(e_res_stat[namn].shape[2]).astype(bool)
    elif p == 1:
        #: Nin03
        latmask = getLonLatMask(lat, -5, 5)
        lonmask = getLonLatMask(lon, lonmin['nino3'], lonmax['nino3'])
    elif p == 2:
        #: Nino4
        latmask = getLonLatMask(lat, -5, 5)
        lonmask = getLonLatMask(lon, lonmin['nino4'], lonmax['nino4'])
    mean_vert = np.nanmean(e_res_stat[namn][:, latmask, :][:, :, lonmask], axis=(1, 2))  # @UndefinedVariable
    try:        
        if (p == 0) and(e_res[namn + '_vert'].ndim == 2):
            e_res[namn + '_vert'][:, i] = mean_vert
        else:
            e_res[namn + '_vert'][p, :, i] = mean_vert
    except:
        pdb.set_trace()
    return e_res


def calculateLonMean(e_res, e_res_stat, namn, i, p, lat):
    if p == 0:
        #: Hela tropikerna -30 - 30
        latmask = np.ones(e_res_stat[namn].shape[1]).astype(bool)
    elif p == 1:
        latmask = getLonLatMask(lat, -30, -15)
    elif p == 2:
        latmask = getLonLatMask(lat, -15, 0)
    elif p == 3:
        latmask = getLonLatMask(lat, 0, 15)
    elif p == 4:
        latmask = getLonLatMask(lat, 15, 30)
    elif p == 5:
        latmask = getLonLatMask(lat, -15, 15)

    mean_lon = np.nanmean(e_res_stat[namn][:, latmask, :], axis=1)  # @UndefinedVariable
    if (p == 0) and(e_res[namn + '_lon'].ndim == 3):
        e_res[namn + '_lon'][:, :, i] = mean_lon
    else:
        e_res[namn + '_lon'][p, :, :, i] = mean_lon
    return e_res


def getCbColour():
    CB_color_cycle = {'blue': '#377eb8', 'orange': '#ff7f00', 'green': '#4daf4a',
                      'pink': '#f781bf', 'brown': '#a65628', 'lila': '#984ea3',
                      'grey': '#999999', 'red': '#e41a1c', 'gul': '#dede00'}
    colour_sat = CB_color_cycle['lila']
    colour_hre = CB_color_cycle['brown']
    colour_lre = CB_color_cycle['blue']
    colour_lr6 = CB_color_cycle['green']
    colour = {'sat': colour_sat, 'hre': colour_hre, 'lre': colour_lre, 'lr6': colour_lr6}
    return colour

def getLonTicks(pname, pname_e=None, pname_val=None, pname_e_val=None):
    xticks = {'sat': [1, 45, 90, 135, 180, 225, 270, 315, 358], 
              'hre': [1, 128, 256, 384, 512, 640, 768, 896, 1023], 
              'lre': [1, 64, 128, 192, 256, 320, 384, 448, 511], 
              'lr6': [1, 64, 128, 192, 256, 320, 384, 448, 511]}
    yticks = {'sat': [0, 21, 42, 63, 83], 'hre': [0, 10, 20, 31, 41] ,
              'lre': [0, 10, 20, 31, 41], 'lr6': [0, 10, 20, 31, 41]}
    if pname_e is None:
        return xticks[pname], yticks[pname]
    #: Used for Diff, i.e. when there is 2 name to chose from
    #: Always use the one with smallest resolution
    #: First in shape is height
    #: Last in shape is Longitude
    if pname_e_val.shape[0] < pname_val.shape[0]:
        yt = yticks[pname_e]
    else:
        yt = yticks[pname]
    
    if pname_e_val.shape[-1] < pname_val.shape[-1]:
        xt = xticks[pname_e]
    else:
        xt = xticks[pname]
    return xt, yt

def plotLonTOA(rhd, valminmax, figname_st, useClim=False, use_datline_center=False):
    xticks = {'sat': [1, 45, 90, 135, 180, 225, 270, 315, 358],'lre': [1, 64, 128, 192, 256, 320, 384, 448, 511], 'hre': [1, 128, 256, 384, 512, 640, 768, 896, 1023]}
    yticks = {'sat': [1, 16, 31, 46, 61],'lre': [1, 22, 43, 65, 86], 'hre': [1, 43, 86, 128, 170]}
#     {'sat': ,'lre': }
    for datan in rhd.keys():#['sat', 'lre', 'hre']:
        fig = plt.figure()
        f = 0
        
        f = f + 1
#             endLat = stepL + step
#             stepInd = (stLat[datan] >= stepL) & (stLat[datan] < endLat)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rhdLat = np.nanmean(rhd[datan], axis=0)  # @UndefinedVariable
        warnings.resetwarnings()
        aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
        ax = fig.add_subplot(111)
        im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax[datan] * -1, vmax=valminmax[datan])
        
        xticks, yticks = getLonTicks(datan)
        ax.set_yticks(yticks)
        ax.set_yticklabels(['-30', '-15', '0', '15', '30'])

        ax.set_xticks(xticks)
        ax.set_title(datan)
        ax.set_ylabel('Latitude [deg]')

        if use_datline_center:
            ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'])
        else:
            ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
        ax.set_xlabel('Longitude [deg]')
        barticks = [valminmax[datan]*-1, valminmax[datan]*-0.75, valminmax[datan]*-0.5, valminmax[datan]*-0.25, 0, valminmax[datan]*0.25, valminmax[datan]*0.5, valminmax[datan]*0.75, valminmax[datan]]
        cbar_ax = fig.add_axes([0.2, 0.25, 0.6, 0.01])
        cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)  # @UnusedVariable
#         else:
#             ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
        plt.tight_layout()
        figname = figname_st[datan] + '_top'
        if not use_datline_center:
            figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
        if useClim:
            figname = figname + '_anom'
        fig.savefig(figname + '.png')
#         fig.savefig('test.png')
        fig.show()
    pdb.set_trace()


def plotLonStep(rhd, step, stLat, valminmax, figname_st, useClim=False, use_datline_center=False, useSMA=0):
    
    start_lat = -15
    num_fig = (abs(start_lat) * 2) / step
    if num_fig == 2:
        num_fig = num_fig + 1
    
    if num_fig == 6:
        figsize = (8,24)
    elif num_fig == 4:
        figsize = (8,10)
    elif num_fig == 3:
        figsize = (8,7)
    elif num_fig == 1:
        figsize = (8,4)
    else:
        figsize = (8,8)
    k = 0
    for datan in rhd.keys():#['sat', 'lre', 'hre']:
        #: 'e_' is used for statistic calculation
        if 'e_' in datan:
            continue
        k = k + 1
        fig = plt.figure(figsize=figsize)
#         plt.rcParams['figure.constrained_layout.use'] = True
        f = 0
        for stepL in range(start_lat,abs(start_lat),step)[::-1]:
            f = f + 1
            endLat = stepL + step
            stepInd = getLonLatMask(stLat[datan], stepL, endLat)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rhdLat = np.nanmean(rhd[datan][:,stepInd,:], axis=1)  # @UndefinedVariable
            if datan == 'sat':
                rhdLat = rhdLat[0:86, :]
            warnings.resetwarnings()
            if (datan == 'sat') and (useSMA != 0):
                rhdLat = calculateLeapingMean2D(rhdLat, useSMA)
#                 pdb.set_trace()
            aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
            ax = fig.add_subplot(num_fig,1,f)#, constrained_layout=True)
            im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax[datan] * -1, vmax=valminmax[datan])
            xticks, yticks = getLonTicks(datan)
            ax.set_yticks(yticks)
            ax.set_yticklabels(['0', '5', '10', '15', '20'])
    
            ax.set_xticks(xticks)
            
#             ax.set_title('Latitude, %i - %i' %(stepL, endLat))
            ax.set_ylabel('Height [km]')
            if 'ensoN' in figname_st[datan]:
                if num_fig == 1:
                    ax.text(0.01, 1.04, chr((f-1) + 98), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
                elif num_fig == 3:
                    ax.text(0.01, 1.04, chr((f-1) + 99), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
                elif num_fig == 4:
                    ax.text(0.01, 1.04, chr((f-1) + 101), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            else:
                ax.text(0.01, 1.04, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            if stepL == start_lat:
                if use_datline_center:
                    ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'])
                else:
                    ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
                ax.set_xlabel('Longitude [deg]')
                barticks = [valminmax[datan]*-1, valminmax[datan]*-0.75, valminmax[datan]*-0.5, valminmax[datan]*-0.25, 0, valminmax[datan]*0.25, valminmax[datan]*0.5, valminmax[datan]*0.75, valminmax[datan]]
                if num_fig == 1:
                    cbar_ax = fig.add_axes([0.2, 0.33, 0.6, 0.01])
                else:
                    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.01])#[0.2, 0.04, 0.6, 0.01])
                cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)  # @UnusedVariable
                cbar.set_label('CRH [K/day]')
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
            if f == 1:
                if use_datline_center:
                    ax.text(0.94, 0.93, 'Atl', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.6, 0.93, 'Pac', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.22, 0.93, 'IO', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')

        
        if start_lat == -15:
            fig.tight_layout()#rect=[0, 0.03, 1, 0.95])
        if num_fig > 1:
            plt.subplots_adjust(hspace=0.0001)
        else:
            plt.subplots_adjust(top=1.4)
        
        figname = figname_st[datan] + '_lon_%ideg-step' %step
        if not use_datline_center:
            figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
        if (datan == 'sat') and (useSMA != 0):
            figname = figname + '_SMA-%i' %useSMA
        if useClim:
            figname = figname + '_anom'
        fig.savefig(figname + '.png')
        fig.show()
    pdb.set_trace()


def plotLonEnso(rhd, lats, valminmax, figname_st, useClim=False, use_datline_center=False, useSMA=0):
    
    step = 30
    latmin = -15
    num_fig = 3
    figsize = (8,7)
    k = 0
    for datan in lats.keys():#['sat', 'lre', 'hre']:
        #: 'e_' is used for statistic calculation
        if 'e_' in datan:
            continue
        k = k + 1
        fig = plt.figure(figsize=figsize)
        f = 0
        for mon in ['ensoP', 'ensoN']:
#         plt.rcParams['figure.constrained_layout.use'] = True
            f = f + 1
            latmax = latmin + step
            latmask = getLonLatMask(lats[datan], latmin, latmax)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rhdLat = np.nanmean(rhd['%s_%s' %(mon, datan)]['crh'][:, latmask, :], axis = 1)  # @UndefinedVariable
            if datan == 'sat':
                rhdLat = rhdLat[0:86, :]
            warnings.resetwarnings()
            if (datan == 'sat') and (useSMA != 0):
                rhdLat = calculateLeapingMean2D(rhdLat, useSMA)
            aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
            ax = fig.add_subplot(num_fig,1,f)#, constrained_layout=True)
            im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax[datan] * -1, vmax=valminmax[datan])
#             if datan != 'hre':
#                 ax2 = ax.twiny()
            xticks, yticks = getLonTicks(datan)
            ax.set_yticks(yticks)
            ax.set_yticklabels(['0', '5', '10', '15', '20'])
            ax.set_xticks(xticks)
            ax.set_ylabel('Height [km]')
            if datan != 'hre':
                ax.text(1.03, 0.5, '%s' %mon.upper(), ha='center', va='center',rotation=90,transform=ax.transAxes)
#                 ax.text(0.01, 1.04, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
# #                 ax.yaxis.set_label_position("right")
#                 ax2.set_ylabel(mon.upper())
#                 ax2.tick_params(axis='x', which='both',bottom=False,top=False)
#                 ax2.set_xticklabels(['', '', '', '', '', ''])
#                 ax2.set_yticklabels(['', '', '', '', ''])
#             ax.set_title('Latitude, %i - %i' %(stepL, endLat))
            ax.text(0.01, 1.04, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            if f == 2:
                if use_datline_center:
                    ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'])
                else:
                    ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
                ax.set_xlabel('Longitude [deg]')
                barticks = [valminmax[datan]*-1, valminmax[datan]*-0.75, valminmax[datan]*-0.5, valminmax[datan]*-0.25, 0, valminmax[datan]*0.25, valminmax[datan]*0.5, valminmax[datan]*0.75, valminmax[datan]]
                if num_fig == 1:
                    cbar_ax = fig.add_axes([0.2, 0.33, 0.6, 0.01])
                else:
                    cbar_ax = fig.add_axes([0.2, 0.28, 0.6, 0.01])#[0.2, 0.04, 0.6, 0.01])
                cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)  # @UnusedVariable
                cbar.set_label('CRH [K/day]')
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
            if f == 1:
                if use_datline_center:
                    ax.text(0.94, 0.93, 'Atl', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.6, 0.93, 'Pac', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.22, 0.93, 'IO', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')

        
        figname = figname_st[datan].replace('ensoN', 'enso').replace('ensoP', 'enso') + '_lon'
        if not use_datline_center:
            figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
        if (datan == 'sat') and (useSMA != 0):
            figname = figname + '_SMA-%i' %useSMA
        if useClim:
            figname = figname + '_anom'
        fig.savefig(figname + '.png')
        fig.show()
    pdb.set_trace()


def plotLonSeason_diff_forPresentation(rhd, stLat, valminmax, figname_se, top_rad, bot_rad, useClim=False, use_datline_center=False, extraLon = None):
    figsize = (8,5)
    num_fig = 1
#     fig2 = plt.figure(figsize=figsize)
    for a in range(5):
        fig = plt.figure(figsize=figsize)
        f = 0
        for mon in ['djf']:#, 'jja']:
            f = f + 1
            tr_stepInd = (stLat[top_rad] >= -30) & (stLat[top_rad] <= 30)
            br_stepInd = (stLat[bot_rad] >= -30) & (stLat[bot_rad] <= 30)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tr_rhdLat = np.nanmean(rhd['%s_%s' %(mon, top_rad)]['crh'][:,tr_stepInd,:], axis=1)  # @UndefinedVariable
            br_rhdLat = np.nanmean(rhd['%s_%s' %(mon, bot_rad)]['crh'][:,br_stepInd,:], axis=1)  # @UndefinedVariable
            
            
            rhdLat = calculateLonDiff(tr_rhdLat, br_rhdLat, extraLon)
            warnings.resetwarnings()
            aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
            ax = fig.add_subplot(num_fig + 1,1,f)
            im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax * -1, vmax=valminmax)
            trhd_rs, brhd_rs = reshapeRhdLat(rhd['e_%s_%s' %(mon, top_rad)]['crh_lon'], rhd['e_%s_%s' %(mon, bot_rad)]['crh_lon'], extraLon)
            t1, p1 = ttest_ind(trhd_rs, brhd_rs, axis=2)
            ax.contourf(p1<=0.05, 1, origin='lower', hatches=['', '.'], alpha=0)
    
            xticks, yticks = getLonTicks(top_rad, bot_rad, tr_rhdLat, br_rhdLat)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_yticklabels(['0', '5', '10', '15', '20'], fontsize='large')
            ax.set_ylabel('Height [km]', fontsize='x-large')
            ax.set_title(mon.upper(), fontsize='x-large')
            if ('sat' in [top_rad, bot_rad]) and ('hre' in [top_rad, bot_rad]):
                start_letter = 101 #: e
            else:
                start_letter = 97 #: a
    #         ax.text(0.01, 1.04, chr((f-1) + start_letter), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            #: Ellipse Rectangle
            if a == 1:
                ellipse = Rectangle((0,19), 360, 14, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
            elif (a == 2):
                ellipse = Rectangle((0,8), 360, 12, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
#                 ellipse = Ellipse((25,44), 40, 80, fc='none', ec='green', lw=2)
#                 ax.add_patch(ellipse)
#             elif (a == 3) and (f == 2):
#                 ellipse = Ellipse((90,44), 40, 80, fc='none', ec='green', lw=2)
#                 ax.add_patch(ellipse)
            elif (a == 3):
                if (f == 2):
                    ellipse = Ellipse((60,4), 100, 10, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
                ellipse = Ellipse((250,4), 90, 10, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
                ellipse = Ellipse((340,4), 80, 10, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
            elif (a == 4):
                ellipse = Ellipse((305,22), 40, 40, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
                ellipse = Ellipse((25,22), 40, 40, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
                

            if f == num_fig:
                if use_datline_center:
                    ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'], fontsize='large')
                else:
                    ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'], fontsize='large')
                ax.set_xlabel('Longitude [deg]', fontsize='x-large')
                barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
                cbar_ax = fig.add_axes([0.2, 0.4, 0.6, 0.03])#[0.2, 0.04, 0.6, 0.01])
                cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax)#, ticks=barticks)  # @UnusedVariable
                cbar.set_label('CRH [K/day]', fontsize='x-large')
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
            if f == 1:
                if use_datline_center:
                    ax.text(0.94, 0.93, 'Atl', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.6, 0.93, 'Pac', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.22, 0.93, 'IO', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
#         plt.subplots_adjust(hspace=0.0001) 
        figname = figname_se + '_lon'
        figname = figname.replace('yyy', '%s-%s' %(top_rad.upper(), bot_rad.upper()))
        if not use_datline_center:
            figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
        if useClim:
            figname = figname + '_anom'
        figname = 'Kappa_plots/' + os.path.basename(figname) + '_ellipse-%d' %a
        fig.savefig(figname + '.png')
        fig.show()
    pdb.set_trace()


def plotLonSeason_forPresentation(rhd, stLat, valminmax, figname_se, useClim=False, use_datline_center=False):
#     xticks = {'sat': [1, 45, 90, 135, 180, 225, 270, 315, 358], 'hre': [1, 128, 256, 384, 512, 640, 768, 896, 1023],'lre': [1, 64, 128, 192, 256, 320, 384, 448, 511],'lr6': [1, 64, 128, 192, 256, 320, 384, 448, 511]}
#     yticks = {'sat': [1, 21, 42, 63, 83], 'hre': [1, 10, 20, 31, 41],'lre': [1, 10, 20, 31, 41],'lr6': [1, 10, 20, 31, 41]}

    figsize = (8,6)
    num_fig = 2
    a = -1
    for datans in ['sat'] * 10:#stLat.keys():#['lre', 'sat', 'hre']:
        a = a + 1
        fig = plt.figure(figsize=figsize)
        f = 0
        if a < 5:
            months = ['djf', 'jja']
        else:
            months = ['sat', 'hre']
        for month in months:
            if month in ['sat', 'hre']:
                mon = 'djf'
                datan = month
            else:
                datan = datans
                mon = month
                
            stepInd = (stLat[datan] <= 30) & (stLat[datan] >= -30)
            data_read = '%s_%s' %(mon, datan)
            f = f + 1
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rhdLat = np.nanmean(rhd[data_read]['crh'][:, stepInd, :], axis=1)  # @UndefinedVariable
            if datan == 'sat':
                rhdLat = rhdLat[0:86, :]
            warnings.resetwarnings()
            aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
            ax = fig.add_subplot(num_fig + 1,1,f)
            im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax[datan] * -1, vmax=valminmax[datan])
            xticks, yticks = getLonTicks(datan)
            ax.set_yticks(yticks)
            ax.set_yticklabels(['0', '5', '10', '15', '20'], fontsize='large')
    
            ax.set_xticks(xticks)
            ax.set_ylabel('Height [km]', fontsize='x-large')
            if a > 4:
                if f == 1:
                    ax.set_title('Satellite - %s' %mon.upper(), fontsize='x-large')
                else:
                    ax.set_title('Model - %s' %mon.upper(), fontsize='x-large')
            else:
                ax.set_title(mon.upper(), fontsize='x-large')
#             ax.text(0.01, 1.04, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            #: Ellipse Rectangle
            if a == 1:
                ellipse = Ellipse((147,44), 80, 50, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
            elif (a == 2) and (f == 1):
                ellipse = Ellipse((305,44), 40, 80, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
                ellipse = Ellipse((25,44), 40, 80, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
            elif (a == 3) and (f == 2):
                ellipse = Ellipse((90,44), 40, 80, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
            elif (a == 4) and (f == 2):
                ellipse = Ellipse((60,9), 100, 20, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
                ellipse = Ellipse((250,7), 80, 20, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
                ellipse = Ellipse((340,7), 80, 20, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
            elif a == 6:
                if f==1:
                    ellipse = Rectangle((0,42), 360, 24, fc='none', ec='green', lw=2)
                elif f==2:
                    ellipse = Rectangle((0,19), 1024, 14, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
            elif (a == 7):
                if f==1:
                    ellipse = Rectangle((0,15), 360, 22, fc='none', ec='green', lw=2)
                elif f==2:
                    ellipse = Rectangle((0,7), 1024, 11, fc='none', ec='green', lw=2)
                ax.add_patch(ellipse)
            elif (a == 8):
                if f==1:
                    ellipse = Ellipse((250,8), 90, 15, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
                    ellipse = Ellipse((340,8), 80, 15, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
                elif f==2:
                    ellipse = Ellipse((711,4), 256, 10, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
                    ellipse = Ellipse((967,4), 227, 10, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
            elif (a == 9):
                if f==1:
                    ellipse = Ellipse((305,44), 40, 80, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
                    ellipse = Ellipse((25,44), 40, 80, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
                elif f==2:
                    ellipse = Ellipse((868,22), 113, 40, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
                    ellipse = Ellipse((71,22), 113, 40, fc='none', ec='green', lw=2)
                    ax.add_patch(ellipse)
                    
            
                
            
            
            if f == num_fig:
                if use_datline_center:
                    ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'], fontsize='large')
                else:
                    ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'], fontsize='large')
                ax.set_xlabel('Longitude [deg]', fontsize='x-large')
                barticks = [valminmax[datan]*-1, valminmax[datan]*-0.75, valminmax[datan]*-0.5, valminmax[datan]*-0.25, 0, valminmax[datan]*0.25, valminmax[datan]*0.5, valminmax[datan]*0.75, valminmax[datan]]
                cbar_ax = fig.add_axes([0.2, 0.25, 0.6, 0.02])#[0.2, 0.04, 0.6, 0.01])
                cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)#, fontsize='large')  # @UnusedVariable
                cbar.set_label('CRH [K/day]', fontsize='x-large')
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
            if f == 1:
                if use_datline_center:
                    ax.text(0.94, 0.93, 'Atl', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.6, 0.93, 'Pac', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.22, 0.93, 'IO', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
#         plt.subplots_adjust(hspace=0.0001)
        figname = figname_se[datans] + '_lon'
        if not use_datline_center:
            figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
        if useClim:
            figname = figname + '_anom'
        
        figname = 'Kappa_plots/' + os.path.basename(figname) + '_ellipse-%d' %a
        
        fig.savefig(figname + '.png')
        fig.show()
    pdb.set_trace()

    
def plotLonSeason(rhd, stLat, valminmax, figname_se, useClim=False, use_datline_center=False):
#     xticks = {'sat': [1, 45, 90, 135, 180, 225, 270, 315, 358], 'hre': [1, 128, 256, 384, 512, 640, 768, 896, 1023],'lre': [1, 64, 128, 192, 256, 320, 384, 448, 511],'lr6': [1, 64, 128, 192, 256, 320, 384, 448, 511]}
#     yticks = {'sat': [1, 21, 42, 63, 83], 'hre': [1, 10, 20, 31, 41],'lre': [1, 10, 20, 31, 41],'lr6': [1, 10, 20, 31, 41]}
    figsize = (8,10)
    num_fig = 4
    for datan in stLat.keys():#['lre', 'sat', 'hre']:
        fig = plt.figure(figsize=figsize)
        f = 0
        stepInd = (stLat[datan] <= 30) & (stLat[datan] >= -30)
        for mon in 'djf', 'mam', 'jja', 'son':
            data_read = '%s_%s' %(mon, datan)
            f = f + 1
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rhdLat = np.nanmean(rhd[data_read]['crh'][:, stepInd, :], axis=1)  # @UndefinedVariable
            if datan == 'sat':
                rhdLat = rhdLat[0:86, :]
            warnings.resetwarnings()
            aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
            ax = fig.add_subplot(num_fig,1,f)
            im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax[datan] * -1, vmax=valminmax[datan])
            xticks, yticks = getLonTicks(datan)
            ax.set_yticks(yticks)
            ax.set_yticklabels(['0', '5', '10', '15', '20'])
    
            ax.set_xticks(xticks)
            ax.set_ylabel('Height [km]')
            ax.set_title(mon.upper())
            ax.text(0.01, 1.04, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            if f == num_fig:
                if use_datline_center:
                    ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'])
                else:
                    ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
                ax.set_xlabel('Longitude [deg]')
                barticks = [valminmax[datan]*-1, valminmax[datan]*-0.75, valminmax[datan]*-0.5, valminmax[datan]*-0.25, 0, valminmax[datan]*0.25, valminmax[datan]*0.5, valminmax[datan]*0.75, valminmax[datan]]
                cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.01])#[0.2, 0.04, 0.6, 0.01])
                cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)  # @UnusedVariable
                cbar.set_label('CRH [K/day]')
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
            if f == 1:
                if use_datline_center:
                    ax.text(0.94, 0.93, 'Atl', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.6, 0.93, 'Pac', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                    ax.text(0.22, 0.93, 'IO', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
        plt.subplots_adjust(hspace=0.0001)
        figname = figname_se[datan] + '_lon'
        if not use_datline_center:
            figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
        if useClim:
            figname = figname + '_anom'
        fig.savefig(figname + '.png')
        fig.show()
    pdb.set_trace()

def plotVertRHDSeason(rhd, height, figname_ver, useClim = False, months = ['djf', 'mam', 'jja', 'son']):
    colour = getCbColour()
    height_ver_sat = np.asarray(height['sat'][0:86])
    height_ver_hre = height['hre']
    height_ver_lre = height['lre']
    height_ver_lr6 = height['lr6']
    f = 0
    if len(months) == 4:
        fig = plt.figure(figsize = (12,16))
    elif len(months) == 3:
        fig = plt.figure(figsize = (12,13))
    elif len(months) == 1:
        fig = plt.figure(figsize = (12,5))
    for mon in months:
        for swlw in ['net', 'lw', 'sw']:
            f = f + 1
            ax = fig.add_subplot(len(months),3,f)
            if swlw == 'net':
                use_swlw = 'crh'
            else:
                use_swlw = swlw
        
            rhd_ver_sat = np.nanmean(rhd['%s_sat' %mon][use_swlw], axis=(1,2))[0:86]  # @UndefinedVariable
            rhd_ver_hre = np.nanmean(rhd['%s_hre' %mon][use_swlw], axis=(1,2))  # @UndefinedVariable
            rhd_ver_lre = np.nanmean(rhd['%s_lre' %mon][use_swlw], axis=(1,2))  # @UndefinedVariable
            rhd_ver_lr6 = np.nanmean(rhd['%s_lr6' %mon][use_swlw], axis=(1,2))  # @UndefinedVariable
            
            height_ver_sat = np.where(height_ver_sat > 20000, 20000, height_ver_sat)
            height_ver_hre = np.where(height_ver_hre > 20000, 20000, height_ver_hre)
            height_ver_lre = np.where(height_ver_lre > 20000, 20000, height_ver_lre)
            height_ver_lr6 = np.where(height_ver_lr6 > 20000, 20000, height_ver_lr6)

            ax.plot(rhd_ver_sat, height_ver_sat, color=colour['sat'], lw=2, label= 'SAT')
            ax.plot(rhd_ver_hre, height_ver_hre, color=colour['hre'], lw=2, label= 'E3PH')
            ax.plot(rhd_ver_lre, height_ver_lre, color=colour['lre'], lw=2, label= 'E3P')
            ax.plot(rhd_ver_lr6, height_ver_lr6, color=colour['lr6'], lw=2, label= 'E3')
            ax.vlines(0,0,height_ver_sat[-1],color='g', lw=0.5)
            
            std_ver_sat = np.nanstd(rhd['e_%s_sat' %mon][use_swlw + '_vert'], axis=1)[0:86]
            std_ver_hre = np.nanstd(rhd['e_%s_hre' %mon][use_swlw + '_vert'], axis=1)
            ax.plot((rhd_ver_sat - std_ver_sat), height_ver_sat, color=colour['sat'])
            ax.plot((rhd_ver_sat + std_ver_sat), height_ver_sat, color=colour['sat'])
            ax.plot((rhd_ver_hre - std_ver_hre), height_ver_hre, color=colour['hre'])
            ax.plot((rhd_ver_hre + std_ver_hre), height_ver_hre, color=colour['hre'])
            ax.fill_betweenx(height_ver_sat[3:], rhd_ver_sat[3:], rhd_ver_sat[3:] - std_ver_sat[3:], facecolor=colour['sat'], alpha=0.5)
            ax.fill_betweenx(height_ver_sat[3:], rhd_ver_sat[3:], rhd_ver_sat[3:] + std_ver_sat[3:], facecolor=colour['sat'], alpha=0.5)
            ax.fill_betweenx(height_ver_hre, rhd_ver_hre, rhd_ver_hre - std_ver_hre, facecolor=colour['hre'], alpha=0.5)
            ax.fill_betweenx(height_ver_hre, rhd_ver_hre, rhd_ver_hre + std_ver_hre, facecolor=colour['hre'], alpha=0.5)
            if f in [2, 5, 8, 11]:
                ax.legend(loc=1,prop={'size': 12})
            
            yticks_man = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
            if f in [1, 4, 7, 10]:
                ax.set_ylabel('Height [km]', fontsize='large')
                ylabel_man = (yticks_man / 1000).astype('str')#.astype('int').astype('str')
            else:
                ylabel_man = np.array(['',  '',  '',  '', '', '', '', '', ''])
            if f in [1, 2, 3]:
                ax.set_title(swlw.upper(), fontsize='large')
            if f in [len(months)*3-2, len(months)*3-1, len(months)*3]:
                ax.set_xlabel('Cloud Radiative Heating [K/day]', fontsize='large')
                xlabel_man = [-0.8, -0.4, 0, 0.4, 0.8, 1.2]
            else:
                xlabel_man = ['', '', '', '', '', '']
            if f in [3, 6, 9, 12]:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(mon.upper(), fontsize='large')
            ax.set_yticks(yticks_man)
            ax.set_yticklabels(ylabel_man)
            
            
            ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2])
            ax.set_xticklabels(xlabel_man)
            ax.text(0.02, 0.94, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            
#         ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    plt.tight_layout()
    figname = figname_ver + '_vert'
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    fig.show()

    pdb.set_trace()


def plotVertRHDSeason_area(rhd, height, lats, figname_ver, useClim = False, use_datline_center=False):
    colour = getCbColour()
    height_ver_sat = np.asarray(height['sat'][0:86])
    height_ver_hre = height['hre']
    height_ver_lre = height['lre']
    height_ver_lr6 = height['lr6']
    place = 'southa'
    fig = plt.figure(figsize = (12,16))
    fig.suptitle(place.upper())
    f = 0
    latmin = -10
    latmax = 10
    if use_datline_center:
        lonmin = {'southa': 270}
        lonmax = {'southa': 315}
    else:
        lonmin = {'southa': -90}
        lonmax = {'southa': -45}
    
    for mon in 'djf', 'mam', 'jja', 'son':
        for swlw in ['net', 'lw', 'sw']:
            f = f + 1
            ax = fig.add_subplot(4,3,f)
            if swlw == 'net':
                use_swlw = 'crh'
            else:
                use_swlw = swlw
            latmasksat = getLonLatMask(lats['sat'], latmin, latmax)
            latmaskhre = getLonLatMask(lats['hre'], latmin, latmax)
            latmasklre = getLonLatMask(lats['lre'], latmin, latmax)
            latmasklr6 = getLonLatMask(lats['lr6'], latmin, latmax)
            lonmasksat = getLonLatMask(rhd['%s_sat' %mon]['lon'], lonmin[place], lonmax[place])
            lonmaskhre = getLonLatMask(rhd['%s_hre' %mon]['lon'], lonmin[place], lonmax[place])
            lonmasklre = getLonLatMask(rhd['%s_lre' %mon]['lon'], lonmin[place], lonmax[place])
            lonmasklr6 = getLonLatMask(rhd['%s_lr6' %mon]['lon'], lonmin[place], lonmax[place])
        
            rhd_ver_sat = np.nanmean(rhd['%s_sat' %mon][use_swlw][:, latmasksat, :][:, :, lonmasksat], axis=(1,2))[0:86]  # @UndefinedVariable
            rhd_ver_hre = np.nanmean(rhd['%s_hre' %mon][use_swlw][:, latmaskhre, :][:, :, lonmaskhre], axis=(1,2))  # @UndefinedVariable
            rhd_ver_lre = np.nanmean(rhd['%s_lre' %mon][use_swlw][:, latmasklre, :][:, :, lonmasklre], axis=(1,2))  # @UndefinedVariable
            rhd_ver_lr6 = np.nanmean(rhd['%s_lr6' %mon][use_swlw][:, latmasklr6, :][:, :, lonmasklr6], axis=(1,2))  # @UndefinedVariable
            
            height_ver_sat = np.where(height_ver_sat > 20000, 20000, height_ver_sat)
            height_ver_hre = np.where(height_ver_hre > 20000, 20000, height_ver_hre)
            height_ver_lre = np.where(height_ver_lre > 20000, 20000, height_ver_lre)
            height_ver_lr6 = np.where(height_ver_lr6 > 20000, 20000, height_ver_lr6)
                
            ax.plot(rhd_ver_sat, height_ver_sat, color=colour['sat'], lw=2, label= 'SAT')
            ax.plot(rhd_ver_hre, height_ver_hre, color=colour['hre'], lw=2, label= 'HRE')
            ax.plot(rhd_ver_lre, height_ver_lre, color=colour['lre'], lw=2, label= 'SRE')
            ax.plot(rhd_ver_lr6, height_ver_lr6, color=colour['lr6'], lw=2, label= 'SR6')
            ax.vlines(0,0,height_ver_sat[-1],color='g', lw=0.5)
            if f in [2, 5, 8, 11]:
                ax.legend(loc=1,prop={'size': 12})
            
            yticks_man = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
            if f in [1, 4, 7, 10]:
                ax.set_ylabel('Height [km]')
                ylabel_man = (yticks_man / 1000).astype('str')#.astype('int').astype('str')
            else:
                ylabel_man = np.array(['',  '',  '',  '', '', '', '', '', ''])
            if f in [1, 2, 3]:
                ax.set_title(swlw.upper())
            if f in [10, 11, 12]:
                ax.set_xlabel('Cloud Radiative Heating [K/day]')
                xlabel_man = [-0.8, -0.4, 0, 0.4, 0.8, 1.2]
            else:
                xlabel_man = ['', '', '', '', '', '']
            if f in [3, 6, 9, 12]:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(mon.upper())
            ax.set_yticks(yticks_man)
            ax.set_yticklabels(ylabel_man)
            
            
            ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2])
            ax.set_xticklabels(xlabel_man)
            ax.text(0.02, 0.94, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            
#         ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    plt.tight_layout()
    figname = figname_ver + '_vert_%s' %place
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    fig.show()

    pdb.set_trace()



def plotCFSeason(cf, height, figname_cf, useClim = False):
    colour = getCbColour()
    fig = plt.figure(figsize = (8,8))
    f = 0
    for mon in 'djf', 'mam', 'jja', 'son':
        f = f + 1
        ax = fig.add_subplot(2, 2, f)
        for datan in ['lre', 'hre', 'sat']:
            cfd_ver = np.nanmean(cf['%s_%s' %(mon, datan)]['cfd'], axis=(1,2))  # @UndefinedVariable
            use_height = np.asarray(height[datan])

            if datan == 'sat':
                cfd_ver = cfd_ver[0:86]
                use_height = use_height[0:86]
            use_height = np.where(use_height > 20000, 20000, use_height)
            ax.plot(cfd_ver, use_height, colour[datan], label= datan)
        ax.legend(loc=1,prop={'size': 10})
        
        yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
        if f in [1, 3]:
            ax.set_ylabel('Height [km]')
            ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
        else:
            ylabel = np.array(['',  '',  '',  '', '', '', '', '', ''])
        if f in [3, 4]:
            ax.set_xlabel('Cloud Fraction')
            xlabel = [0, 0.05, 0.10, 0.15, 0.20]
        else:
            xlabel = ['', '', '', '', '']
        
        ax.set_title(mon.upper())
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabel)
    
        ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.set_xticklabels(xlabel)
        
        ax.text(0.02, 0.96, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
        
    plt.tight_layout()
    fig.show()
    figname = figname_cf
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    pdb.set_trace()

    
def plotVertRHD_CF_Season(rhd, height, figname_ver, useClim = False):
    colour = getCbColour()
    height_ver_sat = np.asarray(height['sat'][0:86])
    height_ver_sat = np.where(height_ver_sat > 20000, 20000, height_ver_sat)
    height_ver_lre = height['lre']
    height_ver_lre = np.where(height_ver_lre > 20000, 20000, height_ver_lre)
    height_ver_hre = height['hre']
    height_ver_hre = np.where(height_ver_hre > 20000, 20000, height_ver_hre)
    f = 0
    fig = plt.figure(figsize = (16,16))
    for mon in 'djf', 'mam', 'jja', 'son':
        for swlw in ['net', 'lw', 'sw', 'cf']:
            f = f + 1
            ax = fig.add_subplot(4,4,f)
            if swlw == 'net':
                use_swlw = 'crh'
            elif swlw == 'cf':
                use_swlw = 'cfd'
            else:
                use_swlw = swlw
            
            rhd_ver_sat = np.nanmean(rhd['%s_sat' %mon][use_swlw], axis=(1,2))[0:86]  # @UndefinedVariable
            rhd_ver_lre = np.nanmean(rhd['%s_lre' %mon][use_swlw], axis=(1,2))  # @UndefinedVariable
            rhd_ver_hre = np.nanmean(rhd['%s_hre' %mon][use_swlw], axis=(1,2))  # @UndefinedVariable
        
            ax.plot(rhd_ver_sat, height_ver_sat, color=colour['sat'], label= 'SAT')
            ax.plot(rhd_ver_lre, height_ver_lre, color=colour['lre'], label= 'LRE')
            ax.plot(rhd_ver_hre, height_ver_hre, color=colour['hre'], label= 'HRE')
            
            ax.legend(loc=1,prop={'size': 10})
            
            yticks_man = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
            if f in [1, 5, 9, 13]:
                ax.set_ylabel('Height [km]')
                ylabel_man = (yticks_man / 1000).astype('str')#.astype('int').astype('str')
            else:
                ylabel_man = np.array(['',  '',  '',  '', '', '', '', '', ''])
            if f in [1, 2, 3, 4]:
                ax.set_title(swlw.upper())
            if f in [13, 14, 15]:
                ax.set_xlabel('Cloud Radiative Heating [K/day]')
                xlabel_man = [-0.8, -0.4, 0, 0.4, 0.8, 1.2]
            elif f in [16]:
                ax.set_xlabel('Cloud Fraction')
                xlabel_man = [0, 0.05, 0.10, 0.15, 0.20]
            else:
                xlabel_man = ['', '', '', '', '', '']
            if f in [4, 8, 12, 16]:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(mon.upper())
                xticks_man = [0, 0.05, 0.10, 0.15, 0.20]
            else:
                ax.vlines(0,0,height_ver_sat[-1],color='g')
                xticks_man = [-0.8, -0.4, 0, 0.4, 0.8, 1.2]
            ax.set_yticks(yticks_man)
            ax.set_yticklabels(ylabel_man)
            ax.set_xticks(xticks_man)
            ax.set_xticklabels(xlabel_man)
            ax.text(0.02, 0.94, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
    plt.tight_layout()
    figname = figname_ver + '_vert'
    figname = figname.replace('rh_', 'rh_cf_')
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    fig.show()

    pdb.set_trace()


def plotWCSeason(wc, height, figname_wc, ticksen, useClim = False):
    height_ver_sat = np.asarray(height['sat'][0:86])
    height_ver_lre = height['lre']
    height_ver_hre = height['hre']
    colour = getCbColour()
    fig = plt.figure(figsize = (6,16))
    f = 0
    for mon in 'djf', 'mam', 'jja', 'son':
        for wcil in ['iwc', 'lwc']:
            f = f + 1
            ax = fig.add_subplot(4,2,f)
            if 'wc90' in figname_wc:
                wcil_sat = '90_%s' %wcil
            else:
                wcil_sat = wcil
            wc_sat_ver = np.nanmean(wc['%s_sat' %mon][wcil_sat], axis=(1,2))[0:86] / 1000.  # @UndefinedVariable
            wc_lre_ver = np.nanmean(wc['%s_lre' %mon][wcil], axis=(1,2))  # @UndefinedVariable
            wc_hre_ver = np.nanmean(wc['%s_hre' %mon][wcil], axis=(1,2))  # @UndefinedVariable

            if 'Norm' in figname_wc:
                wc_sat_ver = wc_sat_ver[0:64]
                wc_lre_ver = wc_lre_ver[0:32]
                wc_hre_ver = wc_hre_ver[0:32]
                
                height_ver_sat = height_ver_sat[0:64]
                height_ver_lre = height_ver_lre[0:32]
                height_ver_hre = height_ver_hre[0:32]
            
            height_ver_sat = np.where(height_ver_sat > 20000, 20000, height_ver_sat)
            height_ver_lre = np.where(height_ver_lre > 20000, 20000, height_ver_lre)
            height_ver_hre = np.where(height_ver_hre > 20000, 20000, height_ver_hre)
                
            ax.plot(wc_sat_ver, height_ver_sat, colour['sat'], label='SAT')
            ax.plot(wc_lre_ver, height_ver_lre, colour['lre'], label='LRE')
            ax.plot(wc_hre_ver, height_ver_hre, colour['hre'], label='HRE')
    
            ax.legend(loc=1,prop={'size': 10})
    #         ax.set_title('WC')
            if 'Norm' in figname_wc:
                yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.])#, 22500.])
            else:
                yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
            xticks = ticksen[wcil]
            if f in [1, 3, 5, 7]:
                ax.set_ylabel('Height [km]')
                ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
            else:
                ylabel = [''] * len(yticks)
            
            if f in [7, 8]:
                ax.set_xlabel('%s [g / m3]' %wcil.upper())
                xlabel = xticks.astype('str')
            else:
                xlabel = [''] * len(xticks)
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabel)
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabel)
            ax.text(0.02, 0.96, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
    plt.tight_layout()
    figname = figname_wc
    if useClim:
        figname = figname + '_anom'
    fig.show()
    
    fig.savefig(figname + '.png')
    pdb.set_trace()


def plotWC_CF_Season(wc, height, figname_wc, ticksen, useClim = False, months = ['djf', 'mam', 'jja', 'son']):
    
    colour = getCbColour()
    height_ver_sat = np.asarray(height['sat'][0:86])
    height_ver_hre = height['hre']
    height_ver_lre = height['lre']
    height_ver_lr6 = height['lr6']
    if len(months) == 4:
        fig = plt.figure(figsize = (12,16.5))
    elif len(months) == 3:
        fig = plt.figure(figsize = (12,13))
    elif len(months) == 1:
        fig = plt.figure(figsize = (12,5))
    f = 0
    for mon in months:
        for wcil in ['iwc', 'lwc', 'cfd']:
            f = f + 1
            ax = fig.add_subplot(len(months),3,f)
            if ('wc90' in figname_wc) and (wcil != 'cfd'):
                wcil_sat = '90_%s' %wcil
            else:
                wcil_sat = wcil
            
            if wcil == 'cfd':
                sat_div = 1
                wcil_mod = wcil
            else:
                sat_div = 1000.
                if 'Norm' in figname_wc:
                    wcil_mod = 'c' + wcil
                else:
                    wcil_mod = wcil
            wc_sat_ver = np.nanmean(wc['%s_sat' %mon][wcil_sat], axis=(1,2))[0:86] / sat_div  # @UndefinedVariable
            wc_hre_ver = np.nanmean(wc['%s_hre' %mon][wcil_mod], axis=(1,2))  # @UndefinedVariable
            wc_lre_ver = np.nanmean(wc['%s_lre' %mon][wcil_mod], axis=(1,2))  # @UndefinedVariable
            wc_lr6_ver = np.nanmean(wc['%s_lr6' %mon][wcil_mod], axis=(1,2))  # @UndefinedVariable
            
            if 'Norm' in figname_wc:
                h0ind = height_ver_hre > 17100
                wc_hre_ver[h0ind] = 0
                wc_lre_ver[h0ind] = 0
                wc_lr6_ver[h0ind] = 0    
                            
            height_ver_sat = np.where(height_ver_sat > 20000, 20000, height_ver_sat)
            height_ver_hre = np.where(height_ver_hre > 20000, 20000, height_ver_hre)
            height_ver_lre = np.where(height_ver_lre > 20000, 20000, height_ver_lre)
            height_ver_lr6 = np.where(height_ver_lr6 > 20000, 20000, height_ver_lr6)
                            
            ax.plot(wc_sat_ver, height_ver_sat, colour['sat'], lw=2, label='SAT')
            ax.plot(wc_hre_ver, height_ver_hre, colour['hre'], lw=2, label='E3PH')
            ax.plot(wc_lre_ver, height_ver_lre, colour['lre'], lw=2, label='E3P')
            ax.plot(wc_lr6_ver, height_ver_lr6, colour['lr6'], lw=2, label='E3')
            
            std_ver_sat = np.nanstd(wc['e_%s_sat' %mon][wcil_sat + '_vert'] / sat_div, axis=1)[0:86]
            std_ver_hre = np.nanstd(wc['e_%s_hre' %mon][wcil_mod + '_vert'], axis=1)
            #: Cant be less than 0
            min_std_sat = wc_sat_ver - std_ver_sat
            min_std_sat = np.where(min_std_sat<0, 0, min_std_sat)
            min_std_hre = wc_hre_ver - std_ver_hre
            min_std_hre = np.where(min_std_hre<0, 0, min_std_hre)
            ax.plot(min_std_sat, height_ver_sat, color=colour['sat'])
            ax.plot((wc_sat_ver + std_ver_sat), height_ver_sat, color=colour['sat'])
            ax.plot(min_std_hre, height_ver_hre, color=colour['hre'])
            ax.plot((wc_hre_ver + std_ver_hre), height_ver_hre, color=colour['hre'])
            ax.fill_betweenx(height_ver_sat[3:], wc_sat_ver[3:], min_std_sat[3:], facecolor=colour['sat'], alpha=0.5)
            ax.fill_betweenx(height_ver_sat[3:], wc_sat_ver[3:], wc_sat_ver[3:] + std_ver_sat[3:], facecolor=colour['sat'], alpha=0.5)
            ax.fill_betweenx(height_ver_hre, wc_hre_ver, min_std_hre, facecolor=colour['hre'], alpha=0.5)
            ax.fill_betweenx(height_ver_hre, wc_hre_ver, wc_hre_ver + std_ver_hre, facecolor=colour['hre'], alpha=0.5)
            
#             ax.plot((wc_sat_ver - wc['%s_sat' %mon][wcil_sat + '_std'][0:86]), height_ver_sat, color=colour['sat'])
#             ax.plot((wc_sat_ver + wc['%s_sat' %mon][wcil_sat + '_std'][0:86]), height_ver_sat, color=colour['sat'])
#             ax.fill_betweenx(height, clttmean[height], clttmax[height], facecolor=colour_face, alpha=0.5)
#             ax.fill_betweenx(height, clttmean[height], clttmin[height], facecolor=colour_face, alpha=0.5)
#             pdb.set_trace()
#             if f in [3, 6, 9, 12]:
            if f in [2, 5, 8, 11]:
                ax.legend(loc=1,prop={'size': 12})
#             else:
#                 ax.legend(loc=1,prop={'size': 10})
    #         ax.set_title('WC')
            yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
            xticks = ticksen[wcil]
            if f in [1, 4, 7, 10]:
                ax.set_ylabel('Height [km]', fontsize='large')
                ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
            else:
                ylabel = [''] * len(yticks)
            
            if f in [len(months) * 3-2, len(months) * 3-1, len(months) * 3]:
                if f == (len(months) * 3):
                    ax.set_xlabel('Cloud Fraction', fontsize='large')
                else:
                    ax.set_xlabel('%s [$g / m^3$]' %wcil.upper(), fontsize='large')
                xlabel = xticks.astype('str')
            else:
                xlabel = [''] * len(xticks)
            if f in [3, 6, 9, 12]:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(mon.upper(), fontsize='large')
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabel)
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabel)
            ax.text(0.02, 0.94, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
    plt.tight_layout()
    figname = figname_wc
    if useClim:
        figname = figname + '_anom'
    fig.show()
    fig.savefig(figname + '.png')
    pdb.set_trace()
    

def plotWC_CF_Season_area(wc, height, lats, figname_wc, ticksen, useClim = False, use_datline_center=False):
    height_ver_sat = np.asarray(height['sat'][0:86])
    height_ver_hre = height['hre']
    height_ver_lre = height['lre']
    height_ver_lr6 = height['lr6']
    colour = getCbColour()
    place = 'southa'
    fig = plt.figure(figsize = (12,16.5))
    fig.suptitle(place.upper())
    f = 0
    latmin = -10
    latmax = 10
    if use_datline_center:
        lonmin = {'southa': 270}
        lonmax = {'southa': 315}
    else:
        lonmin = {'southa': -90}
        lonmax = {'southa': -45}
    
    for mon in 'djf', 'mam', 'jja', 'son':
        for wcil in ['iwc', 'lwc', 'cfd']:
            f = f + 1
            ax = fig.add_subplot(4,3,f)
            if ('wc90' in figname_wc) and (wcil != 'cfd'):
                wcil_sat = '90_%s' %wcil
            else:
                wcil_sat = wcil
            
            if wcil == 'cfd':
                sat_div = 1
                wcil_mod = wcil
            else:
                sat_div = 1000.
                if 'Norm' in figname_wc:
                    wcil_mod = 'c' + wcil
                else:
                    wcil_mod = wcil
            latmasksat = getLonLatMask(lats['sat'], latmin, latmax)
            latmaskhre = getLonLatMask(lats['hre'], latmin, latmax)
            latmasklre = getLonLatMask(lats['lre'], latmin, latmax)
            latmasklr6 = getLonLatMask(lats['lr6'], latmin, latmax)
            lonmasksat = getLonLatMask(wc['%s_sat' %mon]['lon'], lonmin[place], lonmax[place])
            lonmaskhre = getLonLatMask(wc['%s_hre' %mon]['lon'], lonmin[place], lonmax[place])
            lonmasklre = getLonLatMask(wc['%s_lre' %mon]['lon'], lonmin[place], lonmax[place])
            lonmasklr6 = getLonLatMask(wc['%s_lr6' %mon]['lon'], lonmin[place], lonmax[place])
            
            wc_sat_ver = np.nanmean(wc['%s_sat' %mon][wcil_sat][:, latmasksat, :][:, :, lonmasksat], axis=(1,2))[0:86] / sat_div  # @UndefinedVariable
            wc_hre_ver = np.nanmean(wc['%s_hre' %mon][wcil_mod][:, latmaskhre, :][:, :, lonmaskhre], axis=(1,2))  # @UndefinedVariable
            wc_lre_ver = np.nanmean(wc['%s_lre' %mon][wcil_mod][:, latmasklre, :][:, :, lonmasklre], axis=(1,2))  # @UndefinedVariable
            wc_lr6_ver = np.nanmean(wc['%s_lr6' %mon][wcil_mod][:, latmasklr6, :][:, :, lonmasklr6], axis=(1,2))  # @UndefinedVariable
            
            if 'Norm' in figname_wc:
                h0ind = height_ver_hre > 17100
                wc_hre_ver[h0ind] = 0
                wc_lre_ver[h0ind] = 0
                wc_lr6_ver[h0ind] = 0    
                            
            height_ver_sat = np.where(height_ver_sat > 20000, 20000, height_ver_sat)
            height_ver_hre = np.where(height_ver_hre > 20000, 20000, height_ver_hre)
            height_ver_lre = np.where(height_ver_lre > 20000, 20000, height_ver_lre)
            height_ver_lr6 = np.where(height_ver_lr6 > 20000, 20000, height_ver_lr6)
                            
            ax.plot(wc_sat_ver, height_ver_sat, colour['sat'], lw=2, label='SAT')
            ax.plot(wc_hre_ver, height_ver_hre, colour['hre'], lw=2, label='HRE')
            ax.plot(wc_lre_ver, height_ver_lre, colour['lre'], lw=2, label='SRE')
            ax.plot(wc_lr6_ver, height_ver_lr6, colour['lr6'], lw=2, label='SR6')
            
#             if f in [3, 6, 9, 12]:
            if f in [2, 5, 8, 11]:
                ax.legend(loc=1,prop={'size': 12})
#             else:
#                 ax.legend(loc=1,prop={'size': 10})
    #         ax.set_title('WC')
            yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
            xticks = ticksen[wcil]
            if f in [1, 4, 7, 10]:
                ax.set_ylabel('Height [km]')
                ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
            else:
                ylabel = [''] * len(yticks)
            
            if f in [10, 11, 12]:
                if f == 12:
                    ax.set_xlabel('Cloud Fraction')
                else:
                    ax.set_xlabel('%s [g / m3]' %wcil.upper())
                xlabel = xticks.astype('str')
            else:
                xlabel = [''] * len(xticks)
            if f in [3, 6, 9, 12]:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(mon.upper())
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabel)
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabel)
            ax.text(0.02, 0.94, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
    plt.tight_layout()
    figname = figname_wc + '_%s' %place
    if useClim:
        figname = figname + '_anom'
    fig.show()
    fig.savefig(figname + '.png')
    pdb.set_trace()


def plotWC_CF_Enso(wc, height, lats, figname_org, ticksen, useClim, use_datline_center=False):
    height_ver_sat = np.asarray(height['sat'][0:86])
    height_ver_hre = height['hre']
    height_ver_lre = height['lre']
    height_ver_lr6 = height['lr6']
    colour = getCbColour()
    latmin = -5
    latmax = 5
    if use_datline_center:
        lonmin = {'nino4': 160, 'nino3': 210}
        lonmax = {'nino4': 210, 'nino3': 270}
    else:
        lonmin = {'nino4': 160, 'nino3': -150}
        lonmax = {'nino4': -150, 'nino3': -90}
    for place in ['nino4', 'nino3']:
        fig = plt.figure(figsize = (12,8.5))
        f = 0
        for mon in ['ensoP', 'ensoN']:
            latmasksat = getLonLatMask(lats['sat'], latmin, latmax)
            latmaskhre = getLonLatMask(lats['hre'], latmin, latmax)
            latmasklre = getLonLatMask(lats['lre'], latmin, latmax)
            latmasklr6 = getLonLatMask(lats['lr6'], latmin, latmax)
            lonmasksat = getLonLatMask(wc['%s_sat' %mon]['lon'], lonmin[place], lonmax[place])
            lonmaskhre = getLonLatMask(wc['%s_hre' %mon]['lon'], lonmin[place], lonmax[place])
            lonmasklre = getLonLatMask(wc['%s_lre' %mon]['lon'], lonmin[place], lonmax[place])
            lonmasklr6 = getLonLatMask(wc['%s_lr6' %mon]['lon'], lonmin[place], lonmax[place])
            for wcil in ['iwc', 'lwc', 'cfd']:
                f = f + 1
                ax = fig.add_subplot(2,3,f)
                if ('wc90' in figname_org) and (wcil != 'cfd'):
                    wcil_sat = '90_%s' %wcil
                else:
                    wcil_sat = wcil
                if wcil == 'cfd':
                    sat_div = 1
                    wcil_mod = wcil
                else:
                    sat_div = 1000.
                    if 'Norm' in figname_org:
                        wcil_mod = 'c' + wcil
                    else:
                        wcil_mod = wcil
                wc_sat_ver = np.nanmean(wc['%s_sat' %mon][wcil_sat][:, latmasksat, :][:, :, lonmasksat], axis=(1,2))[0:86] / sat_div  # @UndefinedVariable
                wc_hre_ver = np.nanmean(wc['%s_hre' %mon][wcil_mod][:, latmaskhre, :][:, :, lonmaskhre], axis=(1,2))  # @UndefinedVariable
                wc_lre_ver = np.nanmean(wc['%s_lre' %mon][wcil_mod][:, latmasklre, :][:, :, lonmasklre], axis=(1,2))  # @UndefinedVariable
                wc_lr6_ver = np.nanmean(wc['%s_lr6' %mon][wcil_mod][:, latmasklr6, :][:, :, lonmasklr6], axis=(1,2))  # @UndefinedVariable
            
                if 'Norm' in figname_org:
                    h0ind = height_ver_hre > 17100
                    wc_hre_ver[h0ind] = 0
                    wc_lre_ver[h0ind] = 0
                    wc_lr6_ver[h0ind] = 0    
                
                if place in ['nino3']:
                    fi = 1
                elif place in ['nino4']:
                    fi = 2
                
                std_ver_sat = np.nanstd(wc['e_%s_sat' %mon][wcil_sat + '_vert'][fi,:,:] / sat_div, axis=1)[0:86]  # @UndefinedVariable
                std_ver_hre = np.nanstd(wc['e_%s_hre' %mon][wcil_mod + '_vert'][fi,:,:], axis=1)  # @UndefinedVariable
                min_std_sat = wc_sat_ver - std_ver_sat
                min_std_hre = wc_hre_ver - std_ver_hre
#                 min_std_sat = np.where(min_std_sat<0, 0, min_std_sat)
#                 min_std_hre = np.where(min_std_hre<0, 0, min_std_hre)
                
                
                height_ver_sat = np.where(height_ver_sat >= 20000, 20000, height_ver_sat)
                height_ver_hre = np.where(height_ver_hre >= 20000, 20000, height_ver_hre)
                height_ver_lre = np.where(height_ver_lre >= 20000, 20000, height_ver_lre)
                height_ver_lr6 = np.where(height_ver_lr6 >= 20000, 20000, height_ver_lr6)
                            
                ax.plot(wc_sat_ver, height_ver_sat, colour['sat'], lw=2, label='SAT')
                ax.plot(min_std_sat, height_ver_sat, color=colour['sat'])
                ax.plot((wc_sat_ver + std_ver_sat), height_ver_sat, color=colour['sat'])
                ax.fill_betweenx(height_ver_sat[3:], wc_sat_ver[3:], min_std_sat[3:], facecolor=colour['sat'], alpha=0.5)
                ax.fill_betweenx(height_ver_sat[3:], wc_sat_ver[3:], wc_sat_ver[3:] + std_ver_sat[3:], facecolor=colour['sat'], alpha=0.5)

                ax.plot(wc_hre_ver, height_ver_hre, colour['hre'], lw=2, label='E3PH')
                ax.plot(min_std_hre, height_ver_hre, color=colour['hre'])
                ax.plot((wc_hre_ver + std_ver_hre), height_ver_hre, color=colour['hre'])
                ax.fill_betweenx(height_ver_hre, wc_hre_ver, min_std_hre, facecolor=colour['hre'], alpha=0.5)
                ax.fill_betweenx(height_ver_hre, wc_hre_ver, wc_hre_ver + std_ver_hre, facecolor=colour['hre'], alpha=0.5)

                ax.plot(wc_lre_ver, height_ver_lre, colour['lre'], lw=2, label='E3P')
                ax.plot(wc_lr6_ver, height_ver_lr6, colour['lr6'], lw=2, label='E3')

                if useClim:
                    ax.vlines(0,0,height_ver_sat[-1],color='g', lw=0.5)
                if f in [2, 5, 8, 11]:
                    ax.legend(loc=1,prop={'size': 12})
                yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
                xticks = ticksen[wcil + '_' + place]
                if f in [1, 4, 7, 10]:
                    ax.set_ylabel('Height [km]', fontsize='large')
                    ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
                else:
                    ylabel = [''] * len(yticks)
                
                if f in [4, 5, 6]:
                    if f == 6:
                        ax.set_xlabel('Cloud Fraction', fontsize='large')
                    else:
                        ax.set_xlabel('%s [$g / m^3$]' %wcil.upper(), fontsize='large')
                    xlabel = xticks.astype('str')
                else:
                    xlabel = [''] * len(xticks)
                if f in [3, 6, 9, 12]:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(mon.upper() + '\n' + place.title(), fontsize='large')
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabel)
                
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabel)
                ax.text(0.02, 0.94, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
        plt.tight_layout()
        figname = figname_org + '_%s' %place
        if useClim:
            figname = figname + '_anom'
        fig.show()
        fig.savefig(figname + '.png')
    pdb.set_trace()

def plotVertRHD_Enso(rhd, height, lats, figname_ver, useClim, use_datline_center=False):
    ticksen = {'net_nino3': np.array([-0.8, -0.4, 0, 0.4, 0.8]), \
               'lw_nino3': np.array([-1.2, -0.8, -0.4, 0, 0.4, 0.8]), \
               'sw_nino3': np.array([-0.8, -0.4, 0, 0.4, 0.8]), \
               'net_nino4': np.array([-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0]), \
               'lw_nino4': np.array([-0.8, -0.4, 0, 0.4, 0.8]), \
               'sw_nino4': np.array([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])}
    
    colour = getCbColour()
    height_ver_sat = np.asarray(height['sat'])[0:86]
    height_ver_hre = height['hre']
    height_ver_lre = height['lre']
    height_ver_lr6 = height['lr6']
    latmin = -5
    latmax = 5
    if use_datline_center:
        lonmin = {'nino4': 160, 'nino3': 210}
        lonmax = {'nino4': 210, 'nino3': 270}
    else:
        lonmin = {'nino4': 160, 'nino3': -150}
        lonmax = {'nino4': -150, 'nino3': -90}
    for place in ['nino4', 'nino3']:
        f = 0
        fig = plt.figure(figsize = (12,8.5))
        for mon in ['ensoP', 'ensoN']:
            
            latmasksat = getLonLatMask(lats['sat'], latmin, latmax)
            latmaskhre = getLonLatMask(lats['hre'], latmin, latmax)
            latmasklre = getLonLatMask(lats['lre'], latmin, latmax)
            latmasklr6 = getLonLatMask(lats['lr6'], latmin, latmax)
            lonmasksat = getLonLatMask(rhd['%s_sat' %mon]['lon'], lonmin[place], lonmax[place])
            lonmaskhre = getLonLatMask(rhd['%s_hre' %mon]['lon'], lonmin[place], lonmax[place])
            lonmasklre = getLonLatMask(rhd['%s_lre' %mon]['lon'], lonmin[place], lonmax[place])
            lonmasklr6 = getLonLatMask(rhd['%s_lr6' %mon]['lon'], lonmin[place], lonmax[place])
            for swlw in ['net', 'lw', 'sw']:
                f = f + 1
                ax = fig.add_subplot(2,3,f)
                if swlw == 'net':
                    use_swlw = 'crh'
                else:
                    use_swlw = swlw
                
                rhd_ver_sat = np.nanmean(rhd['%s_sat' %mon][use_swlw][:, latmasksat, :][:, :, lonmasksat], axis=(1,2))[0:86]  # @UndefinedVariable
                rhd_ver_hre = np.nanmean(rhd['%s_hre' %mon][use_swlw][:, latmaskhre, :][:, :, lonmaskhre], axis=(1,2))  # @UndefinedVariable
                rhd_ver_lre = np.nanmean(rhd['%s_lre' %mon][use_swlw][:, latmasklre, :][:, :, lonmasklre], axis=(1,2))  # @UndefinedVariable
                rhd_ver_lr6 = np.nanmean(rhd['%s_lr6' %mon][use_swlw][:, latmasklr6, :][:, :, lonmasklr6], axis=(1,2))  # @UndefinedVariable

                height_ver_sat = np.where(height_ver_sat > 20000, 20000, height_ver_sat)
                height_ver_hre = np.where(height_ver_hre > 20000, 20000, height_ver_hre)
                height_ver_lre = np.where(height_ver_lre > 20000, 20000, height_ver_lre)
                height_ver_lr6 = np.where(height_ver_lr6 > 20000, 20000, height_ver_lr6)

                ax.plot(rhd_ver_sat, height_ver_sat, color=colour['sat'], lw=2, label= 'SAT')
                ax.plot(rhd_ver_hre, height_ver_hre, color=colour['hre'], lw=2, label= 'E3PH')
                ax.plot(rhd_ver_lre, height_ver_lre, color=colour['lre'], lw=2, label= 'E3P')
                ax.plot(rhd_ver_lr6, height_ver_lr6, color=colour['lr6'], lw=2, label= 'E3')
                ax.vlines(0,0,height_ver_sat[-1],color='g', lw=0.5)
                
                #: Vad ar fi = 0?
                if place in ['nino3']:
                    fi = 1
                elif place in ['nino4']:
                    fi = 2

                std_ver_sat = np.nanstd(rhd['e_%s_sat' %mon][use_swlw + '_vert'][fi,:,:], axis=1)[0:86]  # @UndefinedVariable
                std_ver_hre = np.nanstd(rhd['e_%s_hre' %mon][use_swlw + '_vert'][fi,:,:], axis=1)  # @UndefinedVariable
                
                ax.plot((rhd_ver_sat - std_ver_sat), height_ver_sat, color=colour['sat'])
                ax.plot((rhd_ver_sat + std_ver_sat), height_ver_sat, color=colour['sat'])
                ax.plot((rhd_ver_hre - std_ver_hre), height_ver_hre, color=colour['hre'])
                ax.plot((rhd_ver_hre + std_ver_hre), height_ver_hre, color=colour['hre'])
                ax.fill_betweenx(height_ver_sat[3:], rhd_ver_sat[3:], rhd_ver_sat[3:] - std_ver_sat[3:], facecolor=colour['sat'], alpha=0.5)
                ax.fill_betweenx(height_ver_sat[3:], rhd_ver_sat[3:], rhd_ver_sat[3:] + std_ver_sat[3:], facecolor=colour['sat'], alpha=0.5)
                ax.fill_betweenx(height_ver_hre, rhd_ver_hre, rhd_ver_hre - std_ver_hre, facecolor=colour['hre'], alpha=0.5)
                ax.fill_betweenx(height_ver_hre, rhd_ver_hre, rhd_ver_hre + std_ver_hre, facecolor=colour['hre'], alpha=0.5)
                
                if f in [2, 5, 8, 11]:
                    ax.legend(loc=1,prop={'size': 12})
                
                yticks_man = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
                xticks = ticksen[swlw + '_' + place]
                if f in [1, 4, 7, 10]:
                    ax.set_ylabel('Height [km]', fontsize='large')
                    ylabel_man = (yticks_man / 1000).astype('str')
                else:
                    ylabel_man = np.array(['',  '',  '',  '', '', '', '', '', ''])
                if f in [1, 2, 3]:
                    ax.set_title(swlw.upper(), fontsize='large')
                if f in [4, 5, 6]:
                    ax.set_xlabel('Cloud Radiative Heating [K/day]', fontsize='large')
                    xlabel = xticks.astype('str')
#                     xlabel_man = [-0.8, -0.4, 0, 0.4, 0.8, 1.2]
                else:
#                     xlabel_man = ['', '', '', '', '', '']
                    xlabel = [''] * len(xticks)
                if f in [3, 6, 9, 12]:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(mon.upper() + '\n' + place.title(), fontsize='large')
                ax.set_yticks(yticks_man)
                ax.set_yticklabels(ylabel_man)
                
                
#                 ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2])
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabel)
                ax.text(0.02, 0.94, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
            
        plt.tight_layout()
        figname = figname_ver + '_%s' %place
        figname = figname + '_vert'
        if useClim:
            figname = figname + '_anom'
        fig.savefig(figname + '.png')
        fig.show()
    
    pdb.set_trace()
 
    
def plotVertRHD(rhd_swlw, height, figname_ver, useClim = False):
    colour = getCbColour()
    f = 0
    fig = plt.figure(figsize = (12,5))
    for swlw in ['net', 'lw', 'sw']:
        f = f + 1
        ax = fig.add_subplot(1,3,f)
        rhd_ver_sat = np.nanmean(rhd_swlw[swlw]['sat'], axis=(1,2))[0:86]  # @UndefinedVariable
        rhd_ver_lre = np.nanmean(rhd_swlw[swlw]['lre'], axis=(1,2))  # @UndefinedVariable
        rhd_ver_hre = np.nanmean(rhd_swlw[swlw]['hre'], axis=(1,2))  # @UndefinedVariable
        height_ver_sat = height['sat'][0:86]
        height_ver_lre = height['lre']
        height_ver_hre = height['hre']
        
        ax.plot(rhd_ver_sat, height_ver_sat, color=colour['sat'], label= 'SAT')
        ax.plot(rhd_ver_lre, height_ver_lre, color=colour['lre'], label= 'LRE')
        ax.plot(rhd_ver_hre, height_ver_hre, color=colour['hre'], label= 'HRE')
        
        ax.legend(loc=1,prop={'size': 10})
        
        ax.set_xlabel('Cloud Radiative Heating [K/day]')
        if f == 1:
            ax.set_ylabel('Height [km]')
        ax.set_title(swlw.upper())

        yticks_man = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
        ylabel_man = (yticks_man / 1000).astype('str')#.astype('int').astype('str')
        ax.set_yticks(yticks_man)
        ax.set_yticklabels(ylabel_man)
        
        ax.vlines(0,0,height_ver_sat[-1],color='g')
        ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2])
#         ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    plt.tight_layout()
    figname = figname_ver + '_vert'
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    fig.show()

    pdb.set_trace()


def plotCF(cf, height, figname_cf, useClim = False):
    colour = getCbColour()
    fig = plt.figure()
    f = -1
    for datan in ['lre', 'hre', 'sat']:
        f = f + 1
        ax = fig.add_subplot(111)
        cfd_ver = np.nanmean(cf[datan], axis=(1,2))  # @UndefinedVariable
        
        use_height = np.asarray(height[datan])
        if datan == 'sat':
            cfd_ver = cfd_ver[0:86]
            use_height = use_height[0:86]
        use_height = np.where(use_height > 20000, 20000, use_height)
        ax.plot(cfd_ver, use_height, colour[datan], label= datan)
    ax.legend()
    ax.set_xlabel('Cloud Fraction')
    ax.set_ylabel('Height [km]')
    ax.set_title('CF')
    
    yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
    ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabel)

    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
        
    plt.tight_layout()
    fig.show()
    figname = figname_cf
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    pdb.set_trace()


def plotWC(wc, height, figname_wc, ticksen, useClim = False):
    colour = getCbColour()
    fig = plt.figure()
    f = 0
    for wcil in ['iwc', 'lwc']:
        f = f + 1
        l = -1
        ax = fig.add_subplot(1,2,f)
        for datan in ['sat', 'hre', 'lre']:
            l = l + 1
            wc_ver = np.nanmean(wc['%s_%s' %(datan, wcil)], axis=(1,2))  # @UndefinedVariable
#             (np.nanmean(wc['%s_%s' %('sat', 'iwc')], axis=(1,2))[0:86][::2] / 1000. ) / np.nanmean(wc['%s_%s' %('lre', 'iwc')], axis=(1,2))
            use_height = np.asarray(height[datan])
            if datan == 'sat':
                wc_ver = wc_ver[0:86] / 1000.
                use_height = use_height[0:86]
            if 'Norm' in figname_wc:
                if datan == 'sat':
                    wc_ver = wc_ver[0:64]
                    use_height = use_height[0:64]
                else:
                    wc_ver = wc_ver[0:32]
                    use_height = use_height[0:32]
            use_height = np.where(use_height > 20000, 20000, use_height)
            ax.plot(wc_ver, use_height, colour[datan], label= datan.upper())

        ax.legend(loc=1,prop={'size': 10})
        ax.set_xlabel('%s [g / m3]' %wcil.upper())
#         ax.set_title('WC')
        if f == 1:
            ax.set_ylabel('Height [km]')
        
        yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
        if 'Norm' in figname_wc:
            yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.])#, 22500.])
        ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabel)
        
        ax.set_xticks(ticksen[wcil])
        ax.text(0.02, 0.96, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
    plt.tight_layout()
    figname = figname_wc
    if useClim:
        figname = figname + '_anom'
    fig.show()
    
    fig.savefig(figname + '.png')
    pdb.set_trace()


def reshapeRhdLat(trhd, brhd, extraLon):
    #: Height, sat to model
    if (trhd.shape[0] == 125) and (brhd.shape[0] == 43):
        trhd = (trhd[1:87, :][::2, :] + trhd[1:87, :][1::2, :])/2
    elif (brhd.shape[0] == 125) and (trhd.shape[0] == 43):
        brhd = (brhd[1:87, :][::2, :] + brhd[1:87, :][1::2, :])/2
    #: Lon, sat to model
    if (trhd.shape[1] == 360) and (brhd.shape[1] != 360):
        if extraLon[0].shape[0] == 360:
            sat_extralon = extraLon[0]
            mod_extraLon = extraLon[1]
        else:
            sat_extralon = extraLon[1]
            mod_extraLon = extraLon[0]
        ind = np.searchsorted(sat_extralon, mod_extraLon)
        
        if np.abs(trhd.shape[-1] - brhd.shape[-1]) <= 1:
            brhd_e = np.zeros(trhd.shape[0:-1] + (brhd.shape[-1],))
        else:
            brhd_e = np.zeros(trhd.shape)
        for i in range(trhd.shape[1]):
            if i == 0:
                brhd_e[:, i] = np.mean(brhd[:,np.where((ind == i) | (ind == 360))[0]], axis=1)
            else:
                brhd_e[:, i] = np.mean(brhd[:,np.where(ind == i)[0]], axis=1)
        brhd = brhd_e
    elif (brhd.shape[1] == 360) and (trhd.shape[1] != 360):
        if extraLon[0].shape[0] == 360:
            sat_extralon = extraLon[0]
            mod_extraLon = extraLon[1]
        else:
            sat_extralon = extraLon[1]
            mod_extraLon = extraLon[0]
        ind = np.searchsorted(sat_extralon, mod_extraLon)
        
        if np.abs(brhd.shape[-1] - trhd.shape[-1]) <= 1:
            trhd_e = np.zeros(brhd.shape[0:-1] + (trhd.shape[-1],))
        else:
            trhd_e = np.zeros(brhd.shape)
        for i in range(brhd.shape[1]):
            if i == 0:
                trhd_e[:, i] = np.mean(trhd[:,np.where((ind == i) | (ind == 360))[0]], axis=1)
            else:
                trhd_e[:, i] = np.mean(trhd[:,np.where(ind == i)[0]], axis=1)
        trhd = trhd_e
    #: Lon, hre to lre (lc6)
    if (trhd.shape[1] == 1024) and (brhd.shape[1] == 512):
        trhd = (trhd[:, ::2] + trhd[:, 1::2]) / 2
    elif (brhd.shape[1] == 1024) and (trhd.shape[1] == 512):
        brhd = (brhd[:, ::2] + brhd[:, 1::2]) / 2
    return trhd, brhd

def calculateLonDiff(trhd, brhd, extraLon=None):
    trhd_rs, brhd_rs = reshapeRhdLat(trhd, brhd, extraLon)
    retv = trhd_rs - brhd_rs
    return retv


def plotLonTOA_diff(rhd, valminmax, figname_st, useClim=False, use_datline_center=False):
    xticks = {'sat': [1, 45, 90, 135, 180, 225, 270, 315, 358],'lre': [1, 64, 128, 192, 256, 320, 384, 448, 511], 'hre': [1, 128, 256, 384, 512, 640, 768, 896, 1023]}
    yticks = {'lre': [1, 22, 43, 65, 85]}
#     {'sat': ,'lre': }
    fig = plt.figure()
    f = 0
    
    f = f + 1
#             endLat = stepL + step
#             stepInd = (stLat[datan] >= stepL) & (stLat[datan] < endLat)
    warnings.simplefilter("ignore", category=RuntimeWarning)
    hr_rhdLat = np.nanmean(rhd['hre'], axis=0)  # @UndefinedVariable
    lr_rhdLat = np.nanmean(rhd['lre'], axis=0)  # @UndefinedVariable
    hr_lres = (hr_rhdLat[::2, ::2] + hr_rhdLat[1::2, 1::2]) / 2
    rhdLat = hr_lres - lr_rhdLat[1:, :]
    warnings.resetwarnings()
    aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
    ax = fig.add_subplot(111)
    im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax * -1, vmax=valminmax)
    ax.set_yticks(yticks['lre'])
    ax.set_yticklabels(['-30', '-15', '0', '15', '30'])

    ax.set_xticks(xticks['lre'])
    ax.set_title('High - Low')
    ax.set_ylabel('Latitude [deg]')

    if use_datline_center:
        ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'])
    else:
        ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
    ax.set_xlabel('Longitude [deg]')
    barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
    cbar_ax = fig.add_axes([0.2, 0.25, 0.6, 0.01])
    cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)  # @UnusedVariable
#         else:
#             ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
    plt.tight_layout()
    figname = figname_st + '_top'
    if not use_datline_center:
        figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
#         fig.savefig('test.png')
    fig.show()
    pdb.set_trace()


def plotLonStep_diff(rhd, step, stLat, valminmax, figname_st, top_rad, bot_rad, useClim=False, use_datline_center=False, useSMA = 0, extraLon = None):
#     xticks = {'sat': [1, 45, 90, 135, 180, 225, 270, 315, 358], 'hre': [1, 128, 256, 384, 512, 640, 768, 896, 1023], 'lre': [1, 64, 128, 192, 256, 320, 384, 448, 511], 'lr6': [1, 64, 128, 192, 256, 320, 384, 448, 511]}
#     yticks = {'sat': [1, 21, 42, 63, 83], 'hre': [1, 10, 20, 31, 41], 'lre': [1, 10, 20, 31, 41], 'lr6': [1, 10, 20, 31, 41]}
#     {'sat': ,'lre': }
    start_lat = -15
    num_fig = (abs(start_lat) * 2) / step
    if num_fig == 2:
        num_fig = num_fig + 1
    
    if num_fig == 6:
        figsize = (8,24)
    elif num_fig == 4:
        figsize = (8,10)
    elif num_fig == 3:
        figsize = (8,7)
    elif num_fig == 1:
        figsize = (8,4)
    else:
        figsize = (8,8)
        
    fig = plt.figure(figsize=figsize)    
    f = 0
    for stepL in range(start_lat,abs(start_lat),step)[::-1]:
        f = f + 1
        endLat = stepL + step
        tr_stepInd = getLonLatMask(stLat[top_rad], stepL, endLat)
        br_stepInd = getLonLatMask(stLat[bot_rad], stepL, endLat)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tr_rhdLat = np.nanmean(rhd[top_rad][:,tr_stepInd,:], axis=1)  # @UndefinedVariable
        br_rhdLat = np.nanmean(rhd[bot_rad][:,br_stepInd,:], axis=1)  # @UndefinedVariable
        if (top_rad == 'sat') and (useSMA != 0):
            tr_rhdLat = calculateLeapingMean2D(tr_rhdLat, useSMA)
        if (bot_rad == 'sat') and (useSMA != 0):
            br_rhdLat = calculateLeapingMean2D(br_rhdLat, useSMA)
        rhdLat = calculateLonDiff(tr_rhdLat, br_rhdLat, extraLon=extraLon)
        warnings.resetwarnings()

        aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
        ax = fig.add_subplot(num_fig,1,f)
        im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax['lre'] * -1, vmax=valminmax['lre'])
        
        if (stepL == -30) and (endLat == 30):
            fi = 1
        if (stepL == -30) and (endLat == -15):
            fi = 1
        elif (stepL == -15) and (endLat == 0):
            fi = 2
        elif (stepL == 0) and (endLat == 15):
            fi = 3
        elif (stepL == 15) and (endLat == 30):
            fi = 4
        elif (stepL == -15) and (endLat == 15):
            fi = 5
        trhd_rs, brhd_rs = reshapeRhdLat(rhd['e_%s' %(top_rad)][fi,:,:,:], rhd['e_%s' %(bot_rad)][fi,:,:,:], extraLon)
        
        #: Calculate t-test
        t1, p1 = ttest_ind(trhd_rs, brhd_rs, axis=2)
        ax.contourf(p1<=0.05, 1, origin='lower', hatches=['', '.'], alpha=0)

        xticks, yticks = getLonTicks(top_rad, bot_rad, tr_rhdLat, br_rhdLat)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels(['0', '5', '10', '15', '20'])
        ax.set_ylabel('Height [km]')
#         ax.set_title('Latitude, %i - %i' %(stepL, endLat))
        if 'ensoP' in figname_st:
            #: Sat - hre
            if ('sat' in [top_rad, bot_rad]) and ('hre' in [top_rad, bot_rad]):
                if num_fig in [1]:
                    start_letter = 99 #: c
                elif num_fig in [2, 3]: #: 3 is some times used as 2
                    start_letter = 101 #: e
                elif num_fig in [4]:
                    start_letter = 105 #: i
            else:
                start_letter = 97 #: a
        elif 'ensoN' in figname_st:
            #: Sat - hre
            if ('sat' in [top_rad, bot_rad]) and ('hre' in [top_rad, bot_rad]):
                if num_fig in [1]:
                    start_letter = 100 #: d
                elif num_fig in [2, 3]: #: 3 is some times used as 2
                    start_letter = 103 #: e
                elif num_fig in [4]:
                    start_letter = 109 #: i
            else:
                if num_fig in [1]:
                    start_letter = 98 #: d
                elif num_fig in [2, 3]: #: 3 is some times used as 2
                    start_letter = 99 #: e
                elif num_fig in [4]:
                    start_letter = 101 #: i
        ax.text(0.01, 1.04, chr((f-1) + start_letter), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
        
        if stepL == start_lat:
            if use_datline_center:
                ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'])
            else:
                ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
            ax.set_xlabel('Longitude [deg]')
            barticks = [valminmax['lre']*-1, valminmax['lre']*-0.75, valminmax['lre']*-0.5, valminmax['lre']*-0.25, 0, valminmax['lre']*0.25, valminmax['lre']*0.5, valminmax['lre']*0.75, valminmax['lre']]
            if num_fig == 1:
                cbar_ax = fig.add_axes([0.2, 0.33, 0.6, 0.01])#[0.2, 0.26, 0.6, 0.01])
            else:
                cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.01])
            cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax)#, ticks=barticks)  # @UnusedVariable
            cbar.set_label('CRH [K/day]')
        else:
            ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
        if f == 1:
            if use_datline_center:
                ax.text(0.94, 0.93, 'Atl', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                ax.text(0.6, 0.93, 'Pac', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                ax.text(0.22, 0.93, 'IO', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
    
    if start_lat == -15:
        fig.tight_layout()#rect=[0, 0.03, 1, 0.95])
    if num_fig > 1:
        plt.subplots_adjust(hspace=0.0001)
    else:
        plt.subplots_adjust(top=1.4)
    
    plt.subplots_adjust(hspace=0.0001)
    
    figname = figname_st + '_lon_%ideg-step' %step
    figname = figname.replace('yyy', '%s-%s' %(top_rad.upper(), bot_rad.upper()))
    if not use_datline_center:
        figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
#         fig.savefig('test.png')
    fig.show()
    pdb.set_trace()


def plotLonEnso_diff(rhd, lats, valminmax, figname_st, top_rad, bot_rad, useClim=False, use_datline_center=False, useSMA = 0, extraLon = None):
    
    step = 30
    latmin = -15
    num_fig = 3
    figsize = (8,7)
    fig = plt.figure(figsize=figsize)    
    f = 0
    for mon in ['ensoP', 'ensoN']:
        f = f + 1
        latmax = latmin + step
        tr_latmask = getLonLatMask(lats[top_rad], latmin, latmax)
        br_latmask = getLonLatMask(lats[bot_rad], latmin, latmax)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tr_rhdLat = np.nanmean(rhd['%s_%s' %(mon, top_rad)]['crh'][:,tr_latmask,:], axis=1)  # @UndefinedVariable
        br_rhdLat = np.nanmean(rhd['%s_%s' %(mon, bot_rad)]['crh'][:,br_latmask,:], axis=1)  # @UndefinedVariable
        if (top_rad == 'sat') and (useSMA != 0):
            tr_rhdLat = calculateLeapingMean2D(tr_rhdLat, useSMA)
        if (bot_rad == 'sat') and (useSMA != 0):
            br_rhdLat = calculateLeapingMean2D(br_rhdLat, useSMA)
        rhdLat = calculateLonDiff(tr_rhdLat, br_rhdLat, extraLon=extraLon)
        warnings.resetwarnings()

        aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
        ax = fig.add_subplot(num_fig,1,f)
        im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax['lre'] * -1, vmax=valminmax['lre'])
        
        if (latmin == -30) and (latmax == 30):
            fi = 1
        if (latmin == -30) and (latmax == -15):
            fi = 1
        elif (latmin == -15) and (latmax == 0):
            fi = 2
        elif (latmin == 0) and (latmax == 15):
            fi = 3
        elif (latmin == 15) and (latmax == 30):
            fi = 4
        elif (latmin == -15) and (latmax == 15):
            fi = 5
        trhd_rs, brhd_rs = reshapeRhdLat(rhd['e_%s_%s' %(mon, top_rad)]['crh_lon'][fi,:,:,:], rhd['e_%s_%s' %(mon, bot_rad)]['crh_lon'][fi,:,:,:], extraLon)
        
        #: Calculate t-test
        t1, p1 = ttest_ind(trhd_rs, brhd_rs, axis=2)  # @UnusedVariable
#         if f==1:
#             for i in range(p1.shape[0]):
#                 for j in range(p1.shape[1]):
#                     if p1[i,j] <= 0.05:
#                         ax.scatter(x=j,y=i, c='k', s=0.01)
#         else:
        ax.contourf(p1<=0.05, 1, origin='lower', hatches=['', '.'], alpha=0)

        xticks, yticks = getLonTicks(top_rad, bot_rad, tr_rhdLat, br_rhdLat)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels(['0', '5', '10', '15', '20'])
        ax.set_ylabel('Height [km]')
        
        #: Sat - hre
        if ('sat' in [top_rad, bot_rad]) and ('hre' in [top_rad, bot_rad]):
            start_letter = 99 #: c
        else:
            start_letter = 97 #: a
        ax.text(0.01, 1.04, chr((f-1) + start_letter), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
        ax.text(1.03, 0.5, '%s' %mon.upper(), ha='center', va='center',rotation=90,transform=ax.transAxes)
        if f == 2:
            if use_datline_center:
                ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'])
            else:
                ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
            ax.set_xlabel('Longitude [deg]')
            barticks = [valminmax['lre']*-1, valminmax['lre']*-0.75, valminmax['lre']*-0.5, valminmax['lre']*-0.25, 0, valminmax['lre']*0.25, valminmax['lre']*0.5, valminmax['lre']*0.75, valminmax['lre']]
            cbar_ax = fig.add_axes([0.2, 0.28, 0.6, 0.01])#[0.2, 0.04, 0.6, 0.01])
            cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax)#, ticks=barticks)  # @UnusedVariable
            cbar.set_label('CRH [K/day]')
        else:
            ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
        
        if f == 1:
            if use_datline_center:
                ax.text(0.94, 0.93, 'Atl', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                ax.text(0.6, 0.93, 'Pac', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                ax.text(0.22, 0.93, 'IO', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')

    
    figname = figname_st.replace('ensoN', 'enso').replace('ensoP', 'enso') + '_lon'
    figname = figname.replace('yyy', '%s-%s' %(top_rad.upper(), bot_rad.upper()))
    if not use_datline_center:
        figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
    if ('sat' in [top_rad, bot_rad]) and (useSMA != 0):
        figname = figname + '_SMA-%i' %useSMA
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    fig.show()
    pdb.set_trace()


def plotLonSeason_diff(rhd, stLat, valminmax, figname_se, top_rad, bot_rad, useClim=False, use_datline_center=False, extraLon = None):
    figsize = (8,10)
    num_fig = 4
    fig = plt.figure(figsize=figsize)
#     fig2 = plt.figure(figsize=figsize)
    f = 0
    for mon in ['djf', 'mam', 'jja', 'son']:
        f = f + 1
        tr_stepInd = (stLat[top_rad] >= -30) & (stLat[top_rad] <= 30)
        br_stepInd = (stLat[bot_rad] >= -30) & (stLat[bot_rad] <= 30)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tr_rhdLat = np.nanmean(rhd['%s_%s' %(mon, top_rad)]['crh'][:,tr_stepInd,:], axis=1)  # @UndefinedVariable
        br_rhdLat = np.nanmean(rhd['%s_%s' %(mon, bot_rad)]['crh'][:,br_stepInd,:], axis=1)  # @UndefinedVariable
        
        
        rhdLat = calculateLonDiff(tr_rhdLat, br_rhdLat, extraLon)
        warnings.resetwarnings()
        aspect = float('%.2f' %(0.25 / (rhdLat.shape[0] / (rhdLat.shape[1] * 1.))))
        ax = fig.add_subplot(num_fig,1,f)
        im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax * -1, vmax=valminmax)
        trhd_rs, brhd_rs = reshapeRhdLat(rhd['e_%s_%s' %(mon, top_rad)]['crh_lon'], rhd['e_%s_%s' %(mon, bot_rad)]['crh_lon'], extraLon)
        t1, p1 = ttest_ind(trhd_rs, brhd_rs, axis=2)
        ax.contourf(p1<=0.05, 1, origin='lower', hatches=['', '.'], alpha=0)

        xticks, yticks = getLonTicks(top_rad, bot_rad, tr_rhdLat, br_rhdLat)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels(['0', '5', '10', '15', '20'])
        ax.set_ylabel('Height [km]')
        ax.set_title(mon.upper())
        if ('sat' in [top_rad, bot_rad]) and ('hre' in [top_rad, bot_rad]):
            start_letter = 101 #: e
        else:
            start_letter = 97 #: a
        ax.text(0.01, 1.04, chr((f-1) + start_letter), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
        
        if f == num_fig:
            if use_datline_center:
                ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315', '360'])
            else:
                ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
            ax.set_xlabel('Longitude [deg]')
            barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
            cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.01])#[0.2, 0.04, 0.6, 0.01])
            cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax)#, ticks=barticks)  # @UnusedVariable
            cbar.set_label('CRH [K/day]')
        else:
            ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
        if f == 1:
            if use_datline_center:
                ax.text(0.94, 0.93, 'Atl', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                ax.text(0.6, 0.93, 'Pac', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
                ax.text(0.22, 0.93, 'IO', ha='center', va='center', transform = ax.transAxes)#, fontsize='x-large')
#         plt.annotate('Here it is!',xy=(0.5,0.5),xytext=(0.2,0.2),
#              arrowprops=dict(arrowstyle='->',lw=1.5))
#     from matplotlib.patches import FancyArrow
#     FancyArrow(1,0.2,0, 0.1)
    plt.subplots_adjust(hspace=0.0001) 
    figname = figname_se + '_lon'
    figname = figname.replace('yyy', '%s-%s' %(top_rad.upper(), bot_rad.upper()))
    if not use_datline_center:
        figname = os.path.dirname(figname) + '/map_' + os.path.basename(figname)
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    fig.show()
#     fig2.show()
    pdb.set_trace()


def plotVertRHD_diff(rhd_net, rhd_sw, rhd_lw, height, figname_ver, useClim = False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rhd_ver = np.nanmean(rhd_net['hre'], axis=(1,2)) - np.nanmean(rhd_net['lre'], axis=(1,2))  # @UndefinedVariable
    rhd_ver_sw = np.nanmean(rhd_sw['hre'], axis=(1,2)) - np.nanmean(rhd_sw['lre'], axis=(1,2))  # @UndefinedVariable
    rhd_ver_lw = np.nanmean(rhd_lw['hre'], axis=(1,2)) - np.nanmean(rhd_lw['lre'], axis=(1,2))  # @UndefinedVariable
    height_ver = height['hre']
    
    ax.plot(rhd_ver, height_ver, 'k', label= 'NET')
    ax.plot(rhd_ver_lw, height_ver, 'b', label= 'LW')
    ax.plot(rhd_ver_sw, height_ver, 'r', label= 'SW')
    
    ax.legend(loc=1,prop={'size': 10})
    ax.set_xlabel('Cloud Radiative Heating [K/day]')
    ax.set_ylabel('Height [km]')
    ax.set_title('High - Low')

    yticks_man = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
    ylabel_man = (yticks_man / 1000).astype('str')#.astype('int').astype('str')
    ax.set_yticks(yticks_man)
    ax.set_yticklabels(ylabel_man)
    
    ax.vlines(0,0,20500,color='g')

    ax.set_xticks([-0.12, -0.06, 0, 0.06, 0.12])
    plt.tight_layout()
    figname = figname_ver + '_vert'
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    fig.show()
    
    pdb.set_trace()


def plotCF_diff(cf, height, figname_cf, useClim = False):
    
    fig = plt.figure()
    f = -1
        
    f = f + 1
    ax = fig.add_subplot(111)
    cfd_ver = np.nanmean(cf['hre'], axis=(1,2)) - np.nanmean(cf['lre'], axis=(1,2))  # @UndefinedVariable
    
    ax.plot(cfd_ver, height['lre'], 'b')#, label= datan)

#     ax.legend()
    ax.set_xlabel('Cloud Fraction')
    ax.set_ylabel('Height [km]')
    ax.set_title('High - Low')
    
    yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
    ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabel)

    ax.vlines(0,0,height['lre'][-1],color='g')
    ax.set_xticks([-0.015, -0.012, -0.009, -0.006, -0.003, 0, 0.003, 0.006])
    plt.tight_layout()
    fig.show()
    
    figname = figname_cf
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    pdb.set_trace()


def plotWC_diff(wc, height, figname_wc, ticksen, useClim = False):
    
    fig = plt.figure()
#     colour = getCbColour()
    f = 0
    for wcil in ['iwc', 'lwc']:
        f = f + 1
        l = -1
        ax = fig.add_subplot(1,2,f)
        l = l + 1
        wc_ver = np.nanmean(wc['%s_%s' %('hre', wcil)], axis=(1,2)) - np.nanmean(wc['%s_%s' %('lre', wcil)], axis=(1,2))  # @UndefinedVariable
        use_height = height['lre']
        if 'Norm' in figname_wc:
            wc_ver = wc_ver[0:32]
            use_height = use_height[0:32]
        
        
        ax.plot(wc_ver, use_height, 'b')

        ax.set_xlabel('%s [g / m3]' %wcil.upper())
#         ax.set_title('WC')
        if f == 1:
            ax.set_ylabel('Height [km]')
        yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.,17500., 20000.])#, 22500.])
        if 'Norm' in figname_wc:
            yticks = np.array([0.,  2500.,  5000.,  7500., 10000., 12500., 15000.])#, 22500.])
        ylabel = (yticks / 1000).astype('str')#.astype('int').astype('str')
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabel)
        
        ax.vlines(0,0,use_height[-1],color='g')
        
        ax.set_xticks(ticksen[wcil])
        ax.text(0.02, 0.94, chr((f-1) + 97), ha='center', va='center', transform = ax.transAxes, fontsize='x-large')
    plt.tight_layout()
    figname = figname_wc
    if useClim:
        figname = figname + '_anom'
    fig.show()
    
    fig.savefig(figname + '.png')
    pdb.set_trace()


def getEceart(obt, rad, latind, obt_cf=None):
    if rad == 248:
        ret = {'cfd': obt[b'cfd']}
    elif rad == 4:
        ret = {'iwc': obt[b'iwc'], 'lwc': obt[b'lwc'], \
               'ciwc': obt[b'ciwc'], 'clwc': obt[b'clwc'], \
               'cfd': obt[b'cfd']}
    else:
        crh_c = obt[b'c_sw'] + obt[b'c_lw']
        crh_a = obt[b'a_sw'] + obt[b'a_lw']
        crh_sw = obt[b'a_sw'] - obt[b'c_sw']
        crh_lw = obt[b'a_lw'] - obt[b'c_lw']
        crh = crh_a - crh_c
        ret = {'c_sw': obt[b'c_sw'], 'a_sw': obt[b'a_sw'], 'sw': crh_sw, \
               'c_lw': obt[b'c_lw'], 'a_lw': obt[b'a_lw'], 'lw': crh_lw, 
               'c_crh': crh_c, 'a_crh': crh_a, 'crh': crh}
    if obt_cf is not None:
        ret.update({'cfd': obt_cf[b'cfd']})
    #: Fix latitude
    for arname, val in ret.items():
        if val.shape[1] == latind.sum():
            continue
        elif val.shape[1] == latind.shape[0]:
            ret[arname] = val[:, latind, :]
        else:
            print('Markligt')
            pdb.set_trace()
    return ret


def getSat(obt, rad = 1):
#         crh_c = obt['c_sw'] + obt['c_lw']
#         crh_a = obt['a_sw'] + obt['a_lw']
#         crh = crh_a - crh_c
    try:
        if rad == 248:
            ret = {'cfd': obt[b'cfd']}
        elif rad in [2, 3]:
            ret = {'c_sw': obt[b'c_tf_sw'], 'a_sw': obt[b'a_tf_sw'], 'sw': obt[b'tf_sw'], \
                   'c_lw': obt[b'c_tf_lw'], 'a_lw': obt[b'a_tf_lw'], 'lw': obt[b'tf_lw'], \
                   'c_crh': obt[b'c_tf'], 'a_crh': obt[b'a_tf'], 'crh': obt[b'tf']}
        elif rad in [4]:
            ret = {'c_iwc': obt[b'c_iwc'], 'a_iwc': obt[b'a_iwc'], 'iwc': obt[b'iwc'], \
                   'c_lwc': obt[b'c_lwc'], 'a_lwc': obt[b'a_lwc'], 'lwc': obt[b'lwc'], \
                   '90_iwc': obt[b'90_iwc'], '90_lwc': obt[b'90_lwc'], \
                'cfd': obt[b'cfd']}#, \
#                  'c_iwc_std': obt[b'c_iwc_std'], 'a_iwc_std': obt[b'a_iwc_std'], 'iwc_std': obt[b'iwc_std'], \
#                  'c_lwc_std': obt[b'c_lwc_std'], 'a_lwc_std': obt[b'a_lwc_std'], 'lwc_std': obt[b'lwc_std'], \
#                  '90_iwc_std': obt[b'90_iwc_std'], '90_lwc_std': obt[b'90_lwc_std'], \
#                  'cfd_std': obt[b'cfd_std']}
        else:
            ret = {'c_sw': obt[b'c_sw'], 'a_sw': obt[b'a_sw'], 'sw': obt[b'sw'], \
                   'c_lw': obt[b'c_lw'], 'a_lw': obt[b'a_lw'], 'lw': obt[b'lw'], \
                   'c_crh': obt[b'c_crh'], 'a_crh': obt[b'a_crh'], 'crh': obt[b'crh'], \
                   'cfd': obt[b'cfd']}
    except:
        if rad == 248:
            ret = {'cfd': obt['cfd']}
        elif rad in [2, 3]:
            ret = {'c_sw': obt['c_tf_sw'], 'a_sw': obt['a_tf_sw'], 'sw': obt['tf_sw'], \
                   'c_lw': obt['c_tf_lw'], 'a_lw': obt['a_tf_lw'], 'lw': obt['tf_lw'], \
                   'c_crh': obt['c_tf'], 'a_crh': obt['a_tf'], 'crh': obt['tf']}
        elif rad in [4]:
            ret = {'c_iwc': obt['c_iwc'], 'a_iwc': obt['a_iwc'], 'iwc': obt['iwc'], \
                   'c_lwc': obt['c_lwc'], 'a_lwc': obt['a_lwc'], 'lwc': obt['lwc'], \
                   '90_iwc': obt['90_iwc'], '90_lwc': obt['90_lwc'], \
                'cfd': obt['cfd']}#, \
#                  'c_iwc_std': obt['c_iwc_std'], 'a_iwc_std': obt['a_iwc_std'], 'iwc_std': obt['iwc_std'], \
#                  'c_lwc_std': obt['c_lwc_std'], 'a_lwc_std': obt['a_lwc_std'], 'lwc_std': obt['lwc_std'], \
#                  '90_iwc_std': obt['90_iwc_std'], '90_lwc_std': obt['90_lwc_std'], \
#                  'cfd_std': obt['cfd_std']}
        else:
            ret = {'c_sw': obt['c_sw'], 'a_sw': obt['a_sw'], 'sw': obt['sw'], \
                   'c_lw': obt['c_lw'], 'a_lw': obt['a_lw'], 'lw': obt['lw'], \
                   'c_crh': obt['c_crh'], 'a_crh': obt['a_crh'], 'crh': obt['crh'], \
                   'cfd': obt['cfd']}
        
                
    return ret


def loadSat(year_name, season, clt, return_clim=False):
    sat_filename = 'sat_y-%s_s-%s_clt-%i' %(year_name, season, clt)
    sat_loadname = 'Clim_val/%s' %sat_filename
    try:
        sat_val = np.load(sat_loadname + '.npy', encoding='bytes')[0]
        sat_stLat = sat_val[b'stLat']
    except:
        sat_val = np.load(sat_loadname + '.npy')[0]
        sat_stLat = sat_val['stLat']
    if return_clim:
        return sat_val
    else:
        return sat_val, sat_stLat, sat_filename


def loadModel(mod_res, ece_ver, year_name, season, rad_read, return_clim=False):#, loadextra=False):
#             if loadextra:
#                 lre_filename = 'LR%s_y-%s_s-%s_rad-%i_extra' %(ece_ver, year_name, season, rad_read)
#             else:
#                 lre_filename = 'LR%s_y-%s_s-%s_rad-%i' %(ece_ver, year_name, season, rad_read)
    mod_filename = '%s%s_y-%s_s-%s_rad-%i' %(mod_res, ece_ver, year_name, season, rad_read)
    mod_loadname = 'Clim_val/%s' %mod_filename
    try:
        mod_val = np.load(mod_loadname + '.npy', encoding='bytes')[0]
    except:
        mod_val = np.load(mod_loadname + '.npy')[0]
    #: Needed to get the cf at the same time as rad.
    if rad_read == 1:
        try:
            mod_val_cfd = np.load(mod_loadname.replace('rad-%i' %rad_read, 'rad-4') + '.npy', encoding='bytes')[0]
        except:
            mod_val_cfd = np.load(mod_loadname.replace('rad-%i' %rad_read, 'rad-4') + '.npy')[0]
    else:
        mod_val_cfd = None

    #: Prepeare Lat
    mod_lat = mod_val[b'lat']
    mlatind = ((mod_lat >= -30) & (mod_lat <= 30))
    mod_stLat = mod_lat[mlatind]
    
    if return_clim:
        return mod_val, mlatind, mod_val_cfd
    else:
        return mod_val, mod_stLat, mod_filename, mlatind, mod_val_cfd




def changeTo360(obt, ind360):
    if isinstance(obt, dict):
        retv = {}
        for arname, value in obt.items():
            if '_std' in arname:
                retv.update({arname: value})
                continue
            if value.ndim == 3:
                retv.update({arname: value[:, :, ind360]})
            else:
                retv.update({arname: value[ind360]})
    else:
        if value.ndim == 3:
            retv = obt[:, :, ind360]
        else:
            retv = obt[ind360]
    return retv


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-a', '--Anomalie', action='store_true', default=False, help = \
                      'show anomalie instead of absolute value')
    parser.add_option('-c', '--Clt', type='int', default=9, help = \
                      'cloudtype')
    parser.add_option('-d', '--DatelineCenter', action='store_false', default=True, help = \
                      'Use Dateline as Center on map')
    parser.add_option('-e', '--Extra', action='store_true', default=False, help = \
                      'add _extra to filename')
    parser.add_option('-m', '--Month', type='int', default = -1, help = \
                      'Month')
    parser.add_option('-r', '--Rads', type='int', default=1, help = \
                      'rads')
#     parser.add_option('-l', '--Low', action='store_true', default=False, help = \
#                       'resolution')
    parser.add_option('-y', '--Year', type='int', default = 1, help = \
                      'Year')
    
    options, args = parser.parse_args()
    
    #: Years
    if options.Year == 1:
        year_name = 'all'
    else:
        year_name = '%02i' %options.Year
    
    #: Months / season
    if options.Month == -1:
        seasons = ['djf']
    elif options.Month == -2:
        seasons = ['mam']
    elif options.Month == -3:
        seasons = ['jja'] #'djf' 'mam'  'son'
    elif options.Month == -4:
        seasons = ['son']
    elif options.Month == -5:
        seasons = ['ensoP']
    elif options.Month == -6:
        seasons = ['ensoN']
    elif options.Month == -7:
        seasons = ['djf', 'mam', 'jja', 'son']
    elif options.Month == -8:
        seasons = ['ensoP', 'ensoN']
#     useClim = options.Anomalie
    if options.Month in [-5, -6, -8]:
        useClim = True
    else:
        useClim = False
    clt = options.Clt
    rad = options.Rads
    use_datline_center = options.DatelineCenter

    all_res = {}
    ece_ver = 'NE'
    ece_ver_c = 'C6'
    if rad == 248:
        rad_read = 4
    else:
        rad_read = rad


    
    if useClim:
        #: Sat
#         sat_loadname = sat_loadname.replace('y-%s' %year_name, 'y-all').replace('s-%s' %season, 's-all')#.replace('clt-%i' %clt, 'clt-9')
        sat_clim_val = loadSat('all', 'all', clt, return_clim=useClim)
        sat_clim_res = getSat(sat_clim_val, rad)
        #: HRE
#                 mod_loadname = mod_loadname.replace('y-%s' %year_name, 'y-all').replace('s-%s' %season, 's-all')
        hre_clim_val, hlatind, hre_clim_val_cfd = loadModel('HR', ece_ver, 'all', 'all', rad_read, return_clim=useClim)
        hre_clim_res = getEceart(hre_clim_val, rad, hlatind, obt_cf=hre_clim_val_cfd)
        #: LRE
        lre_clim_val, llatind, lre_clim_val_cfd = loadModel('LR', ece_ver, 'all', 'all', rad_read, return_clim=useClim)
        lre_clim_res = getEceart(lre_clim_val, rad, llatind, obt_cf=lre_clim_val_cfd)
        #: LR6
        lr6_clim_val, l6atind, lr6_clim_val_cfd = loadModel('LR', ece_ver_c, 'all', 'all', rad_read, return_clim=useClim)
        lr6_clim_res = getEceart(lr6_clim_val, rad, l6atind, obt_cf=lr6_clim_val_cfd)
    for season in seasons:
        #: Sat
        sat_val, sat_stLat, sat_filename = loadSat(year_name, season, clt)
        sat_res = getSat(sat_val, rad)
        #: HRE
        hre_val, hre_stLat, hre_filename, hlatind, hre_val_cfd = loadModel('HR', ece_ver, year_name, season, rad_read)
        hre_res = getEceart(hre_val, rad, hlatind, obt_cf=hre_val_cfd)
        #: LRE
        lre_val, lre_stLat, lre_filename, llatind, lre_val_cfd = loadModel('LR', ece_ver, year_name, season, rad_read)#, loadextra=options.Extra)
        lre_res = getEceart(lre_val, rad, llatind, obt_cf=lre_val_cfd)
        #: Cmip6
        lr6_val, lr6_stLat, lr6_filename, l6atind, lr6_val_cfd = loadModel('LR', ece_ver_c, year_name, season, rad_read)
        lr6_res = getEceart(lr6_val, rad, l6atind, obt_cf=lr6_val_cfd)

        if useClim:
            #: SAT
            for arname in sat_clim_res.keys():
                sat_res[arname] = sat_res[arname] - sat_clim_res[arname]
            #: HRE
            for arname in hre_clim_res.keys():
                hre_res[arname] = hre_res[arname] - hre_clim_res[arname]
            #: LRE
            for arname in lre_clim_res.keys():
                lre_res[arname] = lre_res[arname] - lre_clim_res[arname]
            #: Cmip 6
            for arname in lre_clim_res.keys():
                lr6_res[arname] = lr6_res[arname] - lr6_clim_res[arname]
        
        hre_lon360 = np.where(hre_val[b'lon180'] < 0, hre_val[b'lon180'] + 360, hre_val[b'lon180'])
        hre_ind360 = np.argsort(hre_lon360)
        lre_lon360 = np.where(lre_val[b'lon180'] < 0, lre_val[b'lon180'] + 360, lre_val[b'lon180'])
        lre_ind360 = np.argsort(lre_lon360)
        lr6_lon360 = np.where(lr6_val[b'lon180'] < 0, lr6_val[b'lon180'] + 360, lr6_val[b'lon180'])
        lr6_ind360 = np.argsort(lr6_lon360)
#         print (lre_lon360[lre_ind360] == lre_val['lon']).all()
#         print (hre_lon360[hre_ind360] == hre_val['lon']).all()
        
        #:--- Start Lon ---
        sat_lon180 = np.asarray(range(-180,180))
        ind180 = np.asarray(range(len(sat_lon180)))
        sat_lon360 = np.where(sat_lon180 < 0, sat_lon180 + 360, sat_lon180)
        sat_ind360 = np.argsort(sat_lon360)
        
        if use_datline_center:
            sat_res = changeTo360(sat_res, sat_ind360)
            hre_res = changeTo360(hre_res, hre_ind360)
            lre_res = changeTo360(lre_res, lre_ind360)
            lr6_res = changeTo360(lr6_res, lr6_ind360)
            sat_res.update({'lon': sat_lon360[sat_ind360]})
            hre_res.update({'lon': hre_lon360[hre_ind360]})
            lre_res.update({'lon': lre_lon360[lre_ind360]})
            lr6_res.update({'lon': lr6_lon360[lr6_ind360]})
        else:
            sat_res.update({'lon': sat_lon180})
            hre_res.update({'lon': hre_val[b'lon180']})
            lre_res.update({'lon': lre_val[b'lon180']})
            lr6_res.update({'lon': lr6_val[b'lon180']})
        
        #: This is used to get monthly means used for std and student-t-test
        load_stat_name = 'Clim_val/statistic_clt-%i_rad-%i_%s' %(clt, rad, season)
        if os.path.isfile((load_stat_name + '.npy')):
            e_res = np.load(load_stat_name + '.npy')[0]
            e_sat_res = e_res['e_sat_res']
            e_hre_res = e_res['e_hre_res']
            e_lre_res = e_res['e_lre_res']
            e_lr6_res = e_res['e_lr6_res']
        else:
            if season in ['djf', 'mam', 'jja', 'son']:
                ant_mon = 3 * 4 #: ant monad * year
            elif season in ['ensoP']:
                ant_mon = 11
            elif season in ['ensoN']:
                ant_mon = 22
            if season in ['djf', 'ensoP']:
                ant_mon_sat = ant_mon - 1 #: 2009-12 not used
            else:
                ant_mon_sat = ant_mon
            if season in ['ensoP', 'ensoN']:
                lon_shape_sat = [6, 125, 360, ant_mon_sat]
                lon_shape_hre = [6, 43, 1024, ant_mon]
                lon_shape_lre = [6, 43, 512, ant_mon]
                vert_shape_sat = [3, 125, ant_mon_sat]
                vert_shape_mod = [3, 43, ant_mon]
            else:
                lon_shape_sat = [125, 360, ant_mon_sat]
                lon_shape_hre = [43, 1024, ant_mon]
                lon_shape_lre = [43, 512, ant_mon]
                vert_shape_sat = [125, ant_mon_sat]
                vert_shape_mod = [43, ant_mon]
                
            if rad == 1:
                use_rad = ['lw', 'crh', 'sw', 'cfd']
            elif rad == 4:
                use_rad = ['iwc', 'lwc', 'cfd']
            e_sat_res = {}
            e_hre_res = {}
            e_lre_res = {}
            e_lr6_res = {}
            for arname in use_rad:
                e_sat_res.update({'%s_vert' %arname: np.zeros(vert_shape_sat)})
                e_hre_res.update({'%s_vert' %arname: np.zeros(vert_shape_mod)})
                e_lre_res.update({'%s_vert' %arname: np.zeros(vert_shape_mod)})
                e_lr6_res.update({'%s_vert' %arname: np.zeros(vert_shape_mod)})
                if rad == 1:
                    e_sat_res.update({'%s_lon' %arname: np.zeros(lon_shape_sat)})
                    e_hre_res.update({'%s_lon' %arname: np.zeros(lon_shape_hre)})
                    e_lre_res.update({'%s_lon' %arname: np.zeros(lon_shape_lre)})
                    e_lr6_res.update({'%s_lon' %arname: np.zeros(lon_shape_lre)})
                elif (rad == 4) and (arname != 'cfd'):
                    e_sat_res.update({'90_%s_vert' %(arname): np.zeros(vert_shape_sat)})
#             
#             
#             
#             e_sat_res = {'lw_lon': np.zeros([125, 360, ant_mon_sat]), 'crh_lon': np.zeros([125, 360, ant_mon_sat]), 
#                          'sw_lon': np.zeros([125, 360, ant_mon_sat]), 'cfd_lon': np.zeros([125, 360, ant_mon_sat]), 
#                          'lw_vert': np.zeros([125, ant_mon_sat]), 'crh_vert': np.zeros([125, ant_mon_sat]), 
#                          'sw_vert': np.zeros([125, ant_mon_sat]), 'cfd_vert': np.zeros([125, ant_mon_sat])}
#             e_hre_res = {'lw_lon': np.zeros([43, 1024, ant_mon]), 'crh_lon': np.zeros([43, 1024, ant_mon]), 
#                          'sw_lon': np.zeros([43, 1024, ant_mon]), 'cfd_lon': np.zeros([43, 1024, ant_mon]), 
#                          'lw_vert': np.zeros([43, ant_mon]), 'crh_vert': np.zeros([43, ant_mon]), 
#                          'sw_vert': np.zeros([43, ant_mon]), 'cfd_vert': np.zeros([43, ant_mon])}
#             e_lre_res = {'lw_lon': np.zeros([43, 512, ant_mon]), 'crh_lon': np.zeros([43, 512, ant_mon]), 
#                          'sw_lon': np.zeros([43, 512, ant_mon]), 'cfd_lon': np.zeros([43, 512, ant_mon]), 
#                          'lw_vert': np.zeros([43, ant_mon]), 'crh_vert': np.zeros([43, ant_mon]), 
#                          'sw_vert': np.zeros([43, ant_mon]), 'cfd_vert': np.zeros([43, ant_mon])}
#             e_lr6_res = {'lw_lon': np.zeros([43, 512, ant_mon]), 'crh_lon': np.zeros([43, 512, ant_mon]), 
#                          'sw_lon': np.zeros([43, 512, ant_mon]), 'cfd_lon': np.zeros([43, 512, ant_mon]), 
#                          'lw_vert': np.zeros([43, ant_mon]), 'crh_vert': np.zeros([43, ant_mon]), 
#                          'sw_vert': np.zeros([43, ant_mon]), 'cfd_vert': np.zeros([43, ant_mon])}
            mi = -1
            si = -1
            test_hre = np.zeros([43, 170, 1024, ant_mon])
            test_sat = np.zeros([125, 61, 360, ant_mon_sat])
            
            for e_year in ['2007', '2008', '2009', '2010']:
                e_months = all_months_comb['%s_months' %season]
                if season in ['ensoP', 'ensoN']:
                    e_months = e_months[e_year]
                for e_mon in e_months:
                    e_mon = '%02i' %e_mon
                    mi = mi + 1
                    #: Get monthly values
                    #: Sat
                    if not ((e_year == '2009') and (e_mon == '12')):
                        si = si + 1
                        e_sat_val, notused1, notused2 = loadSat(e_year, e_mon, clt)
                        e_sat_res_temp = getSat(e_sat_val, rad)
                        test_sat[:,:,:,si] = e_sat_res_temp['c_sw']
                    
                    #: HRE
                    e_hre_val, notused1, notused2, e_hlatind, e_hre_val_cfd = loadModel('HR', ece_ver, e_year, e_mon, rad_read)
                    e_hre_res_temp = getEceart(e_hre_val, rad, e_hlatind, obt_cf=e_hre_val_cfd)
                    test_hre[:,:,:,mi] = e_hre_res_temp['c_sw']
                    
                    #: LRE
                    e_lre_val, notused1, notused2, e_llatind, e_lre_val_cfd = loadModel('LR', ece_ver, e_year, e_mon, rad_read)#, loadextra=options.Extra)
                    e_lre_res_temp = getEceart(e_lre_val, rad, e_llatind, obt_cf=e_lre_val_cfd)
                    #: Cmip6
                    e_lr6_val, notused1, notused2, e_l6atind, e_lr6_val_cfd = loadModel('LR', ece_ver_c, e_year, e_mon, rad_read)
                    e_lr6_res_temp = getEceart(e_lr6_val, rad, e_l6atind, obt_cf=e_lr6_val_cfd)
                    #: Remove clim values
                    if useClim:
                        if not ((e_year == '2009') and (e_mon == '12')):
                            #: SAT
                            for arname in sat_clim_res.keys():
                                e_sat_res_temp[arname] = e_sat_res_temp[arname] - sat_clim_res[arname]
                        #: HRE
                        for arname in hre_clim_res.keys():
                            e_hre_res_temp[arname] = e_hre_res_temp[arname] - hre_clim_res[arname]
                        #: LRE
                        for arname in lre_clim_res.keys():
                            e_lre_res_temp[arname] = e_lre_res_temp[arname] - lre_clim_res[arname]
                        #: Cmip 6
                        for arname in lre_clim_res.keys():
                            e_lr6_res_temp[arname] = e_lr6_res_temp[arname] - lr6_clim_res[arname]
                    #: Change lon depending on dateline
                    if use_datline_center:
                        if not ((e_year == '2009') and (e_mon == '12')):
                            e_sat_res_temp = changeTo360(e_sat_res_temp, sat_ind360)
                        e_hre_res_temp = changeTo360(e_hre_res_temp, hre_ind360)
                        e_lre_res_temp = changeTo360(e_lre_res_temp, lre_ind360)
                        e_lr6_res_temp = changeTo360(e_lr6_res_temp, lr6_ind360)
                    #: Calculate vertical and longitude monthly mean
                    for arname in use_rad:
                        #: Regions
                        if season in ['ensoP', 'ensoN']:
                            vert_num_area = [0, 1, 2]
                            lon_num_area = [0, 1, 2, 3, 4, 5]
                        else:
                            vert_num_area = [0]
                            lon_num_area = [0]
                        #: First Vertical
                        for na in vert_num_area:
                            e_hre_res = calculateVertMean(e_hre_res, e_hre_res_temp, arname, mi, na, hre_stLat, hre_res['lon'], use_datline_center)
                            e_lre_res = calculateVertMean(e_lre_res, e_lre_res_temp, arname, mi, na, lre_stLat, lre_res['lon'], use_datline_center)
                            e_lr6_res = calculateVertMean(e_lr6_res, e_lr6_res_temp, arname, mi, na, lr6_stLat, lr6_res['lon'], use_datline_center)
                            if not ((e_year == '2009') and (e_mon == '12')):
                                e_sat_res = calculateVertMean(e_sat_res, e_sat_res_temp, arname, si, na, sat_stLat, sat_res['lon'], use_datline_center)
                                if arname in ['iwc', 'lwc']:
                                    e_sat_res = calculateVertMean(e_sat_res, e_sat_res_temp, '90_' + arname, si, na, sat_stLat, sat_res['lon'], use_datline_center)
                        if rad == 1:
                            for na in lon_num_area:
                                e_hre_res = calculateLonMean(e_hre_res, e_hre_res_temp, arname, mi, na, hre_stLat)
                                e_lre_res = calculateLonMean(e_lre_res, e_lre_res_temp, arname, mi, na, lre_stLat)
                                e_lr6_res = calculateLonMean(e_lr6_res, e_lr6_res_temp, arname, mi, na, lr6_stLat)
                                if not ((e_year == '2009') and (e_mon == '12')):
                                    e_sat_res = calculateLonMean(e_sat_res, e_sat_res_temp, arname, si, na, sat_stLat)
            np.save(load_stat_name, [{'e_sat_res': e_sat_res, 'e_hre_res': e_hre_res, 
                                      'e_lre_res': e_lre_res, 'e_lr6_res': e_lr6_res}])
        all_res.update({'%s_sat' %season: sat_res, '%s_hre' %season: hre_res, '%s_lre' %season: lre_res, '%s_lr6' %season: lr6_res})
        all_res.update({'e_%s_sat' %season: e_sat_res, 'e_%s_hre' %season: e_hre_res, 'e_%s_lre' %season: e_lre_res, 'e_%s_lr6' %season: e_lr6_res})
    
    stLat = {'sat': sat_stLat, 'hre': hre_stLat, 'lre': lre_stLat, 'lr6': lr6_stLat}
    height_sat = range(0, 240 * 125, 240)
    all_figname_diff = 'Plots/PlotData/xxx_diff_ece-%s_y-%s_s-%s' %(ece_ver, year_name, season)
    cm6_figname_diff = 'Plots/PlotData/xxx_diff_ece-%s_y-%s_s-%s' %(ece_ver_c, year_name, season)
    if rad in [1, 4, 248]:
        height = {'sat': height_sat, 'hre': hre_val[b'height'], 'lre': lre_val[b'height'], 'lr6': lr6_val[b'height']}
    if rad == 248:
        cfd = {'sat': sat_res['cfd'], 'hre': hre_res['cfd'], 'lre': lre_res['cfd'], 'lr6': lr6_res['cfd']}
        figname = 'Plots/PlotData/cf_ece-%s_y-%s_s-%s' %(ece_ver, year_name, season)
        figname_diff = all_figname_diff.replace('/xxx_', '/cf_')
        if options.Month == -7:
            figname = figname.replace(season, '4')
            plotCFSeason(all_res, height, figname, useClim)
        else:
            plotCF_diff(cfd, height, figname_diff, useClim)
            plotCF(cfd, height, figname, useClim)
    elif rad in [1, 2, 3]:
        sat_figname = 'Plots/PlotData/rh_%s' %sat_filename
        hre_figname = 'Plots/PlotData/rh_%s' %hre_filename.replace('_rad-%i' %rad, '')
        lre_figname = 'Plots/PlotData/rh_%s' %lre_filename.replace('_rad-%i' %rad, '')
        lr6_figname = 'Plots/PlotData/rh_%s' %lr6_filename.replace('_rad-%i' %rad, '')        
        rhd = {'sat': sat_res['crh'], 'hre': hre_res['crh'], 'lre': lre_res['crh'], 'lr6': lr6_res['crh'], \
               'e_sat': e_sat_res['crh_lon'], 'e_hre': e_hre_res['crh_lon'], \
               'e_lre': e_lre_res['crh_lon'], 'e_lr6': e_lr6_res['crh_lon']}
        rhd_sw = {'sat': sat_res['sw'], 'hre': hre_res['sw'], 'lre': lre_res['sw'], 'lr6': lr6_res['sw']}
        rhd_lw = {'sat': sat_res['lw'], 'hre': hre_res['lw'], 'lre': lre_res['lw'], 'lr6': lr6_res['lw']}
        figname = {'sat': sat_figname, 'hre': hre_figname, 'lre': lre_figname, 'lr6': lr6_figname}
        figname_diff = all_figname_diff.replace('/xxx_', '/rh_')
        figname_diff = figname_diff.replace('ece-%s' %ece_ver, 'ece-%s-yyy' %ece_ver)
        if rad == 1:
            if options.Month in [-5, -6]:
                sat_valminmax = 2
            else:
                sat_valminmax = 2
            if clt == 8:
                sat_valminmax = 8
            
            hre_valminmax = 2
            lre_valminmax = 2
            lr6_valminmax = 2
                        
            if useClim:
                sat_valminmax = 1 #sat_valminmax / 2
                hre_valminmax = 1 #hre_valminmax / 2
                lre_valminmax = 1 #lre_valminmax / 2
                lr6_valminmax = 1 #lre_valminmax / 2
            
            valminmax = {'sat': sat_valminmax, 'hre': hre_valminmax, 'lre': lre_valminmax, 'lr6': lr6_valminmax}
            valminmax_diff = {'lre': 0.5, 'lr6': 0.5}
            if options.Month == -7:
                figname = {'sat': sat_figname.replace(season, '4'), 'hre': hre_figname.replace(season, '4'), 'lre': lre_figname.replace(season, '4'), 'lr6': lr6_figname.replace(season, '4')}
                figname_vert = figname['hre'].replace('HR%s' %ece_ver, 'All')
                figname_diff = figname_diff.replace(season, '4')
                
                figname_vert_1 = figname['hre'].replace('HR%s' %ece_ver, 'All').replace('all_s-4', 'all_s-1')
                figname_vert_3 = figname['hre'].replace('HR%s' %ece_ver, 'All').replace('all_s-4', 'all_s-3')
                
                
                
                plotVertRHDSeason(all_res, height, figname_vert_1, useClim, months=['djf'])
                plotVertRHDSeason(all_res, height, figname_vert_3, useClim, months=['mam', 'jja', 'son'])
                plotVertRHDSeason(all_res, height, figname_vert, useClim)
                plotLonSeason_diff(all_res, stLat, 1, figname_diff, 'hre', 'sat', useClim, use_datline_center, extraLon=(hre_res['lon'], sat_res['lon']))


#                 plotLonSeason_diff(all_res, stLat, 1, figname_diff, 'lre', 'sat', useClim, use_datline_center, extraLon=(lre_res['lon'], sat_res['lon']))
#                 plotLonSeason_diff(all_res, stLat, 1, figname_diff, 'lr6', 'sat', useClim, use_datline_center, extraLon=(lr6_res['lon'], sat_res['lon']))
#                 plotLonSeason_diff(all_res, stLat, 0.5, figname_diff, 'hre', 'lr6', useClim, use_datline_center)
                plotLonSeason_diff(all_res, stLat, 0.5, figname_diff, 'hre', 'lre', useClim, use_datline_center)
                plotLonSeason_diff(all_res, stLat, 0.5, figname_diff, 'lr6', 'lre', useClim, use_datline_center)
                plotLonSeason(all_res, stLat, valminmax, figname, useClim, use_datline_center)
#                 plotVertRHDSeason_area(all_res, height, stLat, figname_vert, useClim, use_datline_center)
#                 plotVertRHD_CF_Season(all_res, height, figname_vert, useClim)
                print('Time for presentation')
                pdb.set_trace()
                plotLonSeason_forPresentation(all_res, stLat, valminmax, figname, useClim, use_datline_center)
                plotLonSeason_diff_forPresentation(all_res, stLat, 1, figname_diff, 'hre', 'sat', useClim, use_datline_center, extraLon=(hre_res['lon'], sat_res['lon']))
                pdb.set_trace()
            else:
                figname_vert = figname['hre'].replace('HR%s' %ece_ver, 'All')
                if options.Month == -8:
                    figname_vert = figname_vert.replace(season, 'enso')
                    plotVertRHD_Enso(all_res, height, stLat, figname_vert, useClim, use_datline_center)
                    plotLonEnso_diff(all_res, stLat, valminmax_diff, figname_diff, 'hre', 'sat', useClim=useClim, use_datline_center=use_datline_center, useSMA=5, extraLon=(hre_res['lon'], sat_res['lon']))
                    plotLonEnso_diff(all_res, stLat, valminmax_diff, figname_diff, 'hre', 'lre', useClim=useClim, use_datline_center=use_datline_center)
                    plotLonEnso_diff(all_res, stLat, valminmax_diff, figname_diff, 'lr6', 'lre', useClim=useClim, use_datline_center=use_datline_center)
                    plotLonEnso(all_res, stLat, valminmax, figname, useClim, use_datline_center, useSMA=5)
                    
#                 if useClim == False:
# #                     figname_vert = {'sw': figname['hre'].replace('HR%s' %ece_ver, 'SW'), \
# #                                     'lw': figname['hre'].replace('HR%s' %ece_ver, 'LW'), \
# #                                     'net': figname['hre'].replace('HR%s' %ece_ver, 'NET')}
#                     plotVertRHD({'net':rhd, 'sw':rhd_sw, 'lw':rhd_lw}, height, figname_vert, useClim)
#                     plotVertRHD_diff(rhd, rhd_sw, rhd_lw, height, figname_diff, useClim)
                else:
                    plotLonStep(rhd, 30, stLat, valminmax, figname, useClim, use_datline_center, useSMA=5)
                    plotLonStep_diff(rhd, 30, stLat, valminmax_diff, figname_diff, 'hre', 'sat', useClim=useClim, use_datline_center=use_datline_center, useSMA=5, extraLon=(hre_res['lon'], sat_res['lon']))
                    plotLonStep_diff(rhd, 30, stLat, valminmax_diff, figname_diff, 'hre', 'lre', useClim=useClim, use_datline_center=use_datline_center)
                    plotLonStep_diff(rhd, 30, stLat, valminmax_diff, figname_diff, 'lr6', 'lre', useClim=useClim, use_datline_center=use_datline_center)
#                     plotLonStep_diff(rhd, 15, stLat, valminmax_diff, figname_diff, 'lre', 'sat', useClim=useClim, use_datline_center=use_datline_center, useSMA=5, extraLon=(lre_res['lon'], sat_res['lon']))
#                     plotLonStep_diff(rhd, 15, stLat, valminmax_diff, figname_diff, 'lr6', 'sat', useClim=useClim, use_datline_center=use_datline_center, useSMA=5, extraLon=(lr6_res['lon'], sat_res['lon']))
#                     plotLonStep_diff(rhd, 15, stLat, valminmax_diff, figname_diff, 'hre', 'lr6', useClim=useClim, use_datline_center=use_datline_center)
        else:
            figname = {'sat': sat_figname.replace('/rh_', '/rf_'), 'lre': lre_figname.replace('/rh_', '/rf_'), 'hre': hre_figname.replace('/rh_', '/rf_')}
            figname_diff = all_figname_diff.replace('/xxx_', '/rf_')
            valminmax_diff = 100
            valminmax = {'sat': 300, 'lre': 300, 'hre': 300}
            if useClim:
                valminmax = {'sat': 100, 'lre': 100, 'hre': 100}
            plotLonTOA_diff(rhd, valminmax_diff, figname_diff, useClim, use_datline_center)
            plotLonTOA(rhd, valminmax, figname, useClim, use_datline_center)
    
    elif rad in [4]:
        figname90 = 'Plots/PlotData/wc90_%s' %hre_filename.replace('_rad-%i' %rad, '').replace('HR%s' %ece_ver, 'All')
        ticksen90 = {'iwc': np.array([0, 0.005, 0.01, 0.015]), \
                     'lwc': np.array([0, 0.01, 0.02, 0.03]), \
                     'cfd': np.array([0, 0.05, 0.10, 0.15, 0.20])}
            
        if options.Month == -7:
            
            figname90 = figname90.replace(season, '4')
            figname90_cf = figname90.replace('wc90', 'wc90_cf')
            figname90_cf_1 = figname90_cf.replace('all_s-4', 'all_s-1')
            figname90_cf_3 = figname90_cf.replace('all_s-4', 'all_s-3')
#             figname90_cf = figname90_cf.replace('/wc90', '/wc90Norm')
            plotWC_CF_Season(all_res, height, figname90_cf_1, ticksen90, useClim, months=['djf'])
            plotWC_CF_Season(all_res, height, figname90_cf_3, ticksen90, useClim, months=['mam', 'jja', 'son'])
            plotWC_CF_Season(all_res, height, figname90_cf, ticksen90, useClim)
#             plotWC_CF_Season_area(all_res, height, stLat,  figname90_cf, ticksen90, useClim, use_datline_center)
#             plotWCSeason(all_res, height, figname90, ticksen90, useClim)
        elif options.Month in [-8]:
            ticksen90 = {'iwc_nino3': np.array([-0.01, 0, 0.01]), \
                         'lwc_nino3': np.array([-0.02, 0, 0.02, 0.03]), \
                         'cfd_nino3': np.array([-0.20, 0, 0.20]), \
                         'iwc_nino4': np.array([-0.01, 0, 0.01, 0.02, 0.025]), \
                         'lwc_nino4': np.array([-0.02, 0, 0.02, 0.03]), \
                         'cfd_nino4': np.array([-0.20, 0, 0.20, 0.40])}
            figname90 = figname90.replace(season, 'enso')
            figname90_cf = figname90.replace('wc90', 'wc90_cf')
            plotWC_CF_Enso(all_res, height, stLat, figname90_cf, ticksen90, useClim, use_datline_center)
        
        pdb.set_trace()


    
# python2 plotData.py -m -7 -r 4
# python2 plotData.py -m -7 -r 1
# python2 plotData.py -m -8 -r 4
# python2 plotData.py -m -8 -r 1
    
#: Presentation Defence Fig 1
    
    
    
    
    
    
    