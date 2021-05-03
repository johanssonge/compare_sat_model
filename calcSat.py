'''
Created on 23 Mar 2018

@author: a001721
'''
import numpy as np
import h5py
import pdb
import sys
from matplotlib import pyplot as plt
from optparse import OptionParser
import warnings

# sys.path.insert(0, '/home/sm_erjoh/Projects/PhD-2')
# sys.path.insert(0, '/home/sm_erjoh/Projects/PhD-3')
# from plotNrOfclouds import calcNRH#getCf, getRH, calcNRh

all_months_comb = {'djf_months': [1, 2, 12], 
                   'mam_months': [3, 4, 5], 
                   'jja_months': [6, 7, 8], 
                   'son_months': [9, 10, 11], 
                   'ensoP_months': {'2007':[1,2], '2008':[], '2009':[6,7,8,9,10,11,12], '2010':[1,2]}, 
                   'ensoN_months': {'2007':[6,7,8,9,10,11,12], '2008':[1,2,3,4,5,12], '2009':[1,2], '2010':[6,7,8,9,10,11,12]}, 
                   'all_months': range(1,13)}



def getCf(h5f, h5fA, li, clt, nc = False, rel = False):
    if clt == 45:
        #: Read the value from cloud 4 and 5 and add them together
        #: Day time
        dval = h5f['cf/d/d_%i' %4][()][:, li, :] + h5f['cf/d/d_%i' %5][()][:, li, :]
        #: Night time
        nval = h5f['cf/n/n_%i' %4][()][:, li, :] + h5f['cf/n/n_%i' %5][()][:, li, :]
    else:
        #: Read the value from cloud
        #: Day time 
        dval = h5f['cf/d/d_%i' %clt][()][:, li, :]
        #: Night time
        nval = h5f['cf/n/n_%i' %clt][()][:, li, :]
    if rel:
        #: If rel (relative) is true, the value from all cloudy is used in division
        #: Day time
        dallval = h5fA['cf/d/d_%i' %9][()][:, li, :]
        #: Night time
        nallval = h5fA['cf/n/n_%i' %9][()][:, li, :]
    else:
        #: Default
        #: Absoute value, the value from all pixels is used in division
        #: Day time
        dallval = h5fA['cf/d/d_%i' %90][()][:, li, :]
        #: Night time
        nallval = h5fA['cf/n/n_%i' %90][()][:, li, :]
    #: Cloudy value from both day and night
#     val = np.sum((dval + nval) , axis=ax)
    val = dval + nval
    #: Divid value from both day and night
#     allval = np.sum((dallval + nallval), axis=ax)
    allval = dallval + nallval
    #: Cloudy Day
#     sdval = np.sum(dval, axis=ax)
    sdval = dval
    #: Divide Day
#     sdallval = np.sum(dallval, axis=ax)
    sdallval = dallval
    #: Cloudy Night
#     snval = np.sum(nval, axis=ax)
    snval = nval
    #: Divide Night
#     snallval = np.sum(nallval, axis=ax)
    snallval = nallval
    if nc == False:
        #: Default
        #: No previus value are sent in. Therfor only return the present values
        nc = {'nrCl': val, 'nrAll': allval, 'nrDCl': sdval, 'nrDAll': sdallval, \
                'nrNCl': snval, 'nrNAll': snallval}
    else:
        #: Previous values are sent in to the function. 
        #: Add the present values to the previous
        nc['nrCl'] = nc['nrCl'] + val
        nc['nrAll'] = nc['nrAll'] + allval
        nc['nrDCl'] = nc['nrDCl'] + sdval
        nc['nrDAll'] = nc['nrDAll']+ sdallval
        nc['nrNCl'] = nc['nrNCl'] + snval
        nc['nrNAll'] = nc['nrNAll'] + snallval
    return nc

def getRH(h5f, h5fC, li, clt, ax, clah = False, claa = False, clrh = False, clra = False):    

    #: clah = cloudy Heating values
    #: claa = cloudy Antal values
    #: clrh = clear Heating values
    #: clra = clear Antal values
    if clt == 45:
        dshr = h5f['rh/d/heating/shr_d_%i' %4][()][:, li, :] + \
                      h5f['rh/d/heating/shr_d_%i' %5][()][:, li, :]
        dNshr = h5f['rh/d/heating/norm_shr_d_%i' %4][()][:, li, :] + \
                       h5f['rh/d/heating/norm_shr_d_%i' %5][()][:, li, :]
        dlhr = h5f['rh/d/heating/lhr_d_%i' %4][()][:, li, :] +  \
                      h5f['rh/d/heating/lhr_d_%i' %5][()][:, li, :]
        nlhr = h5f['rh/n/heating/lhr_n_%i' %4][()][:, li, :] + \
                      h5f['rh/n/heating/lhr_n_%i' %5][()][:, li, :]
        ant_dshr = h5f['rh/d/quantity/antal_shr_d_%i' %4][()][:, li, :] + \
                          h5f['rh/d/quantity/antal_shr_d_%i' %5][()][:, li, :]
        ant_dNshr = h5f['rh/d/quantity/antal_norm_shr_d_%i' %4][()][:, li, :] + \
                           h5f['rh/d/quantity/antal_norm_shr_d_%i' %5][()][:, li, :]
        ant_dlhr = h5f['rh/d/quantity/antal_lhr_d_%i' %4][()][:, li, :] + \
                          h5f['rh/d/quantity/antal_lhr_d_%i' %5][()][:, li, :]
        ant_nlhr = h5f['rh/n/quantity/antal_lhr_n_%i' %4][()][:, li, :] + \
                          h5f['rh/n/quantity/antal_lhr_n_%i' %5][()][:, li, :]
    else:
        dshr = h5f['rh/d/heating/shr_d_%i' %clt][()][:, li, :]
        dNshr = h5f['rh/d/heating/norm_shr_d_%i' %clt][()][:, li, :]
        dlhr = h5f['rh/d/heating/lhr_d_%i' %clt][()][:, li, :]
        nlhr = h5f['rh/n/heating/lhr_n_%i' %clt][()][:, li, :]
        ant_dshr = h5f['rh/d/quantity/antal_shr_d_%i' %clt][()][:, li, :]
        ant_dNshr = h5f['rh/d/quantity/antal_norm_shr_d_%i' %clt][()][:, li, :]
        ant_dlhr = h5f['rh/d/quantity/antal_lhr_d_%i' %clt][()][:, li, :]
        ant_nlhr = h5f['rh/n/quantity/antal_lhr_n_%i' %clt][()][:, li, :]
    
    
    d_c_shr = h5fC['rh/d/heating/shr_d_%i' %0][()][:, li, :]
    d_c_Nshr = h5fC['rh/d/heating/norm_shr_d_%i' %0][()][:, li, :]
    d_c_lhr = h5fC['rh/d/heating/lhr_d_%i' %0][()][:, li, :]
    n_c_lhr = h5fC['rh/n/heating/lhr_n_%i' %0][()][:, li, :]
    ant_d_c_shr = h5fC['rh/d/quantity/antal_shr_d_%i' %0][()][:, li, :]
    ant_d_c_Nshr = h5fC['rh/d/quantity/antal_norm_shr_d_%i' %0][()][:, li, :]
    ant_d_c_lhr = h5fC['rh/d/quantity/antal_lhr_d_%i' %0][()][:, li, :]
    ant_n_c_lhr = h5fC['rh/n/quantity/antal_lhr_n_%i' %0][()][:, li, :]


    if False in [clah, claa, clrh, clra]:
        clah = {'dshr': dshr, 'dNshr': dNshr, 'dlhr': dlhr, 'nlhr': nlhr}#, \
#                 'drh': dshr + dlhr, 'dNrh': dNshr + dlhr, \
#                 'rh': dshr + dlhr + nlhr, 'Nrh': dNshr + dlhr + nlhr}
        
        claa = {'dshr': ant_dshr, 'dNshr': ant_dNshr, 'dlhr': ant_dlhr, 'nlhr': ant_nlhr}#, \
#                 'drh': ant_dshr + ant_dlhr, 'dNrh': ant_dNshr + ant_dlhr, \
#                 'rh': ant_dshr + ant_dlhr + ant_nlhr, 'Nrh': ant_dNshr + ant_dlhr + ant_nlhr}
        
        clrh = {'dshr': d_c_shr, 'dNshr': d_c_Nshr, 'dlhr': d_c_lhr, 'nlhr': n_c_lhr}#, \
#                 'drh': d_c_shr + d_c_lhr, 'dNrh': d_c_Nshr + d_c_lhr, \
#                 'rh': d_c_shr + d_c_lhr + n_c_lhr, 'Nrh': d_c_Nshr + d_c_lhr + n_c_lhr}
        
        clra = {'dshr': ant_d_c_shr, 'dNshr': ant_d_c_Nshr, 'dlhr': ant_d_c_lhr, 'nlhr': ant_n_c_lhr}#, \
#                 'drh': ant_d_c_shr + ant_d_c_lhr, 'dNrh': ant_d_c_Nshr + ant_d_c_lhr, \
#                 'rh': ant_d_c_shr + ant_d_c_lhr + ant_n_c_lhr, 'Nrh': ant_d_c_Nshr + ant_d_c_lhr + ant_n_c_lhr}
    else:
        clah['dshr'] = clah['dshr'] + dshr
        clah['dNshr'] = clah['dNshr'] + dNshr
        clah['dlhr'] = clah['dlhr'] + dlhr
        clah['nlhr'] = clah['nlhr'] + nlhr
#         clah['drh'] = clah['drh'] + dshr + dlhr
#         clah['dNrh'] = clah['dNrh'] + dNshr + dlhr
#         clah['rh'] = clah['rh'] + dshr + dlhr + nlhr
#         clah['Nrh'] = clah['Nrh'] + dNshr + dlhr +nlhr
        
        claa['dshr'] = claa['dshr'] + ant_dshr
        claa['dNshr'] = claa['dNshr'] + ant_dNshr
        claa['dlhr'] = claa['dlhr'] + ant_dlhr
        claa['nlhr'] = claa['nlhr'] + ant_nlhr
#         claa['drh'] = claa['drh'] + ant_dshr + ant_dlhr
#         claa['dNrh'] = claa['dNrh'] + ant_dNshr + ant_dlhr
#         claa['rh'] = claa['rh'] + ant_dshr + ant_dlhr + ant_nlhr
#         claa['Nrh'] = claa['Nrh'] + ant_dNshr + ant_dlhr +ant_nlhr
        
        clrh['dshr'] = clrh['dshr'] + d_c_shr
        clrh['dNshr'] = clrh['dNshr'] + d_c_Nshr
        clrh['dlhr'] = clrh['dlhr'] + d_c_lhr
        clrh['nlhr'] = clrh['nlhr'] + n_c_lhr
#         clrh['drh'] = clrh['drh'] + d_c_shr + d_c_lhr
#         clrh['dNrh'] = clrh['dNrh'] + d_c_Nshr + d_c_lhr
#         clrh['rh'] = clrh['rh'] + d_c_shr + d_c_lhr + n_c_lhr
#         clrh['Nrh'] = clrh['Nrh'] + d_c_Nshr + d_c_lhr +n_c_lhr
        
        clra['dshr'] = clra['dshr'] + ant_d_c_shr
        clra['dNshr'] = clra['dNshr'] + ant_d_c_Nshr
        clra['dlhr'] = clra['dlhr'] + ant_d_c_lhr
        clra['nlhr'] = clra['nlhr'] + ant_n_c_lhr
#         clra['drh'] = clra['drh'] + ant_d_c_shr + ant_d_c_lhr
#         clra['dNrh'] = clra['dNrh'] + ant_d_c_Nshr + ant_d_c_lhr
#         clra['rh'] = clra['rh'] + ant_d_c_shr + ant_d_c_lhr + ant_n_c_lhr
#         clra['Nrh'] = clra['Nrh'] + ant_d_c_Nshr + ant_d_c_lhr +ant_n_c_lhr
    
    return clah, claa, clrh, clra


def getWC(h5f, h5fC, li, clt, clah = False, claa = False, clrh = False, clra = False):
    
    #: Day time
    diwc = h5f['rh/d/heating/%s_d_%i' %('iwc', clt)][()][:, li, :]
    ant_diwc = h5f['rh/d/quantity/antal_%s_d_%i' %('iwc', clt)][()][:, li, :]
    
    dlwc = h5f['rh/d/heating/%s_d_%i' %('lwc', clt)][()][:, li, :]
    ant_dlwc = h5f['rh/d/quantity/antal_%s_d_%i' %('lwc', clt)][()][:, li, :]
    
    d_c_iwc = h5fC['rh/d/heating/%s_d_%i' %('iwc', 0)][()][:, li, :]
    ant_d_c_iwc = h5fC['rh/d/quantity/antal_%s_d_%i' %('iwc', 0)][()][:, li, :]
    
    d_c_lwc = h5fC['rh/d/heating/%s_d_%i' %('lwc', 0)][()][:, li, :]
    ant_d_c_lwc = h5fC['rh/d/quantity/antal_%s_d_%i' %('lwc', 0)][()][:, li, :]
    
    if False in [clah, claa, clrh, clra]:
        clah = {'diwc': diwc, 'dlwc': dlwc}
        claa = {'diwc': ant_diwc, 'dlwc': ant_dlwc}
        clrh = {'diwc': d_c_iwc, 'dlwc': d_c_lwc}
        clra = {'diwc': ant_d_c_iwc, 'dlwc': ant_d_c_lwc}
    else:
        clah['diwc'] = clah['diwc'] + diwc
        clah['dlwc'] = clah['dlwc'] + dlwc
        claa['diwc'] = claa['diwc'] + ant_diwc
        claa['dlwc'] = claa['dlwc'] + ant_dlwc
        
        clrh['diwc'] = clrh['diwc'] + d_c_iwc
        clrh['dlwc'] = clrh['dlwc'] + d_c_lwc
        clra['diwc'] = clra['diwc'] + ant_d_c_iwc
        clra['dlwc'] = clra['dlwc'] + ant_d_c_lwc
    
    return clah, claa, clrh, clra


def calcNRh(aopCloH, aopCloA, aopClrH, aopClrA, lwsw = 0):
    #: Not shoure if right disciption in this function
    #: Replace nan with 0 for the fraction

    #: 0 = Total Nrh, lw is added later
    #: 2 = Norm Shortwave
    #: 3 = Day total Nrh, Day lw is added later, Only used in plotTropic
    #: 5 = Day Norm Shortwave, samma som 2
    if lwsw in [0, 2, 3, 5]:
        numberCloudy = aopCloA['dNshr']
        RHCloudy = aopCloH['dNshr']
        numberClear = aopClrA['dNshr']
        RHClear = aopClrH['dNshr']
    #: 1 = Total Longwave, day is added later
    #: 4 = Night total rh This is just LW Only used in plotTropic
    elif lwsw in [1, 4]:
        numberCloudy = aopCloA['nlhr']
        RHCloudy = aopCloH['nlhr'] 
        numberClear = aopClrA['nlhr']
        RHClear = aopClrH['nlhr']
        #: Day Longwave Only used in plotTropic
    elif lwsw == 6:
        numberCloudy = aopCloA['dlhr']
        RHCloudy = aopCloH['dlhr']
        numberClear = aopClrA['dlhr']
        RHClear = aopClrH['dlhr']
    #: 7 = Day total rh, Day lw is added later Only used in plotSat
    #: 8 = Day Shortwave Only used in plotSat
    elif lwsw in [7, 8]:
        numberCloudy = aopCloA['dshr']
        RHCloudy = aopCloH['dshr']
        numberClear = aopClrA['dshr']
        RHClear = aopClrH['dshr']
#     tClon = aopCloA['nlhr'] + aopCloA['dlhr'] + aopCloA['dNshr']
#     tCloh = aopCloH['nlhr'] + aopCloH['dlhr']  + aopCloH['dNshr'] 
#     tClrn = aopClrA['nlhr'] + aopClrA['dlhr'] + aopClrA['dNshr']
#     tClrh = aopClrH['nlhr'] + aopClrH['dlhr'] + aopClrH['dNshr']
#     tClo = tCloh / tClon
#     tClr = tClrh / tClrn
#     tC = (tClo - tClr) * aopF

    #: Replace 0 with nan in total number so there is no divide by zero, cloudy-sky
    nanCloA = np.where(numberCloudy == 0, np.nan, numberCloudy)
    #: Dived radiative heating with total number, cloudy-sky
    aopCloNRh = RHCloudy / nanCloA
    #: Replace 0 with nan in total number so there is no divide by zero, clear-sky
    nanClrA = np.where(numberClear == 0, np.nan, numberClear)
    #: Dived radiative heating with total number, clear-sky
    aopClrNRh = RHClear / nanClrA
    #: Extra add dlhr
    if lwsw in [0, 1, 3, 7]:
        numberCloudy_e = aopCloA['dlhr']
        RHCloudy_e = aopCloH['dlhr']
        numberClear_e = aopClrA['dlhr']
        RHClear_e = aopClrH['dlhr']
        #: Replace 0 with nan in total number so there is no divide by zero, cloudy-sky
        nanCloA_e = np.where(numberCloudy_e == 0, np.nan, numberCloudy_e)
        #: Dived radiative heating with total number, cloudy-sky
        aopCloNRh_e = RHCloudy_e / nanCloA_e
        #: Replace 0 with nan in total number so there is no divide by zero, clear-sky
        nanClrA_e = np.where(numberClear_e == 0, np.nan, numberClear_e)
        #: Dived radiative heating with total number, clear-sky
        aopClrNRh_e = RHClear_e / nanClrA_e
        #: Add together        
        aopCloNRh = aopCloNRh + aopCloNRh_e
        aopClrNRh = aopClrNRh + aopClrNRh_e
    #: Extra add nlhr
    if lwsw in [0]:
        numberCloudy_e = aopCloA['nlhr']
        RHCloudy_e = aopCloH['nlhr']
        numberClear_e = aopClrA['nlhr']
        RHClear_e = aopClrH['nlhr']
        #: Replace 0 with nan in total number so there is no divide by zero, cloudy-sky
        nanCloA_e = np.where(numberCloudy_e == 0, np.nan, numberCloudy_e)
        #: Dived radiative heating with total number, cloudy-sky
        aopCloNRh_e = RHCloudy_e / nanCloA_e
        #: Replace 0 with nan in total number so there is no divide by zero, clear-sky
        nanClrA_e = np.where(numberClear_e == 0, np.nan, numberClear_e)
        #: Dived radiative heating with total number, clear-sky
        aopClrNRh_e = RHClear_e / nanClrA_e
        #: Add together        
        aopCloNRh = aopCloNRh + aopCloNRh_e
        aopClrNRh = aopClrNRh + aopClrNRh_e
    
    #: Subtract Clear RH from Cloudy RH
    aopNRh = (aopCloNRh - aopClrNRh)
    return aopNRh, aopClrNRh, aopCloNRh



def calcTOA(toaCloF, toaCloA, toaClrF, toaClrA, lwsw=0):
    #: Not shoure if right disciption in this function
    #: Replace nan with 0 for the fraction
    
    if lwsw == 6:
        numberCloudy = toaCloA['tdlhrA']
        RHCloudy = toaCloF['tdlhrF']
        numberClear = toaClrA['tdlhrA']
        RHClear = toaClrF['tdlhrF']
    elif lwsw in [7, 8]:
        numberCloudy = toaCloA['tdshrA']
        RHCloudy = toaCloF['tdshrF']
        numberClear = toaClrA['tdshrA']
        RHClear = toaClrF['tdshrF']
    else:
        sys.exit()

    #: Replace 0 with nan in total number so there is no divide by zero, cloudy-sky
    nanCloA = np.where(numberCloudy == 0, np.nan, numberCloudy)
    #: Dived radiative heating with total number, cloudy-sky
    toaCloNRh = RHCloudy / nanCloA
    #: Replace 0 with nan in total number so there is no divide by zero, clear-sky
    nanClrA = np.where(numberClear == 0, np.nan, numberClear)
    #: Dived radiative heating with total number, clear-sky
    toaClrNRh = RHClear / nanClrA
    #: Extra add dlhr
    if lwsw in [7]:
        numberCloudy_e = toaCloA['tdlhrA']
        RHCloudy_e = toaCloF['tdlhrF']
        numberClear_e = toaClrA['tdlhrA']
        RHClear_e = toaClrF['tdlhrF']
        #: Replace 0 with nan in total number so there is no divide by zero, cloudy-sky
        nanCloA_e = np.where(numberCloudy_e == 0, np.nan, numberCloudy_e)
        #: Dived radiative heating with total number, cloudy-sky
        toaCloNRh_e = RHCloudy_e / nanCloA_e
        #: Replace 0 with nan in total number so there is no divide by zero, clear-sky
        nanClrA_e = np.where(numberClear_e == 0, np.nan, numberClear_e)
        #: Dived radiative heating with total number, clear-sky
        toaClrNRh_e = RHClear_e / nanClrA_e
        #: Add together        
        toaCloNRh = toaCloNRh + toaCloNRh_e
        toaClrNRh = toaClrNRh + toaClrNRh_e
    
    #: Subtract Clear RH from Cloudy RH
    toaNRh = (toaCloNRh - toaClrNRh)
    return toaNRh, toaClrNRh, toaCloNRh


def calcWC(wcCloF, wcCloA, wcClrF, wcClrA, lwsw=0):
    #: IWC day
    if lwsw == 0:
        numberCloudy = wcCloA['diwc']
        WcCloudy = wcCloF['diwc']
        numberClear = wcClrA['diwc']
        WcClear = wcClrF['diwc']
    #: LWC day
    elif lwsw == 1:
        numberCloudy = wcCloA['dlwc']
        WcCloudy = wcCloF['dlwc']
        numberClear = wcClrA['dlwc']
        WcClear = wcClrF['dlwc']
    #: IWC all, day
    elif lwsw == 2:
        numberCloudy = wcCloA['diwc'] + wcClrA['diwc']
        WcCloudy = wcCloF['diwc']  + wcClrF['diwc']
    #: LWC all, day
    elif lwsw == 3:
        numberCloudy = wcCloA['dlwc'] + wcClrA['dlwc']
        WcCloudy = wcCloF['dlwc']  + wcClrF['dlwc']
    else:
        print('Wrong lwsw')
        sys.exit()
    
    #: Replace 0 with nan in total number so there is no divide by zero, cloudy-sky
    nanCloWcA = np.where(numberCloudy == 0, np.nan, numberCloudy)
    #: Dived wc with total number, cloudy-sky
    CloWc = WcCloudy / nanCloWcA
    
    if lwsw in [0, 1]:
        #: Replace 0 with nan in total number so there is no divide by zero, clear-sky
        nanClrWcA = np.where(numberClear == 0, np.nan, numberClear)
        #: Dived wc with total number, clear-sky
        ClrWc = WcClear / nanClrWcA
        #: Subtract Clear wc from Cloudy wc
        WC = (CloWc - ClrWc)
    else:
        WC = ClrWc = CloWc

    return WC, ClrWc, CloWc


def calcCF(cfnr, dn=''):
    if dn in ['ds', 'dl']:
        dnu = 'd'
    elif dn == 'lw':
        dnu = ''
#         pdb.set_trace()
    else:
        dnu = dn
    #: Deside if use day ('d'), night ('n') or both ('')
    nrall = 'nr%sAll' %dnu.upper()
    nrclo = 'nr%sCl' %dnu.upper()
    #: Replace 0 with nan in totalnumber to avoid divide by zero
    nanhtNr = np.where(cfnr[nrall] == 0, np.nan, cfnr[nrall])
    #: Divide cloudy with total number
    htcf = (cfnr[nrclo] / nanhtNr)
    return htcf


def getCfRh(h5f, h5fA, h5fC, latInd, clt, dn=''):
    #: Anvands ej????
    #: dn = day ('d'), night ('n') or both (''), ds = day shortwave, dl = day longwave
    htNr = getCf(h5f, h5fA, latInd, clt)
    #: htClaH = cloudy Heating values
    #: htClaA = cloudy Antal values
    #: htClrH = clear Heating values
    #: htClrA = clear Antal values
    htClaH, htClaA, htClrH, htClrA = getRH(h5f, h5fC, latInd, clt, 2, clah = False, claa = False, clrh = False, clra = False)
    '''    Cloud Fraction    '''
    htcf = calcCF(htNr, dn)
    '''    Radiative Heating    '''
    if dn == '':
        lwsw = 0 #: Total RH
        pdb.set_trace()
    elif dn == 'd':
        lwsw = 7#3 #: Daily RH
    elif dn == 'n':
        lwsw = 4 #: Night lw / RH
    elif dn == 'ds':
        lwsw = 8#5 #: daily sw
    elif dn == 'dl':
        lwsw = 6 #: daily lw
    elif dn == 'lw':
        lwsw = 1 #: Toatal lw
    htNRh, htClr, htClo = calcNRh(htcf, htClaH, htClaA, htClrH, htClrA, lwsw = lwsw)
#     cf = np.nanmean(htcf, axis = 2)  # @UndefinedVariable
#     rh = np.nanmean(htNRh, axis = 2)  # @UndefinedVariable
#     rhClr = np.nanmean(htClr, axis = 2)  # @UndefinedVariable
    cf = htcf
    rh = htNRh
    rhClr = htClr
    return cf, rh, rhClr



def getTOA(h5f, h5fC, li, clt, btf=False, bta=False, btfC=False, btaC=False):
    
    tdshrF = h5f['TOA-BOACRE/d/TOACRE/tshr_d_%i' %clt][()][:,li,:]
    tdlhrF = h5f['TOA-BOACRE/d/TOACRE/tlhr_d_%i' %clt][()][:,li,:]
    tdshrA = h5f['TOA-BOACRE/d/TOACRE/antal_tshr_d_%i' %clt][()][:,li,:]
    tdlhrA = h5f['TOA-BOACRE/d/TOACRE/antal_tlhr_d_%i' %clt][()][:,li,:]
    
    tdshrFC = h5fC['TOA-BOACRE/d/TOACRE/tshr_d_%i' %0][()][:,li,:]
    tdlhrFC = h5fC['TOA-BOACRE/d/TOACRE/tlhr_d_%i' %0][()][:,li,:]
    tdshrAC = h5fC['TOA-BOACRE/d/TOACRE/antal_tshr_d_%i' %0][()][:,li,:]
    tdlhrAC = h5fC['TOA-BOACRE/d/TOACRE/antal_tlhr_d_%i' %0][()][:,li,:]
    if btf == False:
        btf = {'tdshrF': tdshrF, 'tdlhrF': tdlhrF}
        bta = {'tdshrA': tdshrA, 'tdlhrA': tdlhrA}
        btf = {'tdshrF': tdshrF, 'tdlhrF': tdlhrF}
        bta = {'tdshrA': tdshrA, 'tdlhrA': tdlhrA}
        
        btfC = {'tdshrF': tdshrFC, 'tdlhrF': tdlhrFC}
        btaC = {'tdshrA': tdshrAC, 'tdlhrA': tdlhrAC}
        btfC = {'tdshrF': tdshrFC, 'tdlhrF': tdlhrFC}
        btaC = {'tdshrA': tdshrAC, 'tdlhrA': tdlhrAC}
    else:
        btf['tdshrF'] = btf['tdshrF'] + tdshrF
        btf['tdlhrF'] = btf['tdlhrF'] + tdlhrF
        bta['tdshrA'] = bta['tdshrA'] + tdshrA
        bta['tdlhrA'] = bta['tdlhrA'] + tdlhrA
        
        btfC['tdshrF'] = btfC['tdshrF'] + tdshrFC
        btfC['tdlhrF'] = btfC['tdlhrF'] + tdlhrFC
        btaC['tdshrA'] = btaC['tdshrA'] + tdshrAC
        btaC['tdlhrA'] = btaC['tdlhrA'] + tdlhrAC
    return btf, bta, btfC, btaC

def read4D(h5file, h5fileC, datasets, areaind, retv=False, latlon=False):

    dn = 'd'
    stlat = h5file['Start Latitude'][()]
    lons = np.asarray(range(-180, 180))
    lats = stlat
    latInd = np.where((stlat >= minLat) & (stlat < maxLat))[0]
    ct = h5file.fid.name.split('nc_')[1].split('_')[2]
    if retv == False:
        retv = {}
        for dataset in datasets:
            dset_shape = '%s_%s_%s' %(dataset, 'd', ct)
            dsetC_shape = '%s_%s_%s' %(dataset, 'd', '0')
#             data_shape = h5file['d4/shape/%s' %dset_shape][()]
            retv.update({dset_shape: np.zeros([len(lats), len(lons)])})
            retv.update({dset_shape + '_antal': np.zeros([len(lats), len(lons)])})
            retv.update({dsetC_shape: np.zeros([len(lats), len(lons)])})
            retv.update({dsetC_shape + '_antal': np.zeros([len(lats), len(lons)])})
#         for i in range(data_shape[0]):
#             retv.update({i: []})
#     try:
#         h5file['d4/%s/%s' %(dn, dset)].keys()
#     except:
#         print('%s\t%s\t%s' %(os.path.basename(filename), dn, dataset))
#         continue
    
#     for latind in range(1:len(latin))    
    dset = '%s_%s_%s' %(dataset, dn, ct)
    for arname, val in h5file['d4/%s/%s' %(dn, dset)].items():
        height = int(arname.split('_')[0])
        lat = int(arname.split('_')[1])
        lon = int(arname.split('_')[2])
        #: TODO: Controll if this is correct, tror det ar det faktiskt, maste ju kora fran -30 for det ar beatamt i crhabove
        #: Daremot kanske dat ar onodigt, men beholler so lange
        lati = np.where(lats == (lat - 30))[0][0]
        loni = np.where(lons == (lon - 180))[0][0]
        pdb.set_trace()
        if areaind == '':
            if lat in latInd:
                retv[height] = np.hstack((retv[height], val[()]))
                pdb.set_trace()
        else:
            try:
                areaind[lati, loni]
            except:
                print('bajs')
                pdb.set_trace()
            if areaind[lati, loni]:
                retv[height] = np.hstack((retv[height], val[()]))
    if latlon:
        retv.update({'lon':lons})
        retv.update({'lat':lats})

    return retv



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-c', '--Clt', type='int', default=9, help = \
                      'cloudtype')
    parser.add_option('-m', '--Month', type='int', default = -1, help = \
                      'Month')
    parser.add_option('-l', '--Low', action='store_true', default=False, help = \
                      'resolution')
    parser.add_option('-y', '--Year', type='int', default = 1, help = \
                      'Year')
    parser.add_option('-i', '--Interactive', action='store_true', default=False, help = \
                      'interactive mode')
    options, args = parser.parse_args()
    
    mainDir = '/nobackup/smhid12/sm_erjoh/PhD-2'
    datadir = '%s/Data/hitRate' %mainDir
    mainDir4 = '/nobackup/smhid14/sm_erjoh/PhD-2' #'/nobackup/smhid12/sm_erjoh/PhD-2'
    datadir4 = '%s/Data/hitRate' %mainDir4
    #: Years
    if options.Year == 1:
        years = [7, 8, 9, 10]
        year_name = 'all'
    else:
        years = [options.Year]
        year_name = '20%02i' %options.Year
    #: Months / season
    if options.Month == -1:
        season = 'djf'
        months = all_months_comb['%s_months' %season]
    elif options.Month == -2:
        season = 'mam'
        months = all_months_comb['%s_months' %season]
    elif options.Month == -3:
        season = 'jja' #'djf' 'mam'  'son'
        months = all_months_comb['%s_months' %season]
    elif options.Month == -4:
        season = 'son'
        months = all_months_comb['%s_months' %season]
    elif options.Month == -5:
        season = 'ensoP'
        months = range(1,13)
        enso_month = all_months_comb['%s_months' %season]
    elif options.Month == -6:
        season = 'ensoN'
        months = range(1,13)
        enso_month = all_months_comb['%s_months' %season]
    elif options.Month == -100:
        years = [7, 8, 9, 10]
        season = 'all'
        months = range(1,13)# all_months_comb['%s_months' %season]
    else:
        season = ('%02i' %options.Month)
        months = [options.Month]
    #: Clt
    clt = options.Clt
    if clt == 9:
        clt_name = ''
    else:
        clt_name = '_clt_%i' %clt
    print('Year = %s' %year_name)
    print('Season = %s' %season)
    print('Clt = %i' %clt)
    
    maxLat = 30
    minLat = maxLat * -1
    l = -1
    for year in years:
        for mon in months:
            if options.Month in [-5, -6]:
                if mon not in enso_month['20%02i' %year]:
                    continue
            
            l = l + 1
            satname = '%s/nc_m_%03i_%i_y%02i_utanP.h5' %(datadir4, mon, clt, year)
            satnameA = '%s/nc_m_%03i_90_y%02i.h5' %(datadir, mon, year)
            satnameC = '%s/nc_m_%03i_0_y%02i.h5' %(datadir, mon, year)
#             if satname in ['/nobackup/smhid14/sm_erjoh/PhD-2/Data/hitRate/nc_m_002_9_y08_utanP.h5']:
#                 continue
            
            if (mon == 12) and (year == 9):
                continue
            h5file = h5py.File(satname, 'r')
            h5fileA = h5py.File(satnameA, 'r')
            h5fileC = h5py.File(satnameC, 'r')
            stLat = h5file['Start Latitude'][()]
            latInd = (stLat >= minLat) & (stLat < maxLat)
            allLat = np.ones(stLat.shape).astype('bool')
            if l == 0:
                htNr = getCf(h5file, h5fileA, allLat, clt)
                #: htClaH = cloudy Heating values
                #: htClaA = cloudy Antal values
                #: htClrH = clear Heating values
                #: htClrA = clear Antal values
                wcClaF, wcClaA, wcClrF, wcClrA = getWC(h5file, h5fileC, allLat, clt, clah = False, claa = False, clrh = False, clra = False)
                htClaH, htClaA, htClrH, htClrA = getRH(h5file, h5fileC, allLat, clt, 2, clah = False, claa = False, clrh = False, clra = False)
#                 toaClaF, toaClaA, toaClrF, toaClrA = getTOA(h5file, h5fileC, allLat, clt, btf=False, bta=False, btfC=False, btaC=False)
#                 wcClaF_m = {}; wcClaF_m.update({'diwc':wcClaF['diwc'].reshape([1,125,61,360])});wcClaF_m.update({'dlwc':wcClaF['dlwc'].reshape([1,125,61,360])}) 
#                 wcClaA_m = {}; wcClaA_m.update({'diwc':wcClaA['diwc'].reshape([1,125,61,360])});wcClaA_m.update({'dlwc':wcClaA['dlwc'].reshape([1,125,61,360])})
#                 wcClrF_m = {}; wcClrF_m.update({'diwc':wcClrF['diwc'].reshape([1,125,61,360])});wcClrF_m.update({'dlwc':wcClrF['dlwc'].reshape([1,125,61,360])})
#                 wcClrA_m = {}; wcClrA_m.update({'diwc':wcClrA['diwc'].reshape([1,125,61,360])});wcClrA_m.update({'dlwc':wcClrA['dlwc'].reshape([1,125,61,360])})
#                 
#                 htClaH_m = {}; htClaH_m.update({'dshr':htClaH['dshr'].reshape([1,125,61,360])});htClaH_m.update({'dlhr':htClaH['dlhr'].reshape([1,125,61,360])})
#                 htClaA_m = {}; htClaA_m.update({'dshr':htClaA['dshr'].reshape([1,125,61,360])});htClaA_m.update({'dlhr':htClaA['dlhr'].reshape([1,125,61,360])})
#                 htClrH_m = {}; htClrH_m.update({'dshr':htClrH['dshr'].reshape([1,125,61,360])});htClrH_m.update({'dlhr':htClrH['dlhr'].reshape([1,125,61,360])})
#                 htClrA_m = {}; htClrA_m.update({'dshr':htClrA['dshr'].reshape([1,125,61,360])});htClrA_m.update({'dlhr':htClrA['dlhr'].reshape([1,125,61,360])})
#                 toaClaF_m = {}; toaClaF_m.update({'dshr':toaClaF.reshape([1,125,61,360])});toaClaF_m.update({'dlhr':toaClaF.reshape([1,125,61,360])})
#                 toaClaA_m = {}; toaClaA_m.update({'dshr':toaClaA.reshape([1,125,61,360])});toaClaA_m.update({'dlhr':toaClaA.reshape([1,125,61,360])})
#                 toaClrF_m = {}; toaClrF_m.update({'dshr':toaClrF.reshape([1,125,61,360])});toaClrF_m.update({'dlhr':toaClrF.reshape([1,125,61,360])})
#                 toaClrA_m = {}; toaClrA_m.update({'dshr':wcClaF.reshape([1,125,61,360])});wcClaF_m.update({'dlhr':wcClaF.reshape([1,125,61,360])})
#                 htNr_m = {}; htNr_m.update({'nrDAll':htNr['nrDAll'].reshape([1,125,61,360])});htNr_m.update({'nrDCl':htNr['nrDCl'].reshape([1,125,61,360])})
            else:
                htNr = getCf(h5file, h5fileA, allLat, clt, nc = htNr)
                wcClaF, wcClaA, wcClrF, wcClrA = getWC(h5file, h5fileC, allLat, clt, clah = wcClaF, claa = wcClaA, clrh = wcClrF, clra = wcClrA)
                htClaH, htClaA, htClrH, htClrA = getRH(h5file, h5fileC, allLat, clt, 2, clah =  htClaH, claa =  htClaA, clrh = htClrH, clra = htClrA)
#                 toaClaF, toaClaA, toaClrF, toaClrA = getTOA(h5file, h5fileC, allLat, clt, btf=toaClaF, bta=toaClaA, btfC=toaClrF, btaC=toaClrA)
#                 htNr_n = getCf(h5file, h5fileA, allLat, clt)
#                 wcClaF_n, wcClaA_n, wcClrF_n, wcClrA_n = getWC(h5file, h5fileC, allLat, clt, clah = False, claa = False, clrh = False, clra = False)
#                 htClaH_n, htClaA_n, htClrH_n, htClrA_n = getRH(h5file, h5fileC, allLat, clt, 2, clah = False, claa = False, clrh = False, clra = False)
#                 toaClaF_n, toaClaA_n, toaClrF_n, toaClrA_n = getTOA(h5file, h5fileC, allLat, clt, btf=False, bta=False, btfC=False, btaC=False)

#                 wcClaF_m['diwc'] = np.concatenate((wcClaF_m['diwc'], wcClaF_n['diwc'].reshape([1,125,61,360])))
#                 wcClaF_m['dlwc'] = np.concatenate((wcClaF_m['dlwc'], wcClaF_n['dlwc'].reshape([1,125,61,360]))) 
#                 wcClaA_m['diwc'] = np.concatenate((wcClaA_m['diwc'], wcClaA_n['diwc'].reshape([1,125,61,360])))
#                 wcClaA_m['dlwc'] = np.concatenate((wcClaA_m['dlwc'], wcClaA_n['dlwc'].reshape([1,125,61,360])))
#                 wcClrF_m['diwc'] = np.concatenate((wcClrF_m['diwc'], wcClrF_n['diwc'].reshape([1,125,61,360])))
#                 wcClrF_m['dlwc'] = np.concatenate((wcClrF_m['dlwc'], wcClrF_n['dlwc'].reshape([1,125,61,360])))
#                 wcClrA_m['diwc'] = np.concatenate((wcClrA_m['diwc'], wcClrA_n['diwc'].reshape([1,125,61,360])))
#                 wcClrA_m['dlwc'] = np.concatenate((wcClrA_m['dlwc'], wcClrA_n['dlwc'].reshape([1,125,61,360])))
#                 
#                 htClaH_m['dshr'] = np.concatenate((htClaH_m['dshr'], htClaH_n['dshr'].reshape([1,125,61,360])))
#                 htClaH_m['dlhr'] = np.concatenate((htClaH_m['dlhr'], htClaH_n['dlhr'].reshape([1,125,61,360])))
#                 htClaA_m['dshr'] = np.concatenate((htClaA_m['dshr'], htClaA_n['dshr'].reshape([1,125,61,360])))
#                 htClaA_m['dlhr'] = np.concatenate((htClaA_m['dlhr'], htClaA_n['dlhr'].reshape([1,125,61,360])))
#                 htClrH_m['dshr'] = np.concatenate((htClrH_m['dshr'], htClrH_n['dshr'].reshape([1,125,61,360])))
#                 htClrH_m['dlhr'] = np.concatenate((htClrH_m['dlhr'], htClrH_n['dlhr'].reshape([1,125,61,360])))
#                 htClrA_m['dshr'] = np.concatenate((htClrA_m['dshr'], htClrA_n['dshr'].reshape([1,125,61,360])))
#                 htClrA_m['dlhr'] = np.concatenate((htClrA_m['dlhr'], htClrA_n['dlhr'].reshape([1,125,61,360])))
#                 
#                 htNr_m['nrDAll'] = np.concatenate((htNr_m['nrDAll'], htNr_n['nrDAll'].reshape([1,125,61,360])))
#                 htNr_m['nrDCl'] = np.concatenate((htNr_m['nrDCl'], htNr_n['nrDCl'].reshape([1,125,61,360])))
                
            h5file.close()
            h5fileA.close()
            h5fileC.close()
    cfd = calcCF(htNr, 'd')
    rhd, rhClrd, rhClod = calcNRh(htClaH, htClaA, htClrH, htClrA, lwsw = 7)
    rhds, rhClrds, rhClods = calcNRh(htClaH, htClaA, htClrH, htClrA, lwsw = 8)
    rhdl, rhClrdl, rhClodl = calcNRh(htClaH, htClaA, htClrH, htClrA, lwsw = 6)
    
#     tfd, tfClrd, tfClod = calcTOA(toaClaF, toaClaA, toaClrF, toaClrA, lwsw = 7)
#     tfds, tfClrds, tfClods = calcTOA(toaClaF, toaClaA, toaClrF, toaClrA, lwsw = 8)
#     tfdl, tfClrdl, tfClodl = calcTOA(toaClaF, toaClaA, toaClrF, toaClrA, lwsw = 6)
    
    iwc, iwcClrd, iwcClod = calcWC(wcClaF, wcClaA, wcClrF, wcClrA, lwsw = 0)
    lwc, lwcClrd, lwcClod = calcWC(wcClaF, wcClaA, wcClrF, wcClrA, lwsw = 1)
    
    iwc90, iwcClrd90, iwcClod90 = calcWC(wcClaF, wcClaA, wcClrF, wcClrA, lwsw = 2)
    lwc90, lwcClrd90, lwcClod90 = calcWC(wcClaF, wcClaA, wcClrF, wcClrA, lwsw = 3)


#     cfd_m = calcCF(htNr_m, 'd')
#     
#     rhd_m, rhClrd_m, rhClod_m = calcNRh(htClaH_m, htClaA_m, htClrH_m, htClrA_m, lwsw = 7)
#     rhds_m, rhClrds_m, rhClods_m = calcNRh(htClaH_m, htClaA_m, htClrH_m, htClrA_m, lwsw = 8)
#     rhdl_m, rhClrdl_m, rhClodl_m = calcNRh(htClaH_m, htClaA_m, htClrH_m, htClrA_m, lwsw = 6)
#     
#     iwc_m, iwcClrd_m, iwcClod_m = calcWC(wcClaF_m, wcClaA_m, wcClrF_m, wcClrA_m, lwsw = 0)
#     lwc_m, lwcClrd_m, lwcClod_m = calcWC(wcClaF_m, wcClaA_m, wcClrF_m, wcClrA_m, lwsw = 1)
#     
#     iwc90_m, iwcClrd90_m, iwcClod90_m = calcWC(wcClaF_m, wcClaA_m, wcClrF_m, wcClrA_m, lwsw = 2)
#     lwc90_m, lwcClrd90_m, lwcClod90_m = calcWC(wcClaF_m, wcClaA_m, wcClrF_m, wcClrA_m, lwsw = 3)
    
#     save_dict = {'c_tf_sw': tfClrds, 'a_tf_sw': tfClods, 'tf_sw': tfds, \
#                  'c_tf_lw': tfClrdl, 'a_tf_lw': tfClodl, 'tf_lw': tfdl, \
#                  'c_tf': tfClrd, 'a_tf':tfClod, 'tf': tfd, \
    save_dict = {'c_sw': rhClrds, 'a_sw': rhClods, 'sw': rhds, \
                 'c_lw': rhClrdl, 'a_lw': rhClodl, 'lw': rhdl, \
                 'c_crh': rhClrd, 'a_crh':rhClod, 'crh': rhd, \
                 'c_iwc': iwcClrd, 'a_iwc': iwcClod, 'iwc': iwc, \
                 'c_lwc': lwcClrd, 'a_lwc': lwcClod, 'lwc': lwc, \
                 '90_iwc': iwc90, '90_lwc': lwc90, \
                 'cfd': cfd, 'stLat': stLat}
#                  'c_sw_std': np.nanstd(rhClrds_m, axis=(0,2,3)), 'a_sw_std': np.nanstd(rhClods_m, axis=(0,2,3)), 'sw_std': np.nanstd(rhds_m, axis=(0,2,3)), \
#                  'c_lw_std': np.nanstd(rhClrdl_m, axis=(0,2,3)), 'a_lw_std': np.nanstd(rhClodl_m, axis=(0,2,3)), 'lw_std': np.nanstd(rhdl_m, axis=(0,2,3)), \
#                  'c_crh_std': np.nanstd(rhClrd_m, axis=(0,2,3)), 'a_crh_std': np.nanstd(rhClod_m, axis=(0,2,3)), 'crh_std': np.nanstd(rhd_m, axis=(0,2,3)), \
#                  'c_iwc_std': np.nanstd(iwcClrd_m, axis=(0,2,3)), 'a_iwc_std': np.nanstd(iwcClod_m, axis=(0,2,3)), 'iwc_std': np.nanstd(iwc_m, axis=(0,2,3)), \
#                  'c_lwc_std': np.nanstd(lwcClrd_m, axis=(0,2,3)), 'a_lwc_std': np.nanstd(lwcClod_m, axis=(0,2,3)), 'lwc_std': np.nanstd(lwc_m, axis=(0,2,3)), \
#                  '90_iwc_std': np.nanstd(iwc90_m, axis=(0,2,3)), '90_lwc_std': np.nanstd(lwc90_m, axis=(0,2,3)), \
#                  'cfd_std': np.nanstd(cfd_m, axis=(0,2,3)), \
    savename = 'Clim_val/sat_y-%s_s-%s_clt-%i_utanP' %(year_name, season, clt)
    np.save(savename, [save_dict])
    sys.exit()
    
    
    warnings.simplefilter("ignore", category=RuntimeWarning)
    cfd_ver = np.nanmean(cfd, axis=(1,2))  # @UndefinedVariable
    rhd_ver = np.nanmean(rhd, axis=(1,2))  # @UndefinedVariable
    rhd_ver_sw = np.nanmean(rhds, axis=(1,2))  # @UndefinedVariable
    rhd_ver_lw = np.nanmean(rhdl, axis=(1,2))  # @UndefinedVariable
    warnings.resetwarnings()
    
    height = range(0,240*125,240)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rhd_ver, height, 'k', label= 'NET')
    ax.plot(rhd_ver_sw, height, 'r', label= 'SW')
    ax.plot(rhd_ver_lw, height, 'b', label= 'LW')
    ax.axvline(0,0,1,color='g')
    ax.legend()
    ax.set_xlabel('Cloud Radiative Heating [K / day]')
    ax.set_ylabel('Height [km]')
#         ax.set_title('%s, Total Sky - Clear Sky' %run[0:2])
#     ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2])
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels((yticks / 1000).astype('int'))
#         ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])
    
    figname = 'Plots/sat_vert_%s%s' %(season, clt_name)
    fig.savefig(figname + '.png')
#     fig.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cfd_ver, height, 'k', label= 'CF')
    ax.axvline(0,0,1,color='g')
    ax.legend()
    ax.set_xlabel('Cloud Fraction')
    ax.set_ylabel('Height [km]')
#         ax.set_title('%s, Total Sky - Clear Sky' %run[0:2])
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels((yticks / 1000).astype('int'))
#         ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])
    
    figname = 'Plots/sat_cf_%s%s' %(season, clt_name)
    fig.savefig(figname + '.png')
#     fig.show()

    
    warnings.simplefilter("ignore", category=RuntimeWarning)
    rhd_lon = np.nanmean(rhd, axis=(1))[0:-18, :]  # @UndefinedVariable
    rhd_lon_sw = np.nanmean(rhds, axis=(1))[0:-18, :]  # @UndefinedVariable
    rhd_lon_lw = np.nanmean(rhdl, axis=(1))[0:-18, :]  # @UndefinedVariable
    rhd_lon_a_lw = np.nanmean(rhClodl, axis=(1))[0:-18, :]  # @UndefinedVariable
    rhd_lon_c_lw = np.nanmean(rhClrdl, axis=(1))[0:-18, :]  # @UndefinedVariable
    rhd_lon_a_sw = np.nanmean(rhClods, axis=(1))[0:-18, :]  # @UndefinedVariable
    rhd_lon_c_sw = np.nanmean(rhClrds, axis=(1))[0:-18, :]  # @UndefinedVariable
    
    warnings.resetwarnings()
    rhd_lon_c = rhd_lon_c_sw + rhd_lon_c_lw
    rhd_lon_a = rhd_lon_a_sw + rhd_lon_a_lw
    valminmax = 2
    
    useClim = True
    if options.Month in [-5, -6]:
        valminmax = 1
    if useClim:
        valminmax = 0.5
    crh_lon_dict = {'c_sw': rhd_lon_c_sw, 'a_sw': rhd_lon_a_sw, 'sw': rhd_lon_sw, \
                        'c_lw': rhd_lon_c_lw, 'a_lw': rhd_lon_a_lw, 'lw': rhd_lon_lw, \
                        'c_crh': rhd_lon_c, 'a_crh': rhd_lon_a, 'crh': rhd_lon}
        
    i_name = {0: 'a_', 1: 'c_', 2: ''}
    j_name = {0: 'sw', 1: 'lw', 2: 'crh'}
    
    f = 0
    fig = plt.figure(figsize = (18,12))
    for i in range(3):
        for j in range(3):
            f = f + 1
            ax = fig.add_subplot(3,3,f)
            sub_name = i_name[i] + j_name[j] 
            plot_val = crh_lon_dict[sub_name]
            im = ax.imshow(plot_val, origin='lower', cmap='RdBu_r', aspect=1.5, vmin=valminmax*-1, vmax=valminmax)
    
            ax.set_yticks([1, 21, 42, 63, 83, 104])
            ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])
    
            ax.set_xticks([1, 45, 90, 135, 180, 225, 270, 315, 358])
            ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
            
            ax.set_title(sub_name)
            if f in [1,4,7]:
                ax.set_ylabel('Height [km]')
            if f in [7,8,9]:
                ax.set_xlabel('Longitude [deg]')
            if f in [3,6,9]:
                fig.subplots_adjust(right=0.95)
                if f == 3:
                    cbar_ax = fig.add_axes([0.96, 0.64, 0.008, 0.24])
                if f == 6:
                    cbar_ax = fig.add_axes([0.96, 0.365, 0.008, 0.24])
                if f == 9:
                    cbar_ax = fig.add_axes([0.96, 0.095, 0.008, 0.24])
                barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
                cbar = fig.colorbar(im, cax=cbar_ax)
#                 cbar = fig.colorbar(im, aspect=50, ticks=barticks)#orientation='horizontal', aspect=50, ticks=barticks)
#     figname = 'Plots/sat_lon_%s' %(seas)
    figname = 'Plots/%s_y_%s_s_%s_lon_tot-clr%s' %('sat', year_name, season, clt_name)
    fig.savefig(figname + '.png')
    fig.show()
    if useClim:
        clim_name = 'Clim_val/sat_y-%s_s-%s_clt-%i' %('all', 'all', 9)
        clim_val = np.load(clim_name + '.npy')[0]
        rhd = rhd - clim_val['rhd']
    
    stLon180 = np.asarray(range(-180,180))
    ind180 = np.asarray(range(len(stLon180)))
    stLon360 = np.where(stLon180 < 0, stLon180 + 360, stLon180)
    ind360 = np.argsort(stLon360)
    use_datline_center = True
    
    if use_datline_center:
        indLon = ind360
    else:
        indLon = ind180
    
    pdb.set_trace()
    fig = plt.figure(figsize=(8,8))    
    f = 0
    for stepL in range(-30,30,15)[::-1]:
        f = f + 1
        endLat = stepL + 15
        stepInd = (stLat >= stepL) & (stLat < endLat)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rhdLat = np.nanmean(rhd[:,stepInd,:], axis=1)[0:84, :]  # @UndefinedVariable
        warnings.resetwarnings()
        ax = fig.add_subplot(4,1,f)
        im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=1.08, vmin=valminmax*-1, vmax=valminmax)
    
        ax.set_yticks([1, 21, 42, 63, 83])
        ax.set_yticklabels(['0', '5', '10', '15', '20'])
#         ax.set_yticks([1, 21, 42, 63, 83, 104])
#         ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])

        ax.set_xticks([1, 45, 90, 135, 180, 225, 270, 315, 358])
        ax.set_title('Latitude, %i - %i' %(stepL, endLat))
#         ax.set_title(sub_name)
#         if f in [6]:
        ax.set_ylabel('Height [km]')
        if f in [4]:
            ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
            ax.set_xlabel('Longitude [deg]')
            barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
#             cbar_ax = fig.add_axes([0.17, 0.04, 0.7, 0.01])
            cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.01])
            cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)
#             cbar = fig.colorbar(im, orientation='horizontal', aspect=50, ticks=barticks)
        else:
            ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
    figname = 'Plots/%s_y_%s_s_%s_lon_15deg-step%s' %('sat', year_name, season, clt_name)
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    fig.show()


    fig = plt.figure(figsize=(8,24))    
    f = 0
    for stepL in range(-30,30,5)[::-1]:
        f = f + 1
        endLat = stepL + 5
        stepInd = (stLat >= stepL) & (stLat < endLat)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rhdLat = np.nanmean(rhd[:,stepInd,:], axis=1)[0:-18, :]  # @UndefinedVariable
        warnings.resetwarnings()
        ax = fig.add_subplot(12,1,f)
        im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=1.08, vmin=valminmax*-1, vmax=valminmax)
        ax.set_yticks([1, 21, 42, 63, 83, 104])
        ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])

        ax.set_xticks([1, 45, 90, 135, 180, 225, 270, 315, 358])
        ax.set_title('Latitude, %i - %i' %(stepL, endLat))
#         ax.set_title(sub_name)
        if f in [6]:
            ax.set_ylabel('Height [km]')
        if f in [12]:
            ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
            ax.set_xlabel('Longitude [deg]')
            barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
            cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.01])
            cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)
#             cbar = fig.colorbar(im, orientation='horizontal', aspect=50, ticks=barticks)
        else:
            ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
    fig.show()
    figname = 'Plots/%s_y_%s_s_%s_lon_5deg-step%s' %('sat', year_name, season, clt_name)
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
#     pdb.set_trace()
    fig = plt.figure(figsize=(8,12))
    f = 0
    for stepL in range(-30,0,5)[::-1]:
        f = f + 1
        endLat = np.abs(stepL)
        stepInd = (stLat >= stepL) & (stLat < endLat)

        warnings.simplefilter("ignore", category=RuntimeWarning)
        rhdLat = np.nanmean(rhd[:,stepInd,:], axis=1)[0:-18, :]  # @UndefinedVariable
        warnings.resetwarnings()
        ax = fig.add_subplot(6,1,f)
        im = ax.imshow(rhdLat, origin='lower', cmap='RdBu_r', aspect=1.08, vmin=valminmax*-1, vmax=valminmax)
    
        ax.set_yticks([1, 21, 42, 63, 83, 104])
        ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])
        ax.set_xticks([1, 45, 90, 135, 180, 225, 270, 315, 358])
        ax.set_title('Latitude, %i - %i' %(stepL, endLat))
        
#         ax.set_title(sub_name)
        if f in [3]:
            ax.set_ylabel('Height [km]')
        if f in [6]:
            ax.set_xticklabels(['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180'])
            ax.set_xlabel('Longitude [deg]')
            barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
            cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.01])
            cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)
#             cbar = fig.colorbar(im, orientation='horizontal', aspect=50, ticks=barticks)
        else:
            ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
    fig.show()
    figname = 'Plots/%s_y_%s_s_%s_lon_Ldeg-step%s' %('sat', year_name, season, clt_name)
    if useClim:
        figname = figname + '_anom'
    fig.savefig(figname + '.png')
    pdb.set_trace()
    
            
            
            
            
            
            