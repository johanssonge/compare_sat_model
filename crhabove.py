'''
Created on 25 aug 2014

@author: a001721
'''
import datetime
import glob
import math
from optparse import OptionParser
import os
import pdb
import sys
import time

import h5py

import numpy as np
# import pp  # @UnresolvedImport
# from multiprocessing.pool import worker
import copy
import warnings


class DataObject(object):
    """
    Class to handle data objects with several arrays.
    
    """
    def __getattr__(self, name):
        try:
            return self.geo[name]
        except KeyError:
            try:
                return self.data[name]
            except KeyError:
                raise AttributeError("%s instance has no attribute '%s'" % (self.__class__.__name__, name))
    
    def __setattr__(self, name, value):
        if name in ('geo', 'data'):
            object.__setattr__(self, name, value)
        elif name in self.data:
            self.data[name] = value
        else:
            self.geo[name] = value
            
class CloudsatObject(DataObject):
    def __init__(self):
        DataObject.__init__(self)                            
        self.geo = {
                    'longitude': None,
                    'latitude': None,
                    'Height': None,
                    'TAI_start': None,
                    'UTC_start': None,
                    'elevation': None,
                    'Profile_time': None,
                    'sec1970': None
                    }
        self.data = {
                    'Albedo': None,
                    'FD': None,
                    'FD_NA': None,
                    'FD_NC': None,
                    'FU': None,
                    'FU_NA': None,
                    'FU_NC': None,
                    'QR': None,
                    'RH': None,
                    'TOACRE': None,
                    'BOACRE': None,
                    'FlagCounts': None,
                    'Land_Char': None,
                    'Scene_status': None,
                    'MeanOLR': None,
                    'MeanOSR': None,
                    'MeanQLW': None,
                    'MeanQSW': None,
                    'MeanSFCE': None,
                    'MeanSFCR': None,
                    'MeanSLR': None,
                    'MeanSSR': None,
                    'Meansolar': None,
                    'SigmaOLR': None,
                    'SigmaOSR': None,
                    'SigmaQLW': None,
                    'SigmaQSW': None,
                    'SigmaSFCE': None,
                    'SigmaSFCR': None,
                    'SigmaSLR': None,
                    'SigmaSSR': None,
                    'Sigmasolar': None,
                    'Solar_zenith_angle': None,
                    'Status': None
                    }
def convertDType(val):
#        name = val.dtype.name
    if val.dtype.names is None:
        testType = val.dtype
    else:
        testType = val.dtype[0]
    dtype = np.dtype(testType.type)
    promoType = np.promote_types(testType, testType) #@UndefinedVariable
    if dtype == promoType:
        if dtype == np.dtype('float32'):
            dtype = np.dtype('float')
        re = val.astype(dtype)
    else:
        re = val
    
    return re
def makeDataUsefull(data, nv = False, attrs = [], ntb = True):
    val = data[()]
    val = convertDType(val)
    if attrs == []:
        try:
            factor = data.attrs['factor']
        except:
            factor = 1
        try:
            offset = data.attrs['offset']
        except:
            offset = 0
        try:
            lv = data.attrs['valid_range'][0]
            hv = data.attrs['valid_range'][1]
        except:
            lv = np.nan
            hv = np.nan
        try:
            missing = data.attrs['missing']
        except:
            missing = np.nan
#         data.attrs['_FillValue']
    else:
        try:
            missing = attrs[0][attrs[1] + '.missing'][()][0][0]
        except:
            missing = np.nan
        try:
            lv = attrs[0][attrs[1] + '.valid_range'][()][0][0][0]
            hv = attrs[0][attrs[1] + '.valid_range'][()][0][0][1]
        except:
            lv = np.nan
            hv = np.nan
        factor = attrs[0][attrs[1] + '.factor'][()][0][0]
        offset = attrs[0][attrs[1] + '.offset'][()][0][0]
    if nv:
        # TODO: I remove error message
        np.seterr(invalid='ignore')
        val = np.where(val == missing, np.nan, val)
        if ntb:
            val = np.where(val < lv, np.nan, val)
            val = np.where(val > hv, np.nan, val)
        np.seterr(invalid='warn')
    val = val * 1. / factor
    val = val + offset
#     if attrs[1] in ['QR', 'TOACRE', 'BOACRE']:
#         pdb.set_trace()
    return val

def readFLXHRLidar(filename):
    retv = CloudsatObject()
    newName = {'Longitude':'longitude', 'Latitude': 'latitude', 'DEM_elevation': 'elevation'}
    try:
        h5file = h5py.File(filename, 'r')
    except:
        print('Stupied file')
        print(filename)
        return -1
    root = "2B-FLXHR-LIDAR"
    groups = ['Geolocation Fields', 'Data Fields']
    for group in groups:
        groupFields = h5file["%s/%s" % (root, group)]
        for dataset in groupFields.keys():
            if dataset in retv.geo.keys() or dataset in retv.data.keys() or dataset in newName.keys():
                
                if "Swath Attributes" not in h5file[root]:
                    print('No Attributes')
                    print(filename)
                    h5file.close()
                    return -1
                else:
                    ntb = True
                    if dataset in ['TOACRE', 'BOACRE']:
                        ntb = False
                    val = makeDataUsefull(groupFields[dataset], nv = True, attrs = [h5file["%s/%s" % (root, "Swath Attributes")], dataset], ntb = ntb)
#                     if dataset in ['BOACRE', 'TOACRE']:
#                         if (val[0,:] > 0).any()
#                             print(filename)
#                             pdb.set_trace()
#                 val = groupFields[dataset].value
#                 val = convertDType(val)
#                 try:
#                     factor = h5file["%s/%s" % (root, "Swath Attributes")][dataset + '.factor'].value[0][0]
#                     offset = h5file["%s/%s" % (root, "Swath Attributes")][dataset + '.offset'].value[0][0]
#                 except:
#                     print('strange')
#                     h5file.close()
#                     return -1
#                 if factor != 1:
#                     val = val * 1. / factor
#                 if offset != 0:
#                     val = val + offset
                
                if dataset in retv.geo.keys():
                    retv.geo[dataset] = val
                elif dataset in newName.keys():
                    retv.geo[newName[dataset]] = val
                else:
                    retv.data[dataset] = val
            else:
                continue

    retv = convertTime(retv)
    h5file.close()
    return retv


def convertTime(cloudsat):
    dsec = time.mktime((1993,1,1,0,0,0,0,0,0)) - time.timezone
    start_sec1970 = cloudsat.TAI_start + dsec
#        start_time = time.gmtime(start_sec1970[0])
    cloudsat.sec1970 = cloudsat.Profile_time + start_sec1970
    return cloudsat


def readCloudType(filename, latlon = False):
    retv = {'CL': None, 'CLType': None, 'CLTop': None, 'CLTopF': None, 'CLBase': None}#, 'CF': None}
    h5file = h5py.File(filename, 'r')
    root = "2B-CLDCLASS-LIDAR"
    group = 'Data Fields'
    retv['CLType'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'CloudLayerType')], nv = True)
    retv['CLTop'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'CloudLayerTop')], nv = True)
    retv['CLTopF'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'CloudLayerTop')])
    retv['CLBase'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'CloudLayerBase')], nv = True)
    retv['CL'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Cloudlayer')], nv = True)
#     retv['CF'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'CloudFraction')], nv = True)
#    [16996]
    if latlon:
        group = 'Geolocation Fields'
        lat = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Latitude')], nv = True)
        lon = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Longitude')], nv = True)
        ele = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'DEM_elevation')], nv = True)
        retv.update({'latitude': lat})
        retv.update({'longitude': lon})
        retv.update({'elevation': ele})
       
    h5file.close() 
    return retv


def readGeoType(filename, latlon = False):
    retv = {'Radar_Reflectivity': np.array([])}
    if filename != '':
        h5file = h5py.File(filename, 'r')
        root = "2B-GEOPROF"
        group = 'Data Fields'
        retv['Radar_Reflectivity'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Radar_Reflectivity')], nv=True)
        if latlon:
            group = 'Geolocation Fields'
            lat = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Latitude')], nv=True)
            lon = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Longitude')], nv=True)
            ele = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'DEM_elevation')], nv=True)
            retv.update({'latitude': lat})
            retv.update({'longitude': lon})
            retv.update({'elevation': ele})
           
        h5file.close()
    return retv



def readTauType(filename, latlon = False):
    
    retv = {'layer_optical_depth': np.array([]), 'total_optical_depth': np.array([])}
    if filename != '':
        h5file = h5py.File(filename, 'r')
        root = "2B-TAU"
        group = 'Data Fields'
        retv['layer_optical_depth'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'layer_optical_depth')], nv=True)
        retv['total_optical_depth'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'total_optical_depth')], nv=True)
        if latlon:
            group = 'Geolocation Fields'
            lat = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Latitude')], nv=True)
            lon = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Longitude')], nv=True)
            ele = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'DEM_elevation')], nv=True)
            retv.update({'latitude': lat})
            retv.update({'longitude': lon})
            retv.update({'elevation': ele})
           
        h5file.close()
    return retv



def readCwcType(filename, latlon = False):
    
    retv = {'ice_water_content': np.array([]), 'ice_water_path': np.array([]), \
            'liq_water_content': np.array([]), 'liq_water_path': np.array([])}#, \
#             'ice_water_content_precip': np.array([]), 'ice_water_path_precip': np.array([]), \
#             'liq_water_content_precip': np.array([]), 'liq_water_path_precip': np.array([]), \
#             'ice_water_content_oprecip': np.array([]), 'ice_water_path_oprecip': np.array([]), \
#             'liq_water_content_oprecip': np.array([]), 'liq_water_path_oprecip': np.array([])}
    if filename != '':
        h5file = h5py.File(filename, 'r')
        root = "2B-CWC-RVOD"
        group = 'Data Fields'
        h5file["%s/%s/%s" % (root, group, 'RVOD_ice_water_content')]
        retv['ice_water_content'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'RVOD_ice_water_content')], nv=True)
        retv['ice_water_path'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'RVOD_ice_water_path')], nv=True)
        retv['liq_water_content'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'RVOD_liq_water_content')], nv=True)
        retv['liq_water_path'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'RVOD_liq_water_path')], nv=True)
        temp = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'RVOD_CWC_status')], nv=True).astype(int)
#         flag2bit = [(x>>2) & 1 for x in temp]
        flag2bit = np.bitwise_and((temp >> 2), 1)
        flagT = flag2bit==1
        flagNT = ~flagT
        #: Create a 2D array with all nans
        nans = np.ones(retv['ice_water_content'].shape) * np.nan
        #: Put all values where bit2 = 1 to nan
        retv['ice_water_content'][flagT,:] = nans[flagT,:]
        retv['liq_water_content'][flagT,:] = nans[flagT,:]
        #: Same but for 1D
        retv['ice_water_path'][flagT] = np.nan
        retv['liq_water_path'][flagT] = np.nan
        
        if latlon:
            group = 'Geolocation Fields'
            lat = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Latitude')], nv=True)
            lon = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Longitude')], nv=True)
            ele = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'DEM_elevation')], nv=True)
            retv.update({'latitude': lat})
            retv.update({'longitude': lon})
            retv.update({'elevation': ele})
           
        h5file.close()
    return retv

def readAuxType(filename, latlon = False):
    
    retv = {'Specific_humidity': np.array([]), 'Temperature': np.array([])}
    if filename != '':
        h5file = h5py.File(filename, 'r')
        root = "ECMWF-AUX"
        group = 'Data Fields'
        h5file["%s/%s/%s" % (root, group, 'Specific_humidity')]
        retv['Specific_humidity'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Specific_humidity')], nv=True)
        retv['Temperature'] = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Temperature')], nv=True)
        if latlon:
            group = 'Geolocation Fields'
            lat = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Latitude')], nv=True)
            lon = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'Longitude')], nv=True)
            ele = makeDataUsefull(h5file["%s/%s/%s" % (root, group, 'DEM_elevation')], nv=True)
            retv.update({'latitude': lat})
            retv.update({'longitude': lon})
            retv.update({'elevation': ele})
           
        h5file.close()
    return retv

def npshrfactortest():
    sza_deg = 45
    lat_deg = 20
    itid = 1167660172.03
    sza_rad = np.deg2rad(sza_deg)
    latitude_rad = np.deg2rad(lat_deg)
    d = datetime.datetime.fromtimestamp(itid)
    doy = d.timetuple().tm_yday
    declination_rad = np.arcsin(np.sin(np.deg2rad((360/365.)*(doy-81))) * np.sin(np.deg2rad(23.45)))
    latTalt = np.tan(latitude_rad) * np.tan(declination_rad)
                
    if np.cos(sza_rad) > 0:
        qi = np.cos(sza_rad)
    else:
        qi = np.nan
    if latTalt < -1:
        qd = 0
    else:
        if latTalt > 1:
            h0 = np.pi
        else:
            h0 = np.arccos(-1 * latTalt) 
        qd = (1 / np.pi) *  (h0 * np.sin(latitude_rad) * np.sin(declination_rad) + np.cos(latitude_rad) * np.cos(declination_rad)*np.sin(h0))
    factor = qd / qi
    return factor
def shrfactortest():
    sza_deg = 45
    lat_deg = 20
    itid = 1167660172.03

    sza_rad = math.radians(sza_deg)
    latitude_rad = math.radians(lat_deg)
    d = datetime.datetime.fromtimestamp(itid)
    doy = d.timetuple().tm_yday
    declination_rad = math.asin(math.sin(math.radians((360/365.)*(doy-81))) * math.sin(math.radians(23.45)))
    latTalt = math.tan(latitude_rad) * math.tan(declination_rad)
                
    if math.cos(sza_rad) > 0:
        qi = math.cos(sza_rad)
    else:
        qi = np.nan
    if latTalt < -1:
        qd = 0
    else:
        if latTalt > 1:
            h0 = math.pi
        else:
            h0 = math.acos(-1 * latTalt) 
        qd = (1 / math.pi) *  (h0 * math.sin(latitude_rad) * math.sin(declination_rad) + math.cos(latitude_rad) * math.cos(declination_rad)*math.sin(h0))
    factor = qd / qi
    return factor

def npshrfactor(sza_deg, lat_deg, itid):

    sza_rad = np.radians(sza_deg)
    latitude_rad = np.radians(lat_deg)
    d = datetime.datetime.fromtimestamp(itid)
    doy = d.timetuple().tm_yday
    declination_rad = np.arcsin(np.sin(np.radians((360/365.)*(doy-81))) * np.sin(np.radians(23.45)))
    latTalt = np.tan(latitude_rad) * np.tan(declination_rad)
                
    if np.cos(sza_rad) > 0:
        qi = np.cos(sza_rad)
    else:
        qi = np.nan
    if latTalt < -1:
        qd = 0
    else:
        if latTalt > 1:
            h0 = np.pi
        else:
            h0 = np.arccos(-1 * latTalt) 
        qd = (1 / np.pi) *  (h0 * np.sin(latitude_rad) * np.sin(declination_rad) + np.cos(latitude_rad) * np.cos(declination_rad)*np.sin(h0))
    factor = qd / qi
    return factor

def shrfactor(sza_deg, lat_deg, itid):
    #: Convert Solar zenith angle in degree to radians
    sza_rad = math.radians(sza_deg)
    #: Convert latitude in degrees to radians
    latitude_rad = math.radians(lat_deg)
    #: Find Day Of Year (doy) from timestamp
    d = datetime.datetime.fromtimestamp(itid)
    doy = d.timetuple().tm_yday
    #: solar declination
    declination_rad = math.asin(math.sin(math.radians((360/365.)*(doy-81))) * math.sin(math.radians(23.45)))
    #: solar declination according to Hartman - Global Physical Climatology 1994
    #: declination_rad = 0.006918*math.cos(0) + (-0.399912*math.cos(1*2*math.pi*doy/365) + 0.070257 * math.sin(1*2*math.pi*doy/365)) + (-0.006758*math.cos(2*2*math.pi*doy/365) + 0.000907 * math.sin(2*2*math.pi*doy/365)) + (-0.002697*math.cos(3*2*math.pi*doy/365) + 0.001480 * math.sin(3*2*math.pi*doy/365))
    
    #: Hour angle
    latTalt = math.tan(latitude_rad) * math.tan(declination_rad)
    #: If Q < or = to 0 than qd = 0 But use nan instead beqause cant divide by zero
    if math.cos(sza_rad) > 0:
        qi = math.cos(sza_rad)
    else:
        qi = np.nan
    if latTalt < -1:
        qd = 0
    else:
        if latTalt > 1:
            h0 = math.pi
        else:
            h0 = math.acos(-1 * latTalt) 
        qd = (1 / math.pi) *  (h0 * math.sin(latitude_rad) * math.sin(declination_rad) + math.cos(latitude_rad) * math.cos(declination_rad)*math.sin(h0))
    factor = qd / qi
    return factor

def putValinRightPlace(resDict, shr, lhr, shrN, tod, spec, ct, ih, useSP = False):
    dayornight = False
    if useSP:
        dnlist = ['sp']
        if shr > 0 and spec == 'd':
            dayornight = True
            
    else:
        dnlist = [tod, 'a']
        if shr > 0:
            dayornight = True
            
    if dayornight:
        sw = shr
        sn = 1
    else:
        sw = 0
        sn = 0

    if not np.isnan(shrN) and sw != 0:
        swN = shrN
        snN = 1
    else:
        swN = 0
        snN = 0
        if sw != 0:
            print('knas')
    
    if -9999 in [lhr]:
        lw = 0
        ln = 0
    else:
        lw = lhr
        ln = 1
        
    nhr = lw + sw
    norm_nhr = lw + swN
    for to in dnlist:
        resDict['shr_%s_%i' %(to, ct)][ih] = resDict['shr_%s_%i' %(to, ct)][ih] + [sw, sn]
        resDict['norm_shr_%s_%i' %(to, ct)][ih] = resDict['norm_shr_%s_%i' %(to, ct)][ih] + [swN, snN]
        resDict['lhr_%s_%i' %(to, ct)][ih] = resDict['lhr_%s_%i' %(to, ct)][ih] + [lw, ln]
        resDict['nhr_%s_%i' %(to, ct)][ih] = resDict['nhr_%s_%i' %(to, ct)][ih] + [nhr, ln]
        resDict['norm_nhr_%s_%i' %(to, ct)][ih] = resDict['norm_nhr_%s_%i' %(to, ct)][ih] + [norm_nhr, ln] 
    
    return resDict


def countVal(val, di, ni, av = False):
    if val.shape[0] == 0:
        vd = 0
        vdn = 0
        vn = 0
        vnn = 0
        vad_mn = []
        van_mn = []
    else:
        dnanInd = ~np.isnan(val[di])
        nnanInd = ~np.isnan(val[ni])
        vad_mn = val[di]
        van_mn = val[ni]
        vad = val[di][dnanInd]
        van = val[ni][nnanInd]
        vd = sum(vad)
        vdn = len(vad)
        vn = sum(van)
        vnn = len(van)
    if av:
        return vd, vdn, vad_mn, vn, vnn, van_mn
    else:
        return vd, vdn, vn, vnn


def countValLarg(val, ind, lart, pd=False):
    nanInd = ~np.isnan(val)
    Ind = nanInd & ind
    vi = val[Ind]
    #: TODO: > or >= ??
    vil = vi[vi >= lart]
    v = sum(vi[vi > lart])
    vn = len(vi[vi > lart])
#     if pd:
#         pdb.set_trace()
    return v, vn, vil

def calcWP(wpind, cloudIndT, cloudIndB, height, iwc, lwc):
    #: This is where there is a cloud
    vindT = np.where(cloudIndT[wpind])[0]
    vindB = np.where(cloudIndB[wpind])[0]
#     pdb.set_trace()
    #: Pad with data around
    vindT = np.asarray(range(vindT[0]-1, vindT[-1] + 2))
    vindB = np.asarray(range(vindB[0]-1, vindB[-1] + 2))
    #: Make sure that they do not have the same last ind the same pad is ok. therfore -2
    useIndB = np.where(vindB > vindT[-2])
    vindB = vindB[useIndB]
    #: height of cloud
    vheightT = height[wpind,vindT]
    vheightB = height[wpind,vindB]
#                                         #: make sure that the cloud is inside top bin
#                                         if  cloudtop[wpind] <= vheight[0]:
#                                             #: put actual cloud top in top bin
#                                             vheight[0] = cloudtop[wpind]
#                                         else:
#                                             print('fel cloud top')
#                                             sys.exit()
#                                         #: make sure that the cloud is inside base bin
#                                         if  cloudbase[wpind] >= vheightT[-1]:
#                                             #: put actual cloud base in bsae bin
#                                             vheight[-1] = cloudbase[wpind]
#                                         else:
#                                             print('fel cloud bottomT')
#                                             sys.exit()
    iwc_m_T = 1e-3 *(iwc[wpind,vindT][0:-1] + iwc[wpind, vindT][1:]) / 2
    lwc_m_T = 1e-3 *(lwc[wpind,vindT][0:-1] + lwc[wpind, vindT][1:]) / 2
    iwc_m_B = 1e-3 *(iwc[wpind,vindB][0:-1] + iwc[wpind, vindB][1:]) / 2
    lwc_m_B = 1e-3 *(lwc[wpind,vindB][0:-1] + lwc[wpind, vindB][1:]) / 2
    deltaHT = vheightT[0:-1] - vheightT[1:]
    deltaHB = vheightB[0:-1] - vheightB[1:]
    iwp_from_c_T = iwc_m_T * deltaHT
    lwp_from_c_T = lwc_m_T * deltaHT
    iwp_from_c_B = iwc_m_B * deltaHB
    lwp_from_c_B = lwc_m_B * deltaHB
    iwp_from_c_T = np.sum(iwp_from_c_T)
    lwp_from_c_T = np.sum(lwp_from_c_T)
    iwp_from_c_B = np.sum(iwp_from_c_B)
    lwp_from_c_B = np.sum(lwp_from_c_B)
    return iwp_from_c_T, lwp_from_c_T, iwp_from_c_B, lwp_from_c_B

def getOpticalDepthInd(ctp, otlimit, tclot_rowlat, tclot1D):
    
    if ctp == 11:
        opind = (tclot_rowlat > 0) & (tclot_rowlat <= otlimit[1])
        opind1D = (tclot1D > 0) & (tclot1D <= otlimit[1])
    elif ctp == 12:
        opind = (tclot_rowlat > otlimit[1]) & (tclot_rowlat <= otlimit[2])
        opind1D = (tclot1D > otlimit[1]) & (tclot1D <= otlimit[2])
    elif ctp == 13:
        opind = (tclot_rowlat > otlimit[2]) & (tclot_rowlat <= otlimit[3])
        opind1D = (tclot1D > otlimit[2]) & (tclot1D <= otlimit[3])
    elif ctp == 14:
        opind = (tclot_rowlat > otlimit[3]) & (tclot_rowlat <= otlimit[4])
        opind1D = (tclot1D > otlimit[3]) & (tclot1D <= otlimit[4])
    elif ctp == 15:
        opind = (tclot_rowlat > 0) & (tclot_rowlat <= otlimit[4])
        opind1D = (tclot1D > 0) & (tclot1D <= otlimit[4])
    elif ctp == 8:
        opind = tclot_rowlat >= 23
        opind1D = tclot1D >= 23
    elif ctp  == 27:
        opind = tclot_rowlat >= 3.6
        opind1D = tclot1D >= 3.6
    else:
        opind = np.ones(tclot_rowlat.shape).astype('bool')
        opind1D = np.ones(tclot1D.shape).astype('bool')
    
    return opind, opind1D


def  findValuedData(flxfile, clTfile, geoFile, tauFile, cwcFile, auxFile, differentCloudTypes, latrange, OTlimit, tempname, lt, fCt, onlyUseDict=False):
    loadname = tempname + '.npy'
    startCalc = False
    if lt:
        if not os.path.isfile(loadname):
            #: try to finds tempFiles undependent of the pattern, except right calipso file
            diffTempNames = glob.glob('%s*%s_y*m*w*_u%s.npy' %(tempname.split('_split')[0], fCt, tempname.split('u')[-1]))
            #: If finds file, use one oavsett what org loadname
            if len(diffTempNames) != 0:
                loadname = diffTempNames[0]
        #: If temp file exist and (deafault) options loadTemp, use temp files instead of new calculation
        if os.path.isfile(loadname):
            try:
                resDict, resDict2, resDict3, resDict4, resDict5 = np.load(loadname, allow_pickle=True)
                print('load')
            except:
                print('wrong load')
                startCalc = True
        else:
            startCalc = True
    else:
        startCalc = True
    if startCalc == False:
        return resDict, resDict2, resDict3, resDict4, resDict5
    
    
    
    
    if onlyUseDict:
        #: This one ca be used later for the specific dict
        #: as fore now the dict3 and dict4 can not be seperated
        resdictuse = [True,False]
    else:
        resdictuse = [True,True]
    liddata = readFLXHRLidar(flxfile)
    clcdata = readCloudType(clTfile)
    geodata = readGeoType(geoFile)
    taudata = readTauType(tauFile)
    cwcdata = readCwcType(cwcFile)
    auxdata = readAuxType(auxFile)
    if -1 in [liddata, clcdata]:
        return -1, -1, -1, -1, -1
    #: TODO kolla mon 11 y 07 clt 811 -d 6
    #: Prepering for resDict5
    resDict5 = {}
    #: 3D
    d3rd5 = np.zeros([125,len(latrange), 360], dtype=np.ndarray)
    #: 1D
    d1rd5 = np.zeros([1,len(latrange), 360], dtype=np.ndarray)
    for ii in range(125):
        for jj in range(len(latrange)):
            for kk in range(360):
                d3rd5[ii, jj, kk] = []
                if ii == 0:
                    d1rd5[ii, jj, kk] = []
    
    resDict = {}
    #: radiation and specific humidity (sh)
    for rad in ['shr', 'antal_shr', 'norm_shr', 'antal_norm_shr', 'lhr', 'antal_lhr', \
                'ref', 'antal_ref', 'shu', 'antal_shu', 'tem', 'antal_tem', \
                'sfd', 'antal_sfd', 'lfd', 'antal_lfd', 'sfdnc', 'antal_sfdnc', 'lfdnc', 'antal_lfdnc', \
                'sfu', 'antal_sfu', 'lfu', 'antal_lfu', 'sfunc', 'antal_sfunc', 'lfunc', 'antal_lfunc', \
                'iwc', 'antal_iwc', 'lwc', 'antal_lwc']:
        dn = ['d']
        if ('lhr' in rad) or ('ref' in rad) or ('shu' in rad) or ('tem' in rad):
            dn = ['d', 'n']
        for cld in  dn:
            for typ in differentCloudTypes:
                resDict.update({'%s_%s_%i' %(rad, cld, typ): np.zeros([125,len(latrange), 360])})
                if False: #: anvands ej ??
                    if rad in ['norm_shr', 'lhr', 'shu']:
                        resDict5.update({'%s_%s_%i' %(rad, cld, typ): d3rd5.copy()})
#     resDictT = {k:resDictT[k] + stDict[k] for k in resDictT}
    resDict2 = {}
    #: Cloud Fraction
    for cld in  ['d', 'n']:
        for typ in differentCloudTypes:
            resDict2.update({'%s_%i' %(cld, typ): np.zeros([125,len(latrange), 360])})
    
    resDict3 = {}
    if resdictuse[0]:
        #: boacare and topcare
        dn = ['d', 'n']
        for rad in ['tshr', 'antal_tshr', 'tlhr', 'antal_tlhr', \
                    'bshr', 'antal_bshr', 'blhr', 'antal_blhr']:
            for cld in  dn:
                for typ in differentCloudTypes:
                    resDict3.update({'%s_%s_%i' %(rad, cld, typ): np.zeros([1,len(latrange), 360])})
                    if not 'antal' in rad:
                        resDict5.update({'%s_%s_%i' %(rad, cld, typ): d1rd5.copy()})
    dn = ['d', 'n']
    for rad in ['clBTop', 'clBBase', 'clTTop', 'clTBase', 'otTop', 'otBase', 'otTotal', \
                'clDiff', 'iwcTop', 'iwcBase', 'lwcTop', 'lwcBase', 'iwpTotal', 'lwpTotal', \
                'iwpTop', 'iwpBase', 'lwpTop', 'lwpBase', 'lhrMeanTop', 'lhrMinTop', 'lhrMaxTop', \
                'shrNMeanTop', 'shrNMinTop', 'shrNMaxTop', 'netMeanTop', 'netMinTop', 'netMaxTop', \
                'shuMeanTop', 'shuMinTop', 'shuMaxTop', 'shuMeanBase', 'shuMinBase', 'shuMaxBase', \
                'shu3000', 'shu8000', 'shu12000']:
        for cld in dn:
            for typ in differentCloudTypes:
                if typ < 200 or typ > 900:
                    if rad in ['clTTop', 'clTBase', 'otTop', 'otBase', 'clDiff', \
                               'iwcTop', 'iwcBase', 'lwcTop', 'lwcBase', 'iwpTop', \
                               'iwpBase', 'lwpTop', 'lwpBase', 'lhrMeanTop', 'lhrMinTop', \
                               'lhrMaxTop', 'shrNMeanTop', 'shrNMinTop', 'shrNMaxTop', \
                               'netMeanTop', 'netMinTop', 'netMaxTop', 'shuMeanTop', \
                               'shuMinTop', 'shuMaxTop', 'shuMeanBase', 'shuMinBase', 'shuMaxBase']:
                        continue
                resDict5.update({'%s_%s_%i' %(rad, cld, typ): d1rd5.copy()})

    resDict4 = {}
    if resdictuse[1]:
        #: low medium and heigh clouds
        for rad in ['shr', 'antal_shr', 'norm_shr', 'antal_norm_shr', 'lhr', 'antal_lhr', 'antal_tot']:
            dn = ['d', 'n']
            if 'shr' in rad:
                dn = ['d']
            for cld in  dn:
                for typ in ['L_9', 'M_9', 'H_9', \
                            'L_90', 'M_90', 'H_90']:
                    resDict4.update({'%s_%s_%s' %(rad, cld, typ): np.zeros([1,len(latrange), 360])})
    
    QR = liddata.QR
    SolarZenitAngle = liddata.Solar_zenith_angle
    lat = liddata.latitude
    lon = liddata.longitude
    FD = liddata.FD
    FD_NC = liddata.FD_NC
    FU = liddata.FU
    FU_NC = liddata.FU_NC
    latstep = latrange[1] - latrange[0]
    for l in range(360):
        k = -1
        for latmin in latrange:
            k = k + 1
            latmax = latmin + latstep
            lonmin = l - 180 #: lon goes from -180 - 180. First place, i.e 0, means -180.
            lonmax = lonmin + 1
            if lonmin == -180:
                #: Make sure - 180 is inside boundaries. For lat there is an extra
                lonind = (lon >= lonmin) & (lon <= lonmax)
            else:
                lonind = (lon > lonmin) & (lon <= lonmax)
            #: TODO: Maybee should be (lat >= latmin) & (lat < latmax)
            lattup = (lat > latmin) & (lat <= latmax)
            latind = np.where(lattup & lonind)[0]
            if latind.shape[0] == 0:
                continue
            latlat = lat[latind]
            heighlat = liddata.Height[latind]
            szalat = SolarZenitAngle[latind]
            tidlat = liddata.sec1970[latind]
            shrlat = QR[0,latind,:]
            lhrlat = QR[1,latind,:]
            sfdlat = FD[0,latind,:]
            lfdlat = FD[1,latind,:]
            sfulat = FU[0,latind,:]
            lfulat = FU[1,latind,:]
            sfdnclat = FD_NC[0,latind,:]
            lfdnclat = FD_NC[1,latind,:]
            sfunclat = FU_NC[0,latind,:]
            lfunclat = FU_NC[1,latind,:]
            toashrlat = liddata.TOACRE[0,latind]
            toalhrlat = liddata.TOACRE[1,latind]
            boashrlat = liddata.BOACRE[0,latind]
            boalhrlat = liddata.BOACRE[1,latind]
            #: Cloud type data
            nrCLlat = clcdata['CL'][latind]
            clTYlat = clcdata['CLType'][:,0][latind]
            clTY2Dlat = clcdata['CLType'][latind]
            #: cloud base and top is in km while height is in m.
            clToplat = clcdata['CLTop'][latind, :] * 1000
            clBaselat = clcdata['CLBase'][latind, :] * 1000
            #: Reflectivity data
            if geodata['Radar_Reflectivity'].shape[0] == 0:
                reflat = geodata['Radar_Reflectivity']
            else:
                reflat = geodata['Radar_Reflectivity'][latind, :]            
            #: Tau data
            if taudata['layer_optical_depth'].shape[0] == 0:
                lodlat = np.zeros(heighlat.shape) * np.nan #: 2Dim
                todlat = np.zeros(latind.shape[0]) * np.nan #: 1Dim
            else:
                lodlat = taudata['layer_optical_depth'][latind, :]
                todlat = taudata['total_optical_depth'][latind]
            #: Cwc data
            if cwcdata['ice_water_content'].shape[0] == 0:
                iwclat = np.zeros(heighlat.shape) * np.nan #: 2Dim
                iwplat = np.zeros(latind.shape[0]) * np.nan #: 1Dim
                lwclat = np.zeros(heighlat.shape) * np.nan #: 2Dim
                lwplat = np.zeros(latind.shape[0]) * np.nan #: 1Dim
            else:
                iwclat = cwcdata['ice_water_content'][latind, :]
                iwplat = cwcdata['ice_water_path'][latind]
                lwclat = cwcdata['liq_water_content'][latind, :]
                lwplat = cwcdata['liq_water_path'][latind]
            
            if auxdata['Specific_humidity'].shape[0] == 0:
                shulat = np.zeros(heighlat.shape) * np.nan #: 2Dim
                shu3000lat = np.zeros(heighlat.shape[0]) * np.nan #: 1Dim
                shu8000lat = np.zeros(heighlat.shape[0]) * np.nan #: 1Dim
                shu12000lat = np.zeros(heighlat.shape[0]) * np.nan #: 1Dim
            else:
                shulat = auxdata['Specific_humidity'][latind, :] #: 2Dim
                #: Find the 3000, 8000, 12000 m level
                lev3000 = np.argmin(np.abs(heighlat - 3000), axis=1)
                lev8000 = np.argmin(np.abs(heighlat - 8000), axis=1)
                lev12000 = np.argmin(np.abs(heighlat - 12000), axis=1)
                #: Find the spec humid (SHU) at these levels
                shu3000lat = shulat[range(shulat.shape[0]),lev3000] #: 1Dim
                shu8000lat = shulat[range(shulat.shape[0]),lev8000]
                shu12000lat = shulat[range(shulat.shape[0]),lev12000]

            if auxdata['Temperature'].shape[0] == 0:
                temlat = np.zeros(heighlat.shape) * np.nan #: 2Dim
            else:
                temlat = auxdata['Temperature'][latind, :]
            factorlat = [shrfactor(szalat[f], latlat[f], tidlat[f]) for f in range(len(szalat))]
            factorlat = np.asarray(factorlat)
            
            for j in range(720,30001, 240):
                ih = int(j / 240.)
                if ih == 125:
                    ih = 124
                rowlat, collat = np.where((heighlat >= j) & (heighlat < (j + 240)))
                if len(rowlat) > len(factorlat):
                    # TODO: Check this further
                    factorlat = factorlat[rowlat]
    #                    insertInd = np.where((rowlat[1:] - rowlat[0:-1]) != 1)[0]
    #                    factorlat = np.insert(factorlat, insertInd+1, factorlat[insertInd])
                for ct in differentCloudTypes:
        #            nrCLlat
                    if ct == 0:
                        ctind = np.where(nrCLlat[rowlat] == 0)
                        ctind1D = np.where(nrCLlat == 0)
                    elif ct >= 1 and ct <=8:
                        ctind = np.where((nrCLlat[rowlat] == 1) & (clTYlat[rowlat] == ct))
                        ctind1D = np.where((nrCLlat == 1) & (clTYlat == ct))
                    elif ct == 9 or ct == 99:
                        ctind = np.where(nrCLlat[rowlat] > 0)
                        ctind1D = np.where(nrCLlat > 0)
                        if ct == 9:
                            ctindLMH = np.where(nrCLlat == 1)
                    elif ct == 10:
                        ctind = np.where(nrCLlat[rowlat] > 1)
                        ctind1D = np.where(nrCLlat > 1)
                    elif ct > 200 and ct < 900:
                        #: What ct in second layer are we looking for
                        secLayCt = int(ct/100)
                        #: What type of od are we looking for in highcloud
                        opCtO = ct - (secLayCt * 100)
                        #: if opCtO starts on 7 it is ct 27, therfore secLayCt2 is needed
                        #: if they ends with 6,7,8 they will be divided with distance between the top and bottom layer cloud
                        if opCtO in [70, 71, 72, 73, 74, 75]:
                            opCt = opCtO - 60
                            secLayCt2 = 7
                            secLayCtTot = 27
                        elif ct in [826, 827, 828]:
                            opCt = 11
                            secLayCtTot = 27
                        elif ct in [836, 837, 838]:
                            opCt = 12
                            secLayCtTot = 27
                        elif ct in [846, 847, 848]:
                            opCt = 13
                            secLayCtTot = 27
                        elif ct in [856, 857, 858]:
                            opCt = 14
                            secLayCtTot = 27
                        elif ct in [876, 877, 878]:
                            opCt = 15
                            secLayCtTot = 27
                        else:
                            opCt = opCtO
                            secLayCtTot = secLayCt
                        #: TODO: Fixa pa battrre satt
                        #: Osaker om detta stammer. Ska inte sec layer ha od limit nar det ar 0 pa slutet?
                        if ct in [810, 270, 870]:
                            secLayCtTot = ct
                        #: cloud base and top is in km while height is in m /fixed.
                        cBT = clBaselat[:, 1]
                        cTT = clToplat[:, 1]
                        cBB = clBaselat[:, 0]
                        cTB = clToplat[:, 0]
                        #: Take differense between height and cloud base/top
                        cTmT = (heighlat - cTT[:,None])
                        cBmT = (cBT[:,None] - heighlat)
                        cTmB = (heighlat - cTB[:,None])
                        cBmB = (cBB[:,None] - heighlat)
                        # TODO: I remove error message
                        np.seterr(invalid='ignore')
                        #: Find where difference is below 120m means we find closest since differense is 240m in height
#                         cIndT = (cTmT<=120) & (cBmT<=120)
#                         cIndB = (cTmB<=120) & (cBmB<=120)
                        #: Anvand hela bins for od och i/lwc/p
                        cIndT = (cTmT<240) & (cBmT<=240)
                        cIndB = (cTmB<240) & (cBmB<=240)
#                         if (nrCLlat[rowlat] == 2).any():
#                             pdb.set_trace()
                        #: Calculate sum of optical depth for height bin inside cloud
                        try:
                            odvalTlat = np.sum((np.ma.masked_array(lodlat, ~cIndT)) ,axis=1).data
                            iwcvalTlat = np.sum((np.ma.masked_array(iwclat, ~cIndT)) ,axis=1).data
                            lwcvalTlat = np.sum((np.ma.masked_array(lwclat, ~cIndT)) ,axis=1).data
                        except:
                            print('odvalT')
                        try:
                            odvalBlat = np.sum((np.ma.masked_array(lodlat, ~cIndB)) ,axis=1).data
                            iwcvalBlat = np.sum((np.ma.masked_array(iwclat, ~cIndB)) ,axis=1).data
                            lwcvalBlat = np.sum((np.ma.masked_array(lwclat, ~cIndB)) ,axis=1).data
                        except:
                            print('odvalB')
                        #: Use od threshold to deside what cloud to use
                        opindT, opind1DT = getOpticalDepthInd(opCt, OTlimit, odvalTlat[rowlat], odvalTlat)
                        opindB, opind1DB = getOpticalDepthInd(secLayCtTot, OTlimit, odvalBlat[rowlat], odvalBlat)
                        #: Make sure second layer is secLayCt --in case 27-- second layer is secLayCt or secLayCt2
                        if (ct >= 820) and (ct < 880):#in [870, 871, 872, 873, 874, 875, 876, 877, 878]:
                            seLayTrue = (cTB[rowlat] >= 3500)
                            seLayTrue1D = (cTB >= 3500)
                            if int(str(ct)[-1]) in [6, 7, 8]:
                                diff = cBT - cTB
                                if int(str(ct)[-1]) == 6:
                                    diffTrue = diff <= 1000
                                elif int(str(ct)[-1]) == 7:
                                    diffTrue = (diff > 1000) & (diff <= 3000)
                                elif int(str(ct)[-1]) == 8:
                                    diffTrue = diff > 3000
                                seLayTrue = seLayTrue & diffTrue[rowlat]
                                seLayTrue1D = seLayTrue1D & diffTrue
#                                 if not (seLayTrue == False).all():
#                                     pdb.set_trace()
                        elif opCtO in [70, 71, 72, 73, 74, 75]:
                            seLayTrue = (clTY2Dlat[rowlat][:,0] == secLayCt) | (clTY2Dlat[rowlat][:,0] == secLayCt2)
                            seLayTrue1D = (clTY2Dlat[:,0] == secLayCt) | (clTY2Dlat[:,0] == secLayCt2)
                        else:
                            seLayTrue = (clTY2Dlat[rowlat][:,0] == secLayCt)
                            seLayTrue1D = (clTY2Dlat[:,0] == secLayCt)
                        #: Two layer & toplayer is highcloud (1) & second layer is secLayCt & optical depth threshold
                        ctind = np.where((nrCLlat[rowlat] == 2) & (clTY2Dlat[rowlat][:,1] == 1) & seLayTrue & opindT & opindB)
                        ctind1D = np.where((nrCLlat == 2) & (clTY2Dlat[:,1] == 1) & seLayTrue1D & opind1DT & opind1DB)
                        np.seterr(invalid='warn')
#                         if (ctind[0].shape[0] != 0):
#                                 pdb.set_trace()
#                             if not (ctind1D[0] == ctind[0]).all():
                        
                    elif ct == 27:
                        ctind = np.where((nrCLlat[rowlat] == 1) & ((clTYlat[rowlat] == 2) | (clTYlat[rowlat] == 7)))
                        ctind1D = np.where((nrCLlat == 1) & ((clTYlat == 2) | (clTYlat == 7)))
                    elif ct == 90:
                        ctind = range(len(rowlat))
                        ctind1D = range(len(latind))
                        ctindLMH = range(len(latind))
                    elif ct in [11, 12, 13, 14, 15]:
                        # TODO: I remove error message
                        np.seterr(invalid='ignore')
                        opind, opind1D = getOpticalDepthInd(ct, OTlimit, todlat[rowlat], todlat)
                        ctind = np.where((nrCLlat[rowlat] == 1) & (clTYlat[rowlat] == 1) & opind)
                        ctind1D = np.where((nrCLlat == 1) & (clTYlat == 1) & opind1D)
                        np.seterr(invalid='warn')
#                         if np.where((nrCLlat[rowlat] == 1) & (clTYlat[rowlat] == 1))[0].shape[0] != 0:
#                             pdb.set_trace()
                    elif ct in [18, 127, 187]:
                        # TODO: I remove error message
                        np.seterr(invalid='ignore')
                        if ct == 187:
                            cto = 27
                            opind, opind1D = getOpticalDepthInd(cto, OTlimit, todlat[rowlat], todlat)
                            #: nr of row = 1, cloud top height > 3500 and optical thicknes
                            ctind = np.where((nrCLlat[rowlat] == 1) & (clToplat[:, 0][rowlat] >= 3500) & opind)
                            ctind1D = np.where((nrCLlat == 1) & (clToplat[:, 0] >= 3500) & opind1D)
                        elif ct == 18:
                            cto = 8
                            opind, opind1D = getOpticalDepthInd(cto, OTlimit, todlat[rowlat], todlat)
                            #: nr of row = 1, cltyp = 8 and optical thicknes
                            ctind = np.where((nrCLlat[rowlat] == 1) & (clTYlat[rowlat] == cto) & opind)
                            ctind1D = np.where((nrCLlat == 1) & (clTYlat == cto) & opind1D)
                        elif ct == 127:
                            cto = 27
                            opind, opind1D = getOpticalDepthInd(cto, OTlimit, todlat[rowlat], todlat)
                            ctind = np.where((nrCLlat[rowlat] == 1) & ((clTYlat[rowlat] == 2) | (clTYlat[rowlat] == 7)) & opind)
                            ctind1D = np.where((nrCLlat == 1) & ((clTYlat == 2) | (clTYlat == 7)) & opind1D)
#                         if (ctind[0].shape[0] != 0):
#                             pdb.set_trace()
                        np.seterr(invalid='warn')
                    else:
                        print(ct)
                        print('Fel molntyp')
                    
                    # 1-D #
                    d1s = 0
                    if isinstance(ctind1D,list):
                        if len(ctind1D) != 0:
                            d1s = len(ctind1D)
                    else:
                        if ctind1D[0].shape[0] != 0:
                            d1s = ctind1D[0].shape[0]
                            ctind1D = ctind1D[0]

                    if j == 720:
                        if (d1s != 0):
                            dagtb = np.zeros([d1s]).astype('bool')
                            for i in range(d1s):
                                cti = ctind1D[i]
                                shrlat_nan = np.where(np.isnan(shrlat), -1, shrlat)
                                if (shrlat_nan[cti,:] > 0).any():
                                    dagtb[i] = True
                            #: TODO: nattb is when there is no dagtb
                            nattb = ~dagtb
                            toashr = toashrlat[ctind1D]
                            toalhr = toalhrlat[ctind1D]
                            boashr = boashrlat[ctind1D]
                            boalhr = boalhrlat[ctind1D]

                            toashr_d, toashr_dn, toashr_dar, toashr_n, toashr_nn, toashr_nar = countVal(toashr, dagtb, nattb, True)
                            resDict3['tshr_d_%i' %(ct)][0, k, l] = resDict3['tshr_d_%i' %(ct)][0, k, l] + toashr_d
                            resDict3['antal_tshr_d_%i' %(ct)][0, k, l] = resDict3['antal_tshr_d_%i' %(ct)][0, k, l] + toashr_dn
                            resDict3['tshr_n_%i' %(ct)][0, k, l] = resDict3['tshr_n_%i' %(ct)][0, k, l] + toashr_n
                            resDict3['antal_tshr_n_%i' %(ct)][0, k, l] = resDict3['antal_tshr_n_%i' %(ct)][0, k, l] + toashr_nn
                            toalhr_d, toalhr_dn, toalhr_dar, toalhr_n, toalhr_nn, toalhr_nar = countVal(toalhr, dagtb, nattb, True)
                            resDict3['tlhr_d_%i' %(ct)][0, k, l] = resDict3['tlhr_d_%i' %(ct)][0, k, l] + toalhr_d
                            resDict3['antal_tlhr_d_%i' %(ct)][0, k, l] = resDict3['antal_tlhr_d_%i' %(ct)][0, k, l] + toalhr_dn
                            resDict3['tlhr_n_%i' %(ct)][0, k, l] = resDict3['tlhr_n_%i' %(ct)][0, k, l] + toalhr_n
                            resDict3['antal_tlhr_n_%i' %(ct)][0, k, l] = resDict3['antal_tlhr_n_%i' %(ct)][0, k, l] + toalhr_nn
                            boashr_d, boashr_dn, boashr_dar, boashr_n, boashr_nn, boashr_nar = countVal(boashr, dagtb, nattb, True)
                            resDict3['bshr_d_%i' %(ct)][0, k, l] = resDict3['bshr_d_%i' %(ct)][0, k, l] + boashr_d
                            resDict3['antal_bshr_d_%i' %(ct)][0, k, l] = resDict3['antal_bshr_d_%i' %(ct)][0, k, l] + boashr_dn
                            resDict3['bshr_n_%i' %(ct)][0, k, l] = resDict3['bshr_n_%i' %(ct)][0, k, l] + boashr_n
                            resDict3['antal_bshr_n_%i' %(ct)][0, k, l] = resDict3['antal_bshr_n_%i' %(ct)][0, k, l] + boashr_nn
                            boalhr_d, boalhr_dn, boalhr_dar, boalhr_n, boalhr_nn, boalhr_nar = countVal(boalhr, dagtb, nattb, True)
                            resDict3['blhr_d_%i' %(ct)][0, k, l] = resDict3['blhr_d_%i' %(ct)][0, k, l] + boalhr_d
                            resDict3['antal_blhr_d_%i' %(ct)][0, k, l] = resDict3['antal_blhr_d_%i' %(ct)][0, k, l] + boalhr_dn
                            resDict3['blhr_n_%i' %(ct)][0, k, l] = resDict3['blhr_n_%i' %(ct)][0, k, l] + boalhr_n
                            resDict3['antal_blhr_n_%i' %(ct)][0, k, l] = resDict3['antal_blhr_n_%i' %(ct)][0, k, l] + boalhr_nn
                            if ct != 90:
                                resDict5['tshr_d_%i' %(ct)][0, k, l]= np.hstack((resDict5['tshr_d_%i' %(ct)][0, k, l], toashr_dar))
                                resDict5['tshr_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['tshr_n_%i' %(ct)][0, k, l], toashr_nar))
                                resDict5['tlhr_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['tlhr_d_%i' %(ct)][0, k, l], toalhr_dar))
                                resDict5['tlhr_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['tlhr_n_%i' %(ct)][0, k, l], toalhr_nar))
                                resDict5['bshr_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['bshr_d_%i' %(ct)][0, k, l], boashr_dar))
                                resDict5['bshr_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['bshr_n_%i' %(ct)][0, k, l], boashr_nar))
                                resDict5['blhr_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['blhr_d_%i' %(ct)][0, k, l], boalhr_dar))
                                resDict5['blhr_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['blhr_n_%i' %(ct)][0, k, l], boalhr_nar))
                                #: SHU 3000
                                shu3 = shu3000lat[ctind1D]
                                shu3_d, shu3_dn, shu3_dar, shu3_n, shu3_nn, shu3_nar = countVal(shu3, dagtb, nattb, True)  # @UnusedVariable
                                resDict5['shu3000_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shu3000_d_%i' %(ct)][0, k, l], shu3_dar))
                                resDict5['shu3000_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shu3000_n_%i' %(ct)][0, k, l], shu3_nar))
                                #: SHU 8000
                                shu8 = shu8000lat[ctind1D]
                                shu8_d, shu8_dn, shu8_dar, shu8_n, shu8_nn, shu8_nar = countVal(shu8, dagtb, nattb, True)  # @UnusedVariable
                                resDict5['shu8000_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shu8000_d_%i' %(ct)][0, k, l], shu8_dar))
                                resDict5['shu8000_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shu8000_n_%i' %(ct)][0, k, l], shu8_nar))
                                #: SHU 12000
                                shu12 = shu12000lat[ctind1D]
                                shu12_d, shu12_dn, shu12_dar, shu12_n, shu12_nn, shu12_nar = countVal(shu12, dagtb, nattb, True)  # @UnusedVariable
                                resDict5['shu12000_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shu12000_d_%i' %(ct)][0, k, l], shu12_dar))
                                resDict5['shu12000_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shu12000_n_%i' %(ct)][0, k, l], shu12_nar))
                                #: Optical thicknes total
                                tod = todlat[ctind1D]
                                tod_d, tod_dn, tod_dar, tod_n, tod_nn, tod_nar = countVal(tod, dagtb, nattb, True)  # @UnusedVariable
                                resDict5['otTotal_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['otTotal_d_%i' %(ct)][0, k, l], tod_dar))
                                resDict5['otTotal_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['otTotal_n_%i' %(ct)][0, k, l], tod_nar))
                                #: IWP (total)
                                iwp = iwplat[ctind1D]
                                iwp_d, iwp_dn, iwp_dar, iwp_n, iwp_nn, iwp_nar = countVal(iwp, dagtb, nattb, True)  # @UnusedVariable
                                resDict5['iwpTotal_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwpTotal_d_%i' %(ct)][0, k, l], iwp_dar))
                                resDict5['iwpTotal_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwpTotal_n_%i' %(ct)][0, k, l], iwp_nar))
                                #: LWP (total)
                                lwp = lwplat[ctind1D]
                                lwp_d, lwp_dn, lwp_dar, lwp_n, lwp_nn, lwp_nar = countVal(lwp, dagtb, nattb, True)  # @UnusedVariable
                                resDict5['lwpTotal_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwpTotal_d_%i' %(ct)][0, k, l], lwp_dar))
                                resDict5['lwpTotal_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwpTotal_n_%i' %(ct)][0, k, l], lwp_nar))
                                if ct not in [9, 10, 0]:
                                    #: Cloud Top and Base of base- or single- layer cloud
                                    clBTop = clToplat[ctind1D, 0]
                                    clBBase = clBaselat[ctind1D, 0]
                                    clBTop_d, clBTop_dn, clBTop_dar, clBTop_n, clBTop_nn, clBTop_nar = countVal(clBTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['clBTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['clBTop_d_%i' %(ct)][0, k, l], clBTop_dar))
                                    resDict5['clBTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['clBTop_n_%i' %(ct)][0, k, l], clBTop_nar))
                                    clBBase_d, clBBase_dn, clBBase_dar, clBBase_n, clBBase_nn, clBBase_nar = countVal(clBBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['clBBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['clBBase_d_%i' %(ct)][0, k, l], clBBase_dar))
                                    resDict5['clBBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['clBBase_n_%i' %(ct)][0, k, l], clBBase_nar))
                                if ct > 200 and ct < 900:
                                    #: Cloud Top and Base of top layer cloud
                                    clTTop = clToplat[ctind1D, 1]
                                    clTBase = clBaselat[ctind1D, 1]
                                    clTTop_d, clTTop_dn, clTTop_dar, clTTop_n, clTTop_nn, clTTop_nar = countVal(clTTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['clTTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['clTTop_d_%i' %(ct)][0, k, l], clTTop_dar))
                                    resDict5['clTTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['clTTop_n_%i' %(ct)][0, k, l], clTTop_nar))
                                    clTBase_d, clTBase_dn, clTBase_dar, clTBase_n, clTBase_nn, clTBase_nar = countVal(clTBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['clTBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['clTBase_d_%i' %(ct)][0, k, l], clTBase_dar))
                                    resDict5['clTBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['clTBase_n_%i' %(ct)][0, k, l], clTBase_nar))
                                    #: Differnce [m] between cloud-abowe and cloud-below
                                    clDiff = clTBase - clBTop
                                    clDiff_d, clDiff_dn, clDiff_dar, clDiff_n, clDiff_nn, clDiff_nar = countVal(clDiff, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['clDiff_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['clDiff_d_%i' %(ct)][0, k, l], clDiff_dar))
                                    resDict5['clDiff_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['clDiff_n_%i' %(ct)][0, k, l], clDiff_nar))
                                    #: Optical thickness cloud-abowe
                                    otTop = odvalTlat[ctind1D]
                                    otTop_d, otTop_dn, otTop_dar, otTop_n, otTop_nn, otTop_nar = countVal(otTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['otTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['otTop_d_%i' %(ct)][0, k, l], otTop_dar))
                                    resDict5['otTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['otTop_n_%i' %(ct)][0, k, l], otTop_nar))
                                    #: Optical thickness cloud-below
                                    otBase = odvalBlat[ctind1D]
                                    otBase_d, otBase_dn, otBase_dar, otBase_n, otBase_nn, otBase_nar = countVal(otBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['otBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['otBase_d_%i' %(ct)][0, k, l], otBase_dar))
                                    resDict5['otBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['otBase_n_%i' %(ct)][0, k, l], otBase_nar))
#                                     #: Difference cloud-abowe and total ot (for whole column) This could be the ot for cloud-below
#                                     otDiff = tod - otTop
#                                     otDiff_d, otDiff_dn, otDiff_dar, otDiff_n, otDiff_nn, otDiff_nar = countVal(otDiff, dagtb, nattb, True)  # @UnusedVariable
#                                     resDict5['otDiff_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['otDiff_d_%i' %(ct)][0, k, l], otDiff_dar))
#                                     resDict5['otDiff_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['otDiff_n_%i' %(ct)][0, k, l], otDiff_nar))
                                    #: IWC for cloud-abowe
                                    iwcTop = iwcvalTlat[ctind1D]
                                    iwcTop_d, iwcTop_dn, iwcTop_dar, iwcTop_n, iwcTop_nn, iwcTop_nar = countVal(iwcTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['iwcTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwcTop_d_%i' %(ct)][0, k, l], iwcTop_dar))
                                    resDict5['iwcTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwcTop_n_%i' %(ct)][0, k, l], iwcTop_nar))
                                    #: IWC for cloud-below
                                    iwcBase = iwcvalBlat[ctind1D]
                                    iwcBase_d, iwcBase_dn, iwcBase_dar, iwcBase_n, iwcBase_nn, iwcBase_nar = countVal(iwcBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['iwcBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwcBase_d_%i' %(ct)][0, k, l], iwcBase_dar))
                                    resDict5['iwcBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwcBase_n_%i' %(ct)][0, k, l], iwcBase_nar))
                                    #: LWC for cloud-abowe
                                    lwcTop = lwcvalTlat[ctind1D]
                                    lwcTop_d, lwcTop_dn, lwcTop_dar, lwcTop_n, lwcTop_nn, lwcTop_nar = countVal(lwcTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['lwcTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwcTop_d_%i' %(ct)][0, k, l], lwcTop_dar))
                                    resDict5['lwcTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwcTop_n_%i' %(ct)][0, k, l], lwcTop_nar))
                                    #: LWC for cloud-below
                                    lwcBase = lwcvalBlat[ctind1D]
                                    lwcBase_d, lwcBase_dn, lwcBase_dar, lwcBase_n, lwcBase_nn, lwcBase_nar = countVal(lwcBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['lwcBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwcBase_d_%i' %(ct)][0, k, l], lwcBase_dar))
                                    resDict5['lwcBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwcBase_n_%i' %(ct)][0, k, l], lwcBase_nar))
#                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 28., 56., 83., 107., 137., 163., 185., 205., 236., 261., 268., 262., 237., 207., 176., 142., 103., 68., 45., 33., 25., 19., 16., 13., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
#                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11., 28., 58., 74., 93., 166., 335., 586., 759., 801., 731., 619., 537., 519., 627., 730., 682., 540., 485., 463., 425., 424., 397., 353., 310., 289., 251., 183., 106., 38., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
#                                     [25039., 24799., 24560., 24320., 24080., 23840., 23600., 23360., 23121., 22881., 22641., 22401., 22161., 21922., 21682., 21442., 21202., 20962., 20722., 20483., 20243., 20003., 19763., 19523., 19283., 19044., 18804., 18564., 18324., 18084., 17844., 17605., 17365., 17125., 16885., 16645., 16405., 16166., 15926., 15686., 15446., 15206., 14966., 14727., 14487., 14247., 14007., 13767., 13528., 13288., 13048., 12808., 12568., 12328., 12089., 11849., 11609., 11369., 11129., 10889., 10650., 10410., 10170., 9930., 9690., 9450., 9211., 8971., 8731., 8491., 8251., 8011., 7772., 7532., 7292., 7052., 6812., 6572., 6333., 6093., 5853., 5613., 5373., 5133., 4894., 4654., 4414., 4174., 3934., 3695., 3455., 3215., 2975., 2735., 2495., 2256., 2016., 1776., 1536., 1296., 1056., 817., 577., 337., 97., -143., -383., -622., -862., -1102., -1342., -1582., -1822., -2061., -2301., -2541., -2781., -3021., -3261., -3500., -3740., -3980., -4220., -4460., -4700.]
#                                     741.0
#                                     2788.0
                                    
                                    
                                    iwpTop = np.zeros(len(ctind1D))
                                    lwpTop = np.zeros(len(ctind1D))
                                    iwpBase = np.zeros(len(ctind1D))
                                    lwpBase = np.zeros(len(ctind1D))
                                    
                                    shuMinTop = np.zeros(len(ctind1D))
                                    shuMaxTop = np.zeros(len(ctind1D))
                                    shuMeanTop = np.zeros(len(ctind1D))
                                    
                                    shuMinBase = np.zeros(len(ctind1D))
                                    shuMaxBase = np.zeros(len(ctind1D))
                                    shuMeanBase = np.zeros(len(ctind1D))
                                    
                                    shrNMinTop = np.zeros(len(ctind1D))
                                    shrNMaxTop = np.zeros(len(ctind1D))
                                    shrNMeanTop = np.zeros(len(ctind1D))
                                    
                                    lhrMinTop = np.zeros(len(ctind1D))
                                    lhrMaxTop = np.zeros(len(ctind1D))
                                    lhrMeanTop = np.zeros(len(ctind1D))
                                    
                                    netMinTop = np.zeros(len(ctind1D))
                                    netMaxTop = np.zeros(len(ctind1D))
                                    netMeanTop = np.zeros(len(ctind1D))
                                    wpi = -1
                                    warnings.simplefilter("ignore", category=RuntimeWarning)
                                    for wpind in ctind1D:
                                        wpi = wpi + 1
                                        shr_c_T = shrlat[wpind][cIndT[wpind]]
                                        shrN_c_T = shr_c_T * factorlat[wpind]
                                        lhr_c_T = lhrlat[wpind][cIndT[wpind]]
                                        net_c_T = shrN_c_T + lhr_c_T
                                        shu_c_T = shulat[wpind][cIndT[wpind]]
                                        shu_c_B = shulat[wpind][cIndB[wpind]]
                                        shrNMinTop[wpi] = np.nanmin(shrN_c_T)
                                        shrNMaxTop[wpi] = np.nanmax(shrN_c_T)
                                        shrNMeanTop[wpi] = np.nanmean(shrN_c_T)
                                        
                                        lhrMinTop[wpi] = np.nanmin(lhr_c_T)
                                        lhrMaxTop[wpi] = np.nanmax(lhr_c_T)
                                        lhrMeanTop[wpi] = np.nanmean(lhr_c_T)
                                        
                                        netMinTop[wpi] = np.nanmin(net_c_T)
                                        netMaxTop[wpi] = np.nanmax(net_c_T)
                                        netMeanTop[wpi] = np.nanmean(net_c_T)
                                        
                                        shuMinTop[wpi] = np.nanmin(shu_c_T)
                                        shuMaxTop[wpi] = np.nanmax(shu_c_T)
                                        shuMeanTop[wpi] = np.nanmean(shu_c_T)
                                        
                                        shuMinBase[wpi] = np.nanmin(shu_c_B)
                                        shuMaxBase[wpi] = np.nanmax(shu_c_B)
                                        shuMeanBase[wpi] = np.nanmean(shu_c_B)
                                        
                                        iwp_from_c_T, lwp_from_c_T, iwp_from_c_B, lwp_from_c_B = calcWP(wpind, cIndT, cIndB, heighlat, iwclat, lwclat)
                                        
                                        
#                                         vindT =  np.where(cIndT[wpind])[0]
#                                         vindB =  np.where(cIndB[wpind])[0]
#                                         #: Add ap pad
#                                         vindT = np.asarray(range(vindT[0]-1, vindT[-1] + 2))
#                                         vindB = np.asarray(range(vindB[0]-1, vindB[-1] + 2))
#                                         #: Make sure that they do not have the same last ind the same pas is ok. therfore -2
#                                         useIndB = np.where(vindB > vindT[-2])
#                                         vindB = vindB[useIndB]
#                                         vheightT = heighlat[wpind,vindT]
#                                         vheightB = heighlat[wpind,vindB]
# #                                         if  cTT[wpind] <= vheightT[0]:
# #                                             vheightT[0] = cTT[wpind]
# #                                         else:
# #                                             print('fel cloud topT')
# #                                             sys.exit()
# #                                         if  cBT[wpind] >= vheightT[-1]:
# #                                             vheightT[0] = cBT[wpind]
# #                                         else:
# #                                             print('fel cloud bottomT')
# #                                             sys.exit()
# #                                         
# #                                         if  cTB[wpind] <= vheightB[0]:
# #                                             vheightB[0] = cTB[wpind]
# #                                         else:
# #                                             print('fel cloud topB')
# #                                             sys.exit()
# #                                         if  cBT[wpind] >= vheightB[-1]:
# #                                             vheightT[0] = cTT[wpind]
# #                                         else:
# #                                             print('fel cloud bottomT')
# #                                             sys.exit()
# #                                         cTT[wpind]
# #                                         cBT[wpind]
#                                         if vindT.sum() == 1:
#                                             iwp_from_c_T = -1
#                                             lwp_from_c_T = -1
#                                         else:
#                                             iwc_m_T = 1e-3 *(iwclat[wpind,vindT][0:-1] + iwclat[wpind,vindT][1:]) / 2
#                                             lwc_m_T = 1e-3 *(lwclat[wpind,vindT][0:-1] + lwclat[wpind, vindT][1:]) / 2
#                                             deltaH_T = vheightT[0:-1] - vheightT[1:]
#                                             iwp_from_c_T = iwc_m_T * deltaH_T
#                                             lwp_from_c_T = lwc_m_T * deltaH_T
#                                             iwp_from_c_T = np.sum(iwp_from_c_T)
#                                             lwp_from_c_T = np.sum(lwp_from_c_T)
#                                         if vindB.sum() == 1:
#                                             iwp_from_c_B = -1
#                                             lwp_from_c_B = -1
#                                         else:
#                                             iwc_m_B = 1e-3 *(iwclat[wpind,vindB][0:-1] + iwclat[wpind,vindB][1:]) / 2
#                                             lwc_m_B = 1e-3 *(lwclat[wpind,vindB][0:-1] + lwclat[wpind, vindB][1:]) / 2
#                                             deltaH_B = vheightB[0:-1] - vheightB[1:]
#                                             iwp_from_c_B = iwc_m_B * deltaH_B
#                                             lwp_from_c_B = lwc_m_B * deltaH_B
#                                             iwp_from_c_B = np.sum(iwp_from_c_B)
#                                             lwp_from_c_B = np.sum(lwp_from_c_B)
#                                         
#                                         iwpTtemp, lwpTtemp, iwpBtemp, lwpBtemp = calcWP(wpind, cIndT, cIndB, heighlat, iwclat, lwclat)
#                                         
#                                         if (iwpTtemp != iwp_from_c_T):
#                                             if not (np.isnan(iwpTtemp) and np.isnan(iwp_from_c_T)):
#                                                 print('fel i calc I T')
#                                                 pdb.set_trace()
#                                         if (iwpBtemp != iwp_from_c_B):
#                                             if not (np.isnan(iwpBtemp) and np.isnan(iwp_from_c_B)):
#                                                 print('fel i calc I B')
#                                                 pdb.set_trace()
#                                         if (lwpTtemp != lwp_from_c_T):
#                                             if not (np.isnan(lwpTtemp) and np.isnan(lwp_from_c_T)):
#                                                 print('fel i calc L T')
#                                                 pdb.set_trace()
#                                         if (lwpBtemp != lwp_from_c_B):
#                                             if not (np.isnan(lwpBtemp) and np.isnan(lwp_from_c_B)):
#                                                 print('fel i calc L B')
#                                                 pdb.set_trace()
                                        
                                        
                                                
                                        iwpTop[wpi] = iwp_from_c_T
                                        lwpTop[wpi] = lwp_from_c_T
                                        iwpBase[wpi] = iwp_from_c_B
                                        lwpBase[wpi] = lwp_from_c_B
                                        if False:
                                            if np.abs(iwplat[wpind] - (iwp_from_c_T + iwp_from_c_B)) > 10:
                                                print('fel i ice water path')
                                                print(iwplat[wpind] - (iwp_from_c_T + iwp_from_c_B))
    #                                             pdb.set_trace()
                                            if np.abs(lwplat[wpind] - (lwp_from_c_T + lwp_from_c_B)) > 20:
                                                print('fel i liq water path')
                                                print(lwplat[wpind] - (lwp_from_c_T + lwp_from_c_B))
    #                                             pdb.set_trace()
                                    warnings.resetwarnings()    
                                    #: IWP for cloud-abowe
                                    iwpTop_d, iwpTop_dn, iwpTop_dar, iwpTop_n, iwpTop_nn, iwpTop_nar = countVal(iwpTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['iwpTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwpTop_d_%i' %(ct)][0, k, l], iwpTop_dar))
                                    resDict5['iwpTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwpTop_n_%i' %(ct)][0, k, l], iwpTop_nar))
                                    #: IWP for cloud-below
                                    iwpBase_d, iwpBase_dn, iwpBase_dar, iwpBase_n, iwpBase_nn, iwpBase_nar = countVal(iwpBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['iwpBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwpBase_d_%i' %(ct)][0, k, l], iwpBase_dar))
                                    resDict5['iwpBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['iwpBase_n_%i' %(ct)][0, k, l], iwpBase_nar))
                                    #: LWP for cloud-abowe
                                    lwpTop_d, lwpTop_dn, lwpTop_dar, lwpTop_n, lwpTop_nn, lwpTop_nar = countVal(lwpTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['lwpTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwpTop_d_%i' %(ct)][0, k, l], lwpTop_dar))
                                    resDict5['lwpTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwpTop_n_%i' %(ct)][0, k, l], lwpTop_nar))
                                    #: LWP for cloud-below
                                    lwpBase_d, lwpBase_dn, lwpBase_dar, lwpBase_n, lwpBase_nn, lwpBase_nar = countVal(lwpBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['lwpBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwpBase_d_%i' %(ct)][0, k, l], lwpBase_dar))
                                    resDict5['lwpBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['lwpBase_n_%i' %(ct)][0, k, l], lwpBase_nar))
                                    if len(iwpTop_dar) != len(otBase_dar):
                                        print('fel langd')
                                        
                                    #: shrNMin for cloud-abowe
                                    shrNMinTop_d, shrNMinTop_dn, shrNMinTop_dar, shrNMinTop_n, shrNMinTop_nn, shrNMinTop_nar = countVal(shrNMinTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shrNMinTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shrNMinTop_d_%i' %(ct)][0, k, l], shrNMinTop_dar))
                                    resDict5['shrNMinTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shrNMinTop_n_%i' %(ct)][0, k, l], shrNMinTop_nar))
                                    #: shrNMax for cloud-abowe
                                    shrNMaxTop_d, shrNMaxTop_dn, shrNMaxTop_dar, shrNMaxTop_n, shrNMaxTop_nn, shrNMaxTop_nar = countVal(shrNMaxTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shrNMaxTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shrNMaxTop_d_%i' %(ct)][0, k, l], shrNMaxTop_dar))
                                    resDict5['shrNMaxTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shrNMaxTop_n_%i' %(ct)][0, k, l], shrNMaxTop_nar))
                                    #: shrNMean for cloud-abowe
                                    shrNMeanTop_d, shrNMeanTop_dn, shrNMeanTop_dar, shrNMeanTop_n, shrNMeanTop_nn, shrNMeanTop_nar = countVal(shrNMeanTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shrNMeanTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shrNMeanTop_d_%i' %(ct)][0, k, l], shrNMeanTop_dar))
                                    resDict5['shrNMeanTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shrNMeanTop_n_%i' %(ct)][0, k, l], shrNMeanTop_nar))
                                    #: lhrMin for cloud-abowe
                                    lhrMinTop_d, lhrMinTop_dn, lhrMinTop_dar, lhrMinTop_n, lhrMinTop_nn, lhrMinTop_nar = countVal(lhrMinTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['lhrMinTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['lhrMinTop_d_%i' %(ct)][0, k, l], lhrMinTop_dar))
                                    resDict5['lhrMinTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['lhrMinTop_n_%i' %(ct)][0, k, l], lhrMinTop_nar))
                                    #: lhrMax for cloud-abowe
                                    lhrMaxTop_d, lhrMaxTop_dn, lhrMaxTop_dar, lhrMaxTop_n, lhrMaxTop_nn, lhrMaxTop_nar = countVal(lhrMaxTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['lhrMaxTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['lhrMaxTop_d_%i' %(ct)][0, k, l], lhrMaxTop_dar))
                                    resDict5['lhrMaxTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['lhrMaxTop_n_%i' %(ct)][0, k, l], lhrMaxTop_nar))
                                    #: lhrMean for cloud-abowe
                                    lhrMeanTop_d, lhrMeanTop_dn, lhrMeanTop_dar, lhrMeanTop_n, lhrMeanTop_nn, lhrMeanTop_nar = countVal(lhrMeanTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['lhrMeanTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['lhrMeanTop_d_%i' %(ct)][0, k, l], lhrMeanTop_dar))
                                    resDict5['lhrMeanTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['lhrMeanTop_n_%i' %(ct)][0, k, l], lhrMeanTop_nar))
                                    #: netMin for cloud-abowe
                                    netMinTop_d, netMinTop_dn, netMinTop_dar, netMinTop_n, netMinTop_nn, netMinTop_nar = countVal(netMinTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['netMinTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['netMinTop_d_%i' %(ct)][0, k, l], netMinTop_dar))
                                    resDict5['netMinTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['netMinTop_n_%i' %(ct)][0, k, l], netMinTop_nar))
                                    #: netMax for cloud-abowe
                                    netMaxTop_d, netMaxTop_dn, netMaxTop_dar, netMaxTop_n, netMaxTop_nn, netMaxTop_nar = countVal(netMaxTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['netMaxTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['netMaxTop_d_%i' %(ct)][0, k, l], netMaxTop_dar))
                                    resDict5['netMaxTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['netMaxTop_n_%i' %(ct)][0, k, l], netMaxTop_nar))
                                    #: netMean for cloud-abowe
                                    netMeanTop_d, netMeanTop_dn, netMeanTop_dar, netMeanTop_n, netMeanTop_nn, netMeanTop_nar = countVal(netMeanTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['netMeanTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['netMeanTop_d_%i' %(ct)][0, k, l], netMeanTop_dar))
                                    resDict5['netMeanTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['netMeanTop_n_%i' %(ct)][0, k, l], netMeanTop_nar))
                                    
                                    #: shuMin for cloud-abowe
                                    shuMinTop_d, shuMinTop_dn, shuMinTop_dar, shuMinTop_n, shuMinTop_nn, shuMinTop_nar = countVal(shuMinTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shuMinTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMinTop_d_%i' %(ct)][0, k, l], shuMinTop_dar))
                                    resDict5['shuMinTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMinTop_n_%i' %(ct)][0, k, l], shuMinTop_nar))
                                    #: shuMax for cloud-abowe
                                    shuMaxTop_d, shuMaxTop_dn, shuMaxTop_dar, shuMaxTop_n, shuMaxTop_nn, shuMaxTop_nar = countVal(shuMaxTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shuMaxTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMaxTop_d_%i' %(ct)][0, k, l], shuMaxTop_dar))
                                    resDict5['shuMaxTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMaxTop_n_%i' %(ct)][0, k, l], shuMaxTop_nar))
                                    #: shuMean for cloud-abowe
                                    shuMeanTop_d, shuMeanTop_dn, shuMeanTop_dar, shuMeanTop_n, shuMeanTop_nn, shuMeanTop_nar = countVal(shuMeanTop, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shuMeanTop_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMeanTop_d_%i' %(ct)][0, k, l], shuMeanTop_dar))
                                    resDict5['shuMeanTop_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMeanTop_n_%i' %(ct)][0, k, l], shuMeanTop_nar))    
                                        
                                    #: shuMin for cloud-abowe
                                    shuMinBase_d, shuMinBase_dn, shuMinBase_dar, shuMinBase_n, shuMinBase_nn, shuMinBase_nar = countVal(shuMinBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shuMinBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMinBase_d_%i' %(ct)][0, k, l], shuMinBase_dar))
                                    resDict5['shuMinBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMinBase_n_%i' %(ct)][0, k, l], shuMinBase_nar))
                                    #: shuMax for cloud-abowe
                                    shuMaxBase_d, shuMaxBase_dn, shuMaxBase_dar, shuMaxBase_n, shuMaxBase_nn, shuMaxBase_nar = countVal(shuMaxBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shuMaxBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMaxBase_d_%i' %(ct)][0, k, l], shuMaxBase_dar))
                                    resDict5['shuMaxBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMaxBase_n_%i' %(ct)][0, k, l], shuMaxBase_nar))
                                    #: shuMean for cloud-abowe
                                    shuMeanBase_d, shuMeanBase_dn, shuMeanBase_dar, shuMeanBase_n, shuMeanBase_nn, shuMeanBase_nar = countVal(shuMeanBase, dagtb, nattb, True)  # @UnusedVariable
                                    resDict5['shuMeanBase_d_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMeanBase_d_%i' %(ct)][0, k, l], shuMeanBase_dar))
                                    resDict5['shuMeanBase_n_%i' %(ct)][0, k, l] = np.hstack((resDict5['shuMeanBase_n_%i' %(ct)][0, k, l], shuMeanBase_nar))        
                                    
#                                     iwc_m = 1e-3 *(iwclat[ctind1D,0:-1] + iwclat[ctind1D, 1:]) / 2
#                                     lwc_m = 1e-3 *(lwclat[ctind1D,0:-1] + lwclat[ctind1D, 1:]) / 2
#                                     deltaH = heighlat[ctind1D,0:-1] - heighlat[ctind1D,1:]
#                                     iwp_from_c = iwc_m * deltaH
#                                     lwp_from_c = lwc_m * deltaH
#                                     iwp_from_c_tot = np.sum(iwp_from_c, axis=1)
#                                     lwp_from_c_tot = np.sum(lwp_from_c, axis=1)
                                    
#                                     pdb.set_trace()
#                                     pdb.set_trace()
#                                     'iwpTop', 'iwpBase', 'lwpTop', 'lwpBase'
#                                 else:
                            if resdictuse[1]:
                                if ct in [9, 90]:
                                    if isinstance(ctindLMH,list):
                                        if len(ctindLMH) == 0:
                                            continue
                                    else:
                                        if ctindLMH[0].shape[0] == 0:
                                            continue
                                    rowLMH = rowlat[ctindLMH]
                                    colLMH = collat[ctindLMH]
                                    clTopLMH = clToplat[rowLMH,:][:,0]
    #                                 clBaseLMH = clBaselat[rowLMH,:][:,0]
                                    factorLMH = factorlat[rowLMH]
                #                    sza = szalat[row]
                                    shrLMH = shrlat[rowLMH, colLMH]
                                    lhrLMH = lhrlat[rowLMH, colLMH]
                                    shrNLMH = shrLMH * factorLMH
                                    dagLMH = shrLMH > 0 #: Dag definjerad via shr
                                    dagNLMH = ~np.isnan(shrNLMH) & (shrLMH > 0) #: Dag normalicerad
    #                                 dagRange = dag & clInRange
                                    nattLMH = ~dagLMH
    #                                 nattRange = natt & clInRange
                                    clTopLMH_nan = np.where(np.isnan(clTopLMH), 0, clTopLMH)
                                    Lind = (clTopLMH_nan > 720) & (clTopLMH_nan <= 2000)
                                    Mind = (clTopLMH_nan > 2000) & (clTopLMH_nan <= 6000)
                                    Hind = (clTopLMH_nan > 6000)
                                    
                                    if ct in [90]:
                                        antalDagL = (dagLMH).sum()
                                        antalDagM = (dagLMH).sum()
                                        antalDagH = (dagLMH).sum()
                                        
                                        antalNattL = (nattLMH).sum()
                                        antalNattM = (nattLMH).sum()
                                        antalNattH = (nattLMH).sum()
                                        dagL = dagLMH
                                        dagM = dagLMH
                                        dagH = dagLMH
                                        
                                        nattL = nattLMH
                                        nattM = nattLMH
                                        nattH = nattLMH
        
                                        dagNL = dagNLMH
                                        dagNM = dagNLMH
                                        dagNH = dagNLMH
                                    
                                    elif ct in [9]:
                                        antalDagL = (dagLMH & Lind).sum()
                                        antalDagM = (dagLMH & Mind).sum()
                                        antalDagH = (dagLMH & Hind).sum()
                                        
                                        antalNattL = (nattLMH & Lind).sum()
                                        antalNattM = (nattLMH & Mind).sum()
                                        antalNattH = (nattLMH & Hind).sum()
                                        dagL = dagLMH & Lind
                                        dagM = dagLMH & Mind
                                        dagH = dagLMH & Hind
                                        
                                        nattL = nattLMH & Lind
                                        nattM = nattLMH & Mind
                                        nattH = nattLMH & Hind
        
                                        dagNL = dagNLMH & Lind
                                        dagNM = dagNLMH & Mind
                                        dagNH = dagNLMH & Hind
                                    resDict4['antal_tot_d_L_%i' %(ct)][0, k, l] = resDict4['antal_tot_d_L_%i' %(ct)][0, k, l] + antalDagL
                                    resDict4['antal_tot_n_L_%i' %(ct)][0, k, l] = resDict4['antal_tot_n_L_%i' %(ct)][0, k, l] + antalNattL
                                    resDict4['antal_tot_d_M_%i' %(ct)][0, k, l] = resDict4['antal_tot_d_M_%i' %(ct)][0, k, l] + antalDagM
                                    resDict4['antal_tot_n_M_%i' %(ct)][0, k, l] = resDict4['antal_tot_n_M_%i' %(ct)][0, k, l] + antalNattM
                                    resDict4['antal_tot_d_H_%i' %(ct)][0, k, l] = resDict4['antal_tot_d_H_%i' %(ct)][0, k, l] + antalDagH
                                    resDict4['antal_tot_n_H_%i' %(ct)][0, k, l] = resDict4['antal_tot_n_H_%i' %(ct)][0, k, l] + antalNattH
                                    #: SHW
    #                                 swdL = sum(shrLMH[dagL])
    #                                 swdnL = len(shrLMH[dagL])
                                    swdL, swdnL, swnL, swnnL = countVal(shrLMH, dagL, nattL)  # @UnusedVariable
                                    resDict4['shr_d_L_%i' %(ct)][0, k, l] = resDict4['shr_d_L_%i' %(ct)][0, k, l] + swdL
                                    resDict4['antal_shr_d_L_%i' %(ct)][0, k, l] = resDict4['antal_shr_d_L_%i' %(ct)][0, k, l] + swdnL
    #                                 swdM = sum(shrLMH[dagM])
    #                                 swdnM = len(shrLMH[dagM])
                                    swdM, swdnM, swnM, swnnM = countVal(shrLMH, dagM, nattM)  # @UnusedVariable
                                    resDict4['shr_d_M_%i' %(ct)][0, k, l] = resDict4['shr_d_M_%i' %(ct)][0, k, l] + swdM
                                    resDict4['antal_shr_d_M_%i' %(ct)][0, k, l] = resDict4['antal_shr_d_M_%i' %(ct)][0, k, l] + swdnM
    #                                 swdH = sum(shrLMH[dagH])
    #                                 swdnH = len(shrLMH[dagH])
                                    swdH, swdnH, swnH, swnnH = countVal(shrLMH, dagH, nattH)  # @UnusedVariable
                                    
                                    resDict4['shr_d_H_%i' %(ct)][0, k, l] = resDict4['shr_d_H_%i' %(ct)][0, k, l] + swdH
                                    resDict4['antal_shr_d_H_%i' %(ct)][0, k, l] = resDict4['antal_shr_d_H_%i' %(ct)][0, k, l] + swdnH
    
                                    #: norm_SHW
    #                                 swNdL = sum(shrNLMH[dagNL])
    #                                 swNdnL = len(shrNLMH[dagNL])
                                    swNdL, swNdnL, swNnL, swNnnL = countVal(shrNLMH, dagNL, nattL)  # @UnusedVariable
                                    resDict4['norm_shr_d_L_%i' %(ct)][0, k, l] = resDict4['norm_shr_d_L_%i' %(ct)][0, k, l] + swNdL
                                    resDict4['antal_norm_shr_d_L_%i' %(ct)][0, k, l] = resDict4['antal_norm_shr_d_L_%i' %(ct)][0, k, l] + swNdnL
    #                                 swNdM = sum(shrNLMH[dagNM])
    #                                 swNdnM = len(shrNLMH[dagNM])
                                    swNdM, swNdnM, swNnM, swNnnM = countVal(shrNLMH, dagNM, nattM)  # @UnusedVariable
                                    resDict4['norm_shr_d_M_%i' %(ct)][0, k, l] = resDict4['norm_shr_d_M_%i' %(ct)][0, k, l] + swNdM
                                    resDict4['antal_norm_shr_d_M_%i' %(ct)][0, k, l] = resDict4['antal_norm_shr_d_M_%i' %(ct)][0, k, l] + swNdnM
    #                                 swNdH = sum(shrNLMH[dagNH])
    #                                 swNdnH = len(shrNLMH[dagNH])
                                    swNdH, swNdnH, swNnH, swNnnH = countVal(shrNLMH, dagNH, nattH)  # @UnusedVariable
                                    resDict4['norm_shr_d_H_%i' %(ct)][0, k, l] = resDict4['norm_shr_d_H_%i' %(ct)][0, k, l] + swNdH
                                    resDict4['antal_norm_shr_d_H_%i' %(ct)][0, k, l] = resDict4['antal_norm_shr_d_H_%i' %(ct)][0, k, l] + swNdnH
                                    #: LHW
    #                                 lwdL = sum(lhrLMH[dagL])
    #                                 lwdnL = len(lhrLMH[dagL])
                                    lwdL, lwdnL, lwnL, lwnnL = countVal(lhrLMH, dagL, nattL)
                                    resDict4['lhr_d_L_%i' %(ct)][0, k, l] = resDict4['lhr_d_L_%i' %(ct)][0, k, l] + lwdL
                                    resDict4['antal_lhr_d_L_%i' %(ct)][0, k, l] = resDict4['antal_lhr_d_L_%i' %(ct)][0, k, l] + lwdnL
                                
    #                                 lwnL = sum(lhrLMH[nattL])
    #                                 lwnnL = len(lhrLMH[nattL])
                                    resDict4['lhr_n_L_%i' %(ct)][0, k, l] = resDict4['lhr_n_L_%i' %(ct)][0, k, l] + lwnL
                                    resDict4['antal_lhr_n_L_%i' %(ct)][0, k, l] = resDict4['antal_lhr_n_L_%i' %(ct)][0, k, l] + lwnnL
    #                                 lwdM = sum(lhrLMH[dagM])
    #                                 lwdnM = len(lhrLMH[dagM])
                                    lwdM, lwdnM, lwnM, lwnnM = countVal(lhrLMH, dagM, nattM)
                                    resDict4['lhr_d_M_%i' %(ct)][0, k, l] = resDict4['lhr_d_M_%i' %(ct)][0, k, l] + lwdM
                                    resDict4['antal_lhr_d_M_%i' %(ct)][0, k, l] = resDict4['antal_lhr_d_M_%i' %(ct)][0, k, l] + lwdnM
                                
    #                                 lwnM = sum(lhrLMH[nattM])
    #                                 lwnnM = len(lhrLMH[nattM])
                                    resDict4['lhr_n_M_%i' %(ct)][0, k, l] = resDict4['lhr_n_M_%i' %(ct)][0, k, l] + lwnM
                                    resDict4['antal_lhr_n_M_%i' %(ct)][0, k, l] = resDict4['antal_lhr_n_M_%i' %(ct)][0, k, l] + lwnnM
                                    
    #                                 lwdH = sum(lhrLMH[dagH])
    #                                 lwdnH = len(lhrLMH[dagH])
                                    lwdH, lwdnH, lwnH, lwnnH = countVal(lhrLMH, dagH, nattH)
                                    resDict4['lhr_d_H_%i' %(ct)][0, k, l] = resDict4['lhr_d_H_%i' %(ct)][0, k, l] + lwdH
                                    resDict4['antal_lhr_d_H_%i' %(ct)][0, k, l] = resDict4['antal_lhr_d_H_%i' %(ct)][0, k, l] + lwdnH
                                
    #                                 lwnH = sum(lhrLMH[nattH])
    #                                 lwnnH = len(lhrLMH[nattH])
                                    resDict4['lhr_n_H_%i' %(ct)][0, k, l] = resDict4['lhr_n_H_%i' %(ct)][0, k, l] + lwnH
                                    resDict4['antal_lhr_n_H_%i' %(ct)][0, k, l] = resDict4['antal_lhr_n_H_%i' %(ct)][0, k, l] + lwnnH
                    # 2-D #
                    if isinstance(ctind,list):
                        if len(ctind) == 0:
                            continue
                    else:
                        if ctind[0].shape[0] == 0:
                            continue
                    row = rowlat[ctind]
                    col = collat[ctind]
                    
#: optical thickness
#                     if ct in [11, 12]:
#                         lop = loplat[row, col]
#                         top = toplat[row]
#                         optres = 0
#                         if ct == 11:
#                             opind = np.where(lop <= optres)[0]
#                         else:
#                             opind = np.where(lop > optres)[0]
#                         row = row[opind]
#                         col = col[opind]
#                         if row.shape[0] == 0:
#                             continue
#                         pdb.set_trace()
                    clTop = clToplat[row,:]
                    clBase = clBaselat[row,:]
                    clInRange = np.zeros(row.shape).astype('bool')
                    clBase = np.where(np.isnan(clBase), j + 1, clBase)
                    clTop = np.where(np.isnan(clTop), j - 1, clTop)
                    temp = np.where((j >= clBase) & (j <= clTop))
                    clInRange[temp[0]] = True
                           
                    factor = factorlat[row]
#                    sza = szalat[row]
                    #: radar reflectivity
                    if reflat.shape[0] == 0:
                        ref = reflat
                    else:
                        ref = reflat[row, col]
                    shu = shulat[row, col]
                    tem = temlat[row, col]
                    shr = shrlat[row, col]
                    lhr = lhrlat[row, col]
                    shrN = shr * factor
                    sfd = sfdlat[row, col]
                    lfd = lfdlat[row, col]
                    sfu = sfulat[row, col]
                    lfu = lfulat[row, col]
                    sfdnc = sfdnclat[row, col]
                    lfdnc = lfdnclat[row, col]
                    sfunc = sfunclat[row, col]
                    lfunc = lfunclat[row, col]
                    iwc = iwclat[row, col]
                    lwc = lwclat[row, col]
                    
                    shr_nan = np.where(np.isnan(shr), -1, shr)
                    dag = shr_nan > 0 #: Dag definjerad via shr
                    dagN = ~np.isnan(shrN) & (dag) #: Dag normalicerad
                    dagRange = dag & clInRange
#                    dagS = (sza < 80) & (shr > 0) #: Dag special
#                    dagNS = dagS & dagN #: Dag special och normalicerad
                    #: Samma som ovan men endast molnigapixlar
#                    dagRangeS = dagS & clInRange
#                    dagRangeN = dagN & clInRange
#                    dagRangeNS = dagNS & clInRange
                    # TODO: Realice that shr = nan becomes night pixels
                    natt = ~dag
                    nattRange = natt & clInRange
    #                antalTot = len(row)
                    
                    if ct in [0, 90, 99]:
                        antalDag = dag.sum()
                        antalNatt = natt.sum()
                    elif (ct in [1,2,3,4,5,6,7,8,9,10, 27, 11, 12, 13, 14, 15, 18, 127, 187]) or ((ct > 200) and ct < 900):
                        antalDag = dagRange.sum()
                        antalNatt = nattRange.sum()
                    else:
                        print('fel molntyp')
                    resDict2['d_%i' %(ct)][ih, k, l] = resDict2['d_%i' %(ct)][ih, k, l] + antalDag
                    resDict2['n_%i' %(ct)][ih, k, l] = resDict2['n_%i' %(ct)][ih, k, l] + antalNatt

                    #: SHW
                    swd, swdn, swn, swnn = countVal(shr, dag, natt)  # @UnusedVariable
#                     swd = sum(shr[dag])
#                     swdn = len(shr[dag])
                    resDict['shr_d_%i' %(ct)][ih, k, l] = resDict['shr_d_%i' %(ct)][ih, k, l] + swd
                    resDict['antal_shr_d_%i' %(ct)][ih, k, l] = resDict['antal_shr_d_%i' %(ct)][ih, k, l] + swdn
        
                    #: norm_SHW
                    swNd, swNdn, swNdar, swNn, swNnn, swNnar = countVal(shrN, dagN, natt, True)  # @UnusedVariable
#                     swNd = sum(shrN[dagN])
#                     swNdn = len(shrN[dagN])
                    resDict['norm_shr_d_%i' %(ct)][ih, k, l] = resDict['norm_shr_d_%i' %(ct)][ih, k, l] + swNd
                    resDict['antal_norm_shr_d_%i' %(ct)][ih, k, l] = resDict['antal_norm_shr_d_%i' %(ct)][ih, k, l] + swNdn
                    if False:#ct not in [90, 876, 877, 878]:
                        resDict5['norm_shr_d_%i' %(ct)][ih, k, l] = np.hstack((resDict5['norm_shr_d_%i' %(ct)][ih, k, l], swNdar))
                    #: LHW
                    lwd, lwdn, lwdar, lwn, lwnn, lwnar = countVal(lhr, dag, natt, True)
#                     lwd = sum(lhr[dag])
#                     lwdn = len(lhr[dag])
                    resDict['lhr_d_%i' %(ct)][ih, k, l] = resDict['lhr_d_%i' %(ct)][ih, k, l] + lwd
                    resDict['antal_lhr_d_%i' %(ct)][ih, k, l] = resDict['antal_lhr_d_%i' %(ct)][ih, k, l] + lwdn
                    if False:#ct not in [90, 876, 877, 878]:
                        resDict5['lhr_d_%i' %(ct)][ih, k, l] = np.hstack((resDict5['lhr_d_%i' %(ct)][ih, k, l], lwdar))
#                     lwn = sum(lhr[natt])
#                     lwnn = len(lhr[natt])
                    resDict['lhr_n_%i' %(ct)][ih, k, l] = resDict['lhr_n_%i' %(ct)][ih, k, l] + lwn
                    resDict['antal_lhr_n_%i' %(ct)][ih, k, l] = resDict['antal_lhr_n_%i' %(ct)][ih, k, l] + lwnn
                    if False:#ct not in [90, 876, 877, 878]:
                        resDict5['lhr_n_%i' %(ct)][ih, k, l] = np.hstack((resDict5['lhr_n_%i' %(ct)][ih, k, l], lwnar))
                    #: REF
                    refd, refdn, refn, refnn = countVal(ref, dag, natt)
                    resDict['ref_d_%i' %(ct)][ih, k, l] = resDict['ref_d_%i' %(ct)][ih, k, l] + refd
                    resDict['antal_ref_d_%i' %(ct)][ih, k, l] = resDict['antal_ref_d_%i' %(ct)][ih, k, l] + refdn

                    resDict['ref_n_%i' %(ct)][ih, k, l] = resDict['ref_n_%i' %(ct)][ih, k, l] + refn
                    resDict['antal_ref_n_%i' %(ct)][ih, k, l] = resDict['antal_ref_n_%i' %(ct)][ih, k, l] + refnn
                    
                    #: SHU
                    shud, shudn, shudar, shun, shunn, shunar = countVal(shu, dag, natt, True)
#                     shud = sum(shu[dag])
#                     shudn = len(shu[dag])
                    resDict['shu_d_%i' %(ct)][ih, k, l] = resDict['shu_d_%i' %(ct)][ih, k, l] + shud
                    resDict['antal_shu_d_%i' %(ct)][ih, k, l] = resDict['antal_shu_d_%i' %(ct)][ih, k, l] + shudn
                    if False:#ct not in [90, 876, 877, 878]:
                        resDict5['shu_d_%i' %(ct)][ih, k, l] = np.hstack((resDict5['shu_d_%i' %(ct)][ih, k, l], shudar))
#                     shun = sum(shu[natt])
#                     shunn = len(shu[natt])
                    resDict['shu_n_%i' %(ct)][ih, k, l] = resDict['shu_n_%i' %(ct)][ih, k, l] + shun
                    resDict['antal_shu_n_%i' %(ct)][ih, k, l] = resDict['antal_shu_n_%i' %(ct)][ih, k, l] + shunn
                    if False:#ct not in [90, 876, 877, 878]:
                        resDict5['shu_n_%i' %(ct)][ih, k, l] = np.hstack((resDict5['shu_n_%i' %(ct)][ih, k, l], shunar))
                    
                    #: Temperature
                    temd, temdn, temn, temnn = countVal(tem, dag, natt)
                    resDict['tem_d_%i' %(ct)][ih, k, l] = resDict['tem_d_%i' %(ct)][ih, k, l] + temd
                    resDict['antal_tem_d_%i' %(ct)][ih, k, l] = resDict['antal_tem_d_%i' %(ct)][ih, k, l] + temdn
                    resDict['tem_n_%i' %(ct)][ih, k, l] = resDict['tem_n_%i' %(ct)][ih, k, l] + temn
                    resDict['antal_tem_n_%i' %(ct)][ih, k, l] = resDict['antal_tem_n_%i' %(ct)][ih, k, l] + temnn
                    
#                     #: SFD
#                     sfdd, sfddn, sfdn, sfdnn = countVal(sfd, dag, natt)  # @UnusedVariable
#                     resDict['sfd_d_%i' %(ct)][ih, k, l] = resDict['sfd_d_%i' %(ct)][ih, k, l] + sfdd
#                     resDict['antal_sfd_d_%i' %(ct)][ih, k, l] = resDict['antal_sfd_d_%i' %(ct)][ih, k, l] + sfddn
#                     
#                     #: LFD sparar endast dag trots att natt finns. Anvander bara dagvarden for tillfallet
#                     lfdd, lfddn, lfdn, lfdnn = countVal(lfd, dag, natt)  # @UnusedVariable
#                     resDict['lfd_d_%i' %(ct)][ih, k, l] = resDict['lfd_d_%i' %(ct)][ih, k, l] + lfdd
#                     resDict['antal_lfd_d_%i' %(ct)][ih, k, l] = resDict['antal_lfd_d_%i' %(ct)][ih, k, l] + lfddn
#                     
#                     #: SFD_NC
#                     sfdncd, sfdncdn, sfdncn, sfdncnn = countVal(sfdnc, dag, natt)  # @UnusedVariable
#                     resDict['sfdnc_d_%i' %(ct)][ih, k, l] = resDict['sfdnc_d_%i' %(ct)][ih, k, l] + sfdncd
#                     resDict['antal_sfdnc_d_%i' %(ct)][ih, k, l] = resDict['antal_sfdnc_d_%i' %(ct)][ih, k, l] + sfdncdn
#                     
#                     #: LFD_NC sparar endast dag trots att natt finns. Anvander bara dagvarden for tillfallet
#                     lfdncd, lfdncdn, lfdncn, lfdncnn = countVal(lfdnc, dag, natt)  # @UnusedVariable
#                     resDict['lfdnc_d_%i' %(ct)][ih, k, l] = resDict['lfdnc_d_%i' %(ct)][ih, k, l] + lfdncd
#                     resDict['antal_lfdnc_d_%i' %(ct)][ih, k, l] = resDict['antal_lfdnc_d_%i' %(ct)][ih, k, l] + lfdncdn
#                     
#                     #: SFU
#                     sfud, sfudn, sfun, sfunn = countVal(sfu, dag, natt)  # @UnusedVariable
#                     resDict['sfu_d_%i' %(ct)][ih, k, l] = resDict['sfu_d_%i' %(ct)][ih, k, l] + sfud
#                     resDict['antal_sfu_d_%i' %(ct)][ih, k, l] = resDict['antal_sfu_d_%i' %(ct)][ih, k, l] + sfudn
#                     
#                     #: LFU sparar endast dag trots att natt finns. Anvander bara dagvarden for tillfallet
#                     lfud, lfudn, lfun, lfunn = countVal(lfu, dag, natt)  # @UnusedVariable
#                     resDict['lfu_d_%i' %(ct)][ih, k, l] = resDict['lfu_d_%i' %(ct)][ih, k, l] + lfud
#                     resDict['antal_lfu_d_%i' %(ct)][ih, k, l] = resDict['antal_lfu_d_%i' %(ct)][ih, k, l] + lfudn
#                     
#                     #: SFU_NC
#                     sfuncd, sfuncdn, sfuncn, sfuncnn = countVal(sfunc, dag, natt)  # @UnusedVariable
#                     resDict['sfunc_d_%i' %(ct)][ih, k, l] = resDict['sfunc_d_%i' %(ct)][ih, k, l] + sfuncd
#                     resDict['antal_sfunc_d_%i' %(ct)][ih, k, l] = resDict['antal_sfunc_d_%i' %(ct)][ih, k, l] + sfuncdn
#                     
#                     #: LFU_NC sparar endast dag trots att natt finns. Anvander bara dagvarden for tillfallet
#                     lfuncd, lfuncdn, lfuncn, lfuncnn = countVal(lfunc, dag, natt)  # @UnusedVariable
#                     resDict['lfunc_d_%i' %(ct)][ih, k, l] = resDict['lfunc_d_%i' %(ct)][ih, k, l] + lfuncd
#                     resDict['antal_lfunc_d_%i' %(ct)][ih, k, l] = resDict['antal_lfunc_d_%i' %(ct)][ih, k, l] + lfuncdn
#                     
                    #: IWC sparar endast dag trots att natt finns. Anvander bara dagvarden for tillfallet
                    iwcd, iwcdn, iwcn, iwcnn = countVal(iwc, dag, natt)  # @UnusedVariable
                    resDict['iwc_d_%i' %(ct)][ih, k, l] = resDict['iwc_d_%i' %(ct)][ih, k, l] + iwcd
                    resDict['antal_iwc_d_%i' %(ct)][ih, k, l] = resDict['antal_iwc_d_%i' %(ct)][ih, k, l] + iwcdn

                    #: LWC sparar endast dag trots att natt finns. Anvander bara dagvarden for tillfallet
                    lwcd, lwcdn, lwcn, lwcnn = countVal(lwc, dag, natt)  # @UnusedVariable
                    resDict['lwc_d_%i' %(ct)][ih, k, l] = resDict['lwc_d_%i' %(ct)][ih, k, l] + lwcd
                    resDict['antal_lwc_d_%i' %(ct)][ih, k, l] = resDict['antal_lwc_d_%i' %(ct)][ih, k, l] + lwcdn
                    
                    

    resDict.update({'fname': flxfile})
    np.save(tempname, [resDict, resDict2, resDict3, resDict4, resDict5])
    return resDict, resDict2, resDict3, resDict4, resDict5

def testar(dirflx, dirclc, differentCloudTypes, db):
    filename = '/nobackup/smhid10/sm_erjoh/PhD-1/NrCLouds/nrOfClouds.h5'
    h5file = h5py.File(filename,'w')
#     differentCloudTypes = [0,1,2,3,4,5,6,7,8,9,10,27,90]
    missingCLC  = 0
    for year in [2007, 2008, 2009, 2010]:
        h5file.create_group("%i" %year)
        for mon in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            print('%s %.02i' %(year, mon))
            tic = time.time()
            h5file.create_group("%i/%.02i" %(year, mon))
            stDay = time.strptime("%i %i 1" %(year, mon) ,"%Y %m %d").tm_yday
            if mon == 12:
                enDay = time.strptime("%i %i 31" %(year, 12) ,"%Y %m %d").tm_yday + 1
            else:
                enDay = time.strptime("%i %i 1" %(year, (mon + 1)) ,"%Y %m %d").tm_yday
        
            files = []
            for day in range(stDay,enDay):
                files.extend(glob.glob('%s/%i%0.3i*.h5' %(dirflx, year, day)))
            
            resDict = {}
            for z in ['z1', 'z2', 'z3', 'z123', 'zTropic']:
                for typ in differentCloudTypes:
                    resDict.update({'%s_%i' %(z, typ): 0})
            k = 0
            nroffiles = len(files)
            
            for filen in files:
                k = k + 1
                if k < options.db:
                    continue
                print("%i av %i" %(k, nroffiles))
                lidBasename = os.path.basename(filen)
                ClcName = '%s/%s*2B-CLDCLASS-LIDAR*.h5' %(dirclc, lidBasename.split('2B-FLXHR-LIDAR')[0])
                clcfile = glob.glob(ClcName)
                if len(clcfile) == 1:
                    clTfile = clcfile[0]
                else:
                    missingCLC = missingCLC + 1
                    continue
                liddata = readFLXHRLidar(filen)
                clcdata = readCloudType(clTfile)
                if -1 in [liddata, clcdata]:
                    print('fan')
                    continue
                lat = liddata.latitude
                lon = liddata.longitude
                for zon in ['1', '2', '3', '123', 'Tropic']:
                    if zon == 'Tropic':
                        latind = np.where((lat >= -30) & (lat <= 30))[0]
                    else:
                        if zon == '1':
                            latmin = -19.5; latmax = 39.5; longmin = 65.0; longmax = 75.0;
                        elif zon == '2':
                            latmin = -19.5; latmax = 39.5; longmin = 75.0; longmax = 85.0;
                        elif zon == '3':
                            latmin = -19.5; latmax = 39.5; longmin = 85.0; longmax = 95.0;
                        elif zon == '123':
                            latmin = -19.5; latmax = 39.5; longmin = 65.0; longmax = 95.0;
                        latind = np.where((lat>latmin) & (lat<latmax) & (lon>longmin) & (lon<longmax))[0]
                        
                    nrCLlat = clcdata['CL'][latind]
                    clTYlat = clcdata['CLType'][:,0][latind]

                    for ct in differentCloudTypes:
                        if ct == 0:
                            resDict["z%s_%i" %(zon, ct)] = resDict["z%s_%i" %(zon, ct)] + (nrCLlat == 0).sum()
                        if ct > 0 and ct <= 8:
                            resDict["z%s_%i" %(zon, ct)] = resDict["z%s_%i" %(zon, ct)] + ((nrCLlat == 1) & (clTYlat == ct)).sum()
                        if ct == 9:
                            resDict["z%s_%i" %(zon, ct)] = resDict["z%s_%i" %(zon, ct)] + (nrCLlat > 0).sum()
                        if ct == 10:
                            resDict["z%s_%i" %(zon, ct)] = resDict["z%s_%i" %(zon, ct)] + (nrCLlat > 1).sum()
                        if ct == 27:
                            resDict["z%s_%i" %(zon, ct)] = resDict["z%s_%i" %(zon, ct)] + ((nrCLlat == 1) & ((clTYlat == 2) | (clTYlat == 7))).sum()
                        if ct == 90:
                            resDict["z%s_%i" %(zon, ct)] = resDict["z%s_%i" %(zon, ct)] + len(latind)
            
            print(time.time() - tic)        

            for zc, value in resDict.items():
                zone = zc.split('_')[0][1:]
                ctype = int(zc.split('_')[1])
                groupname = "%i/%.02i/Zone %s" %(year, mon, zone)
                if not "Zone " + zone in h5file["%i/%.02i" %(year, mon)].keys():
                    h5file.create_group(groupname)
                if ctype == 0:
                    h5file.create_dataset(("%s/CT Clear" %groupname), data = value)
                elif ctype > 0 and ctype <= 8:
                    h5file.create_dataset(("%s/CT %i" %(groupname, ctype)), data = value)
                elif ctype == 9:
                    h5file.create_dataset(("%s/CT Cloudy"%groupname), data = value)
                elif ctype == 10:
                    h5file.create_dataset(("%s/CT Multy Layer"%groupname), data = value)
                elif ctype == 27:
                    h5file.create_dataset(("%s/CT %i" %(groupname, 27)), data = value)
                elif ctype == 90:
                    h5file.create_dataset(("%s/CT Total number" %groupname), data = value)
                else:
                    print('ups')
                    print(ctype)
                    continue
    h5file.close()


def SingleFiletoH5(maindir, area, month, latrange, dict1, dict2, dict3, dict4):
    sea = area.split('_')[0]
    fname = dict1['fname']
    fnameBasename = os.path.basename(fname)
    fnameDate = fnameBasename.split('_')[0]
    if isinstance(month, str):
        filenamePath = '%s/Data/singleFile/%s/%s' %(maindir, sea, month)
    else:
        filenamePath = '%s/Data/singleFile/%s/%02i' %(maindir, sea, month)
    if not os.path.exists(filenamePath):
        os.makedirs(filenamePath) 
    filename = filenamePath + '/nc_%s_%s.h5' %(area, fnameDate)
    if not os.path.isfile(filename):
        h5file = h5py.File(filename, 'w')
        h5file.create_dataset('Start Latitude', data = latrange)
        grpcf = h5file.create_group('cf')
        grpcfD = grpcf.create_group('d')
        grpcfN = grpcf.create_group('n')
        
        for strname, val in dict2.items():
            if strname[0] == 'n':
                grpcfN.create_dataset(strname, data = val)
            else:
                grpcfD.create_dataset(strname, data = val)
        grprh = h5file.create_group('rh')
        grprhD = grprh.create_group('d')
        grprhN = grprh.create_group('n')
        grprhDN = grprhD.create_group('quantity')
        grprhNN = grprhN.create_group('quantity')
        grprhDH = grprhD.create_group('heating')
        grprhNH = grprhN.create_group('heating')
        for strname, val in resDict.items():
            if '_n_' in strname:
                if 'antal' in strname:
                    grprhNN.create_dataset(strname, data = val)
                else:
                    grprhNH.create_dataset(strname, data = val)
            else:
                if 'antal' in strname:
                    grprhDN.create_dataset(strname, data = val)
                else:
                    grprhDH.create_dataset(strname, data = val)
        h5file.close()


def getClcGeo(flxname, mainclc, maingeo, maintau, maincwc, mainaux):
    #: Find CLC files
    lidBasename = os.path.basename(flxname)
    ClcName = '%s/%s*2B-CLDCLASS-LIDAR*.h5' %(mainclc, lidBasename.split('2B-FLXHR-LIDAR')[0])
    clcfile = glob.glob(ClcName)
    #: Find Geo files
    y = int(lidBasename[0:4])
    J = int(lidBasename[4:7])
    dates = datetime.datetime.strptime('%i %03i' %(y, J), '%Y %j')
    mon = dates.month
    
    if len(clcfile) == 0:
        #: Find CLC files
        ClcName = '%s/%d/%02d/%s*2B-CLDCLASS-LIDAR*.h5' %(mainclc, y, mon, lidBasename.split('2B-FLXHR-LIDAR')[0])
        clcfile = glob.glob(ClcName)
    
    GeoName = '%s/%i/%02i/%s*.h5' %(maingeo, y, mon, lidBasename.split('2B-FLXHR-LIDAR')[0])
    geofile = glob.glob(GeoName)
    
    TauName = '%s/%i/%02i/%s*.h5' %(maintau, y, mon, lidBasename.split('2B-FLXHR-LIDAR')[0])
    taufile = glob.glob(TauName)
    
    CwcName = '%s/%i/%02i/%s*.h5' %(maincwc, y, mon, lidBasename.split('2B-FLXHR-LIDAR')[0])
    cwcfile = glob.glob(CwcName)
    
    AuxName = '%s/%i/%02i/%s*.h5' %(mainaux, y, mon, lidBasename.split('2B-FLXHR-LIDAR')[0])
    auxfile = glob.glob(AuxName)

    if len(clcfile) > 1:
        print('what clc')
        sys.exit()
    if len(geofile) == 0:
        Geofile = ''
    else:
        Geofile = geofile[0]
    if len(taufile) == 0:
        Taufile = ''
    else:
        Taufile = taufile[0]
    if len(cwcfile) == 0:
        Cwcfile = ''
    else:
        Cwcfile = cwcfile[0]
    if len(auxfile) == 0:
        Auxfile = ''
    else:
        Auxfile = auxfile[0]

    return clcfile, Geofile, Taufile, Cwcfile, Auxfile

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-c', '--ct', type='int', default = -1, help = \
                      'What cloud type to use, Default = 0')
    parser.add_option('-d', '--db', type='int', default = 0, help = \
                      'What file do you want to start from, Default = 0')
    parser.add_option('-f', '--File', type='int', default=0, help = \
                      'Use a file')
    parser.add_option('-l', '--loadTemp', action='store_false', default = True, help = \
                      'If you dont wont to use tempfiles')
    parser.add_option('-m', '--Month', type='int', default=1, help = \
                      'Defines the month')
    parser.add_option('-n', '--num', action='store_true', default=False, help = \
                      'Ett test som jag inte kommer ihag, Default = 0')
    
    parser.add_option('-p', '--par', action='store_true', default=False, help = \
                      'Use parallel python, Deafaul = False')
    parser.add_option('-r', '--resD', action='store_true', default=False, help = \
                      'if true only resDict will be fetched, Default = False')
    parser.add_option('-u', '--fn', action='store_true', default=False, help = \
                      'Andrar filenamnet, Deafaul = False')
    parser.add_option('-v', '--Week', type='int', default=0, help = \
                      'Calc a week')
    parser.add_option('-y', '--Year', type='int', default=2007, help = \
                      'Defines the year')
    options, argument = parser.parse_args()
    mainDir = '/nobackup/smhid14/sm_erjoh/PhD-2'
    mainTemp = '/nobackup/smhid14/sm_erjoh/PhD-2/TempFiles/crhabove'
    mainLid = '/nobackup/smhid17/proj/foua/data/satellit/cloudsat/2B-FLXHR-LIDAR_V4'
    mainClc = '/nobackup/smhid17/proj/foua/data/satellit/cloudsat/2B-CLDCLASS-LIDAR_V4'
    mainGeo = '/nobackup/smhid17/proj/foua/data/satellit/cloudsat/2B-GEOPROF_V4'
    mainTau = '/nobackup/smhid17/proj/foua/data/satellit/cloudsat/2B-TAU_V4'
    mainCwc = '/nobackup/smhid17/proj/foua/data/satellit/cloudsat/2B-CWC-RVOD_V4'
    mainAux = '/nobackup/smhid17/proj/foua/data/satellit/cloudsat/ECMWF-AUX_V4'

    if options.ct == -1:
        differentCloudTypes = [0,1,2,3,4,5,6,7,8,9,10,27,90,99, 11, 12, 13, 14, 15]
        #: Forsta siffran ar molntyp
        #: Andra tva ar od threshold for ct 1
        for ctgrund in range(2,9):
            for cttop in range(10,16):
                newcloudtype = ctgrund * 100 + cttop
                differentCloudTypes.append(newcloudtype)
            if ctgrund == 2:
                for cttop in range(70,76):
                    newcloudtype = ctgrund * 100 + cttop
                    differentCloudTypes.append(newcloudtype)
    else:
        differentCloudTypes = [options.ct]
    if options.num:
        testar(mainLid, mainClc, differentCloudTypes, options.db)
        sys.exit()
    files = []

#     latrange = range(-30,26,5) + range(60,86,5)
    OTlimit = {1:1.3, 2:3.6, 3:9.4, 4:23}
    latrange = range(-30,86,5)
    useGlob = False
    if options.Week != 0:
        #: useGlob makes the tempname not have to be exactly as loadname. This means month/day/weeek can chear
        useGlob = True
        if options.ct == -1:
            fCt = ''
        else:
            fCt = '_%i' %options.ct
        #: Options for daily results
        #: Month
        if options.Week > 2000:
            stmon = options.Week - 2000
            singlefileArea = 'M'
            opptionsText = 'Mon %02i %i%s' %(stmon, options.Year, fCt)
            filename = '%s/Data/hitRate/nc_m_%03i%s.h5' %(mainDir, stmon, fCt)
        #: Day
        elif (options.Week > 1000) and (options.Week < 2000):
            lw = 1 #: lw = length week, in this case 1 day
            stday = options.Week - 1000
            filename = '%s/Data/hitRate/nc_d_%03i%s.h5' %(mainDir, stday, fCt)
            singlefileArea = 'D'
            opptionsText = 'Day %03i %i%s' %(stday, options.Year, fCt)
        #: Week
        else:
            stday = options.Week
            filename = '%s/Data/hitRate/nc_w_%03i%s.h5' %(mainDir, stday, fCt)
            singlefileArea = 'W'
            opptionsText = 'Week %03i %i%s' %(stday, options.Year, fCt)
            if stday >= 357:
                lw = 9 #: Last week, therfore 9 to capture all days in the end
            else:
                lw = 7 #: lw = lenght week, one week is 7 days
        
        if options.Year == 0: #: This will get all years
            years = [2007, 2008, 2009, 2010, 2011]
        else:
            years = [options.Year] #: Only one week
            filename = filename.replace('.h5', '_y%s.h5' %str(options.Year)[2:]) #: Add that year to file name
        #: Find the files for all days
        for year in years:
            if options.Week > 2000:
                stday = time.strptime("%i %i 1" %(year, stmon) ,"%Y %m %d").tm_yday
                if stmon == 12:
                    enDoy = time.strptime("%i %i 31" %(year, 12) ,"%Y %m %d").tm_yday + 1
                else:
                    enDoy = time.strptime("%i %i 1" %(year, (stmon + 1)) ,"%Y %m %d").tm_yday
                lw = enDoy - stday
                umon = stmon
            else:
                umon = '%03i' %stday
            
            for d in range(lw):
                doy = d +  stday
                mon = time.strptime("%i %i" %(year, doy) ,"%Y %j").tm_mon
                files.extend(glob.glob('%s/%d/%0.2d/%s%0.3d*.h5' %(mainLid, year, mon, year, int(doy))))
        print(opptionsText)
        latrange = range(-30, 31)
    elif options.File == 0:
        if options.Year == 0:
            year = 2007
        else:
            year = options.Year
        mon = options.Month
        umon = mon
        opptionsText = "year = %i, month = %i" %(year, mon)
        filename = '%s/Data/hitRate/nc_%i_%02i.h5' %(mainDir, year, mon)
        print(opptionsText)
        stDoy = time.strptime("%i %i 1" %(year, mon) ,"%Y %m %d").tm_yday
        if mon == 12:
            enDoy = time.strptime("%i %i 31" %(year, 12) ,"%Y %m %d").tm_yday + 1
        else:
            enDoy = time.strptime("%i %i 1" %(year, (mon + 1)) ,"%Y %m %d").tm_yday
        for doy in range(stDoy,enDoy):
            files.extend(glob.glob('%s/%i%0.3i*.h5' %(mainLid, year, doy)))
        singlefileArea = '%i' %year
    else:
        umon = options.Month
        if options.File in [1,2,3,4]:
            if options.File==1:
                txtFile = 'en'
                opptionsText = 'Enhanced Negative'
            elif options.File==2:
                txtFile = 'ep'
                opptionsText = 'Enhanced Positive'
            elif options.File==3:
                txtFile = 'cn'
                opptionsText = 'Climatological Negative'
            elif options.File==4:
                txtFile = 'cp'
                opptionsText = 'Climatological Positive'
            filename = '%s/Data/hitRate/nc_%s_m%02i.h5' %(mainDir, txtFile, umon)
            singlefileArea = txtFile
        else:
            wvName = {5: 'AT_10', 6: 'AT_50', 7: 'AT_90', \
                      8: 'BK_10', 9: 'BK_50', 10: 'BK_90', \
                      11: 'LB_10', 12: 'LB_50', 13: 'LB_90', \
                      14: 'PC_10', 15: 'PC_50', 16: 'PC_90'}

            opptionsText = wvName[options.File].replace('_', ' ')
            filename = '%s/Data/hitRate/nc_%s_m%02i.h5' %(mainDir, wvName[options.File], umon)
            txtFile = './arctic_wv/%s_2days' %wvName[options.File]
            singlefileArea = wvName[options.File]
        if options.fn:
            filename = filename.split('.')[0] + '_lo.h5'
        opptionsText = opptionsText + ' %i' %umon
        print(opptionsText)
        tf = open(txtFile + '.txt')
        tL = tf.readlines()
        tf.close()
        for date in tL:
            if '\t' in date:
                year, mon,day,num = date.split('\t')
            else:
                year, mon, day = date.split('\r')[0].split(' ')
            if int(year) not in [2007, 2008, 2009, 2010, 2011]:
                continue
            if int(mon) != umon:
                continue
            doy = time.strptime("%s %s %s" %(year, mon, day) ,"%Y %m %d").tm_yday
            files.extend(glob.glob('%s/%s%0.3i*.h5' %(mainLid, year, int(doy))))
    sttime = time.time()
    k = 0
    nroffiles = len(files)
    print(nroffiles)
    nrf = 0
    missingCLC = 0
    if options.par:
        if (latrange[1] - latrange[0]) >= 5:
            antalKarnaor = 5
        else:
            antalKarnaor = 10
        filerna = []
        temp = []
        fc = 0
        for filen in files:
            fc = fc + 1
            if fc < options.db:
                missingCLC = missingCLC + 1
                k = k + 1
                continue
            temp.append(filen)
            if fc % antalKarnaor  == 0 or fc == nroffiles:
                filerna.append(temp)
                temp = []
        files = filerna
    tempFileEnd = 'y%i_m%i_w%i_u%i_r%i' %(options.Year, options.Month, options.Week, options.fn, options.resD)
    removePrecip = 1
    if removePrecip == 1:
        filename = filename.replace('.h5', '_utanP.h5')
        tempFileEnd + '_utanP'
    for filen in files:
#        sttime = time.time()
        if not options.par:
            k = k + 1
            if k < options.db:
                missingCLC = missingCLC + 1
                continue
            print("%i av %i" %(k, nroffiles))
            clcfile, Geofile, Taufile, Cwcfile, Auxfile = getClcGeo(filen, mainClc, mainGeo, mainTau, mainCwc, mainAux)
            if len(clcfile) == 1:
                Clcfile = clcfile[0]
            else:
                missingCLC = missingCLC + 1
                continue
            tempdir = '%s/%s' %(mainTemp, fCt)
            if not os.path.exists(tempdir):
                os.makedirs(tempdir)
            tempname = '%s/%s_split%s_%s' %(tempdir, os.path.basename(filen).replace('.h5', ''), fCt, tempFileEnd)
            ticar = time.time()
            stDict, stDict2, stDict3, stDict4, stDict5 = findValuedData(filen, Clcfile, Geofile, Taufile, Cwcfile, Auxfile, differentCloudTypes, latrange, OTlimit, tempname, options.loadTemp, fCt, options.resD)
            tocar = time.time()
            print(tocar - ticar)
            if -1 in [stDict, stDict2, stDict3, stDict4]:
                print('hep')
                continue
            if nrf == 0:
                nrf = 1
                resDict = copy.deepcopy(stDict)
                resDict.pop('fname')
#                 resDictT = resDict.copy()
                resDict2 = copy.deepcopy(stDict2)
                resDict3 = copy.deepcopy(stDict3)
                resDict4 = copy.deepcopy(stDict4)
                if (len(differentCloudTypes) == 1) and (differentCloudTypes[0] == 90):
                    continue
                else:
                    resDict5 = copy.deepcopy(stDict5)
            else:
#                 {k: a[k]+ b[k] for k in a}
#                 tic = time.time()
#                 resDictT = {k:resDictT[k] + stDict[k] for k in resDictT}
#                 resDictT = reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.iteritems()), [resDictT,stDict])
#                 print(time.time() - tic)
                for arname, value in stDict.items():
                    if arname == 'fname':
                        continue
                    resDict[arname] = resDict[arname] + value
                for arname, value in stDict2.items():
                    resDict2[arname] = resDict2[arname] + value
                for arname, value in stDict3.items():
                    resDict3[arname] = resDict3[arname] + value
                for arname, value in stDict4.items():
                    resDict4[arname] = resDict4[arname] + value
                if (len(differentCloudTypes) == 1) and (differentCloudTypes[0] == 90):
                    continue
                else:
                    for arname, value in stDict5.items():
#                         tic1=time.time()
                        try:
                            temp_rD5 = np.array([np.concatenate((a,b)) for a,b in zip(resDict5[arname].flat, value.flat)]).reshape(value.shape[0],value.shape[1],value.shape[2])
                        except:
#                             print('da drum drum %s' %arname)
                            temp_rD5 = copy.deepcopy(resDict5[arname])
                            for ii in range(value.shape[0]):
                                for jj in range(value.shape[1]):
                                    for kk in range(value.shape[2]):
                                        temp_rD5[ii, jj, kk] = np.hstack((temp_rD5[ii, jj, kk], value[ii, jj, kk]))
                        resDict5[arname] = temp_rD5
#                     pdb.set_trace()
#                         toc1 = time.time()
#     #                     tic2=time.time()
#     #                     f=np.frompyfunc(lambda a,b: np.concatenate((resDict5[arname],value)),2,1)
#     #                     test2 = f(resDict5[arname],value)
#     #                     toc2 = time.time()
#                         tic=time.time()
#                         for ii in range(value.shape[0]):
#                             for jj in range(value.shape[1]):
#                                 for kk in range(value.shape[2]):
#                                     resDict5[arname][ii, jj, kk] = np.hstack((resDict5[arname][ii, jj, kk], value[ii, jj, kk]))
#                                     if resDict5[arname][ii, jj, kk].shape[0] != temp_rD5[ii, jj, kk].shape[0]:
#                                         print('wrong shape')
#                                         pdb.set_trace()
#                                     if resDict5[arname][ii, jj, kk].shape[0] != 0:
#                                         if not (resDict5[arname][ii, jj, kk] == temp_rD5[ii, jj, kk]).all():
#                                             print('wrong value')
#                                             pdb.set_trace()
#                         toc=time.time()
#                         print(toc-tic, toc1-tic1)
#                         pdb.set_trace()
            if options.Week == 0:
                print('should this one be used?')
                print('Probably just use when created one outputfile per inputfile')
                print('intrusions as time series')
                sys.exit()
                SingleFiletoH5(mainDir, singlefileArea, umon, latrange, stDict, stDict2, stDict3, stDict4)
        
        else:
            jobs = []
            job_server = pp.Server(ncpus=antalKarnaor, ppservers=(), socket_timeout=3600)
#            print "Starting pp with", job_server.get_ncpus(), "workers"
            ticar = time.time()
            for fi in filen:
                k = k + 1
                print("%i av %i" %(k, nroffiles))
                clcfile, Geofile, Taufile, Cwcfile, Auxfile = getClcGeo(fi, mainClc, mainGeo, mainTau, mainCwc, mainAux)
                if len(clcfile) == 1:
                    Clcfile = clcfile[0]
                else:
                    missingCLC = missingCLC + 1
                    continue
                
                tempdir = '%s/%s' %(mainTemp, fCt)
                if not os.path.exists(tempdir):
                    os.makedirs(tempdir)
                tempname = '%s/%s_split%s_%s' %(tempdir, os.path.basename(fi).replace('.h5', ''), fCt, tempFileEnd)
                jobs.append(job_server.submit(findValuedData, (fi, Clcfile, Geofile, Taufile, Cwcfile, Auxfile, differentCloudTypes, latrange, OTlimit, tempname, options.loadTemp, fCt, options.resD), \
                                         (putValinRightPlace,readFLXHRLidar,DataObject,CloudsatObject,CloudsatObject, \
                                          convertDType, convertTime,readCloudType,readGeoType,readTauType,readCwcType,readAuxType,makeDataUsefull,\
                                          shrfactor,countVal,countValLarg, getOpticalDepthInd, calcWP), \
                                         ("numpy as np","h5py", "time", "math", "datetime", "glob", "warnings")))
            job_server.wait()
            for job in jobs:
                result = job()
#                 pdb.set_trace()
                if -1 in result:
                    continue
                stDict = copy.deepcopy(result[0])
                if nrf == 0:
                    nrf = 1
                    resDict = copy.deepcopy(result[0])
                    resDict.pop('fname')
                    resDict2 = copy.deepcopy(result[1])
                    resDict3 = copy.deepcopy(result[2])
                    resDict4 = copy.deepcopy(result[3])
                    if (len(differentCloudTypes) == 1) and (differentCloudTypes[0] == 90):
                        continue
                    else:
                        resDict5 = copy.deepcopy(result[4])
#                         resDict5_t = copy.deepcopy(resDict5)
                else:
                    for arname, value in result[0].items():                        
                        if arname == 'fname':
                            continue
#                         print(arname)
                        resDict[arname] = resDict[arname] + value
                    for arname, value in result[1].items():
                        resDict2[arname] = resDict2[arname] + value
                    for arname, value in result[2].items():
                        resDict3[arname] = resDict3[arname] + value
                    for arname, value in result[3].items():
                        resDict4[arname] = resDict4[arname] + value
                    if (len(differentCloudTypes) == 1) and (differentCloudTypes[0] == 90):
                        continue
                    else:
                        for arname, value in result[4].items():
                            try:
                                temp_rD5 = np.array([np.concatenate((a,b)) for a,b in zip(resDict5[arname].flat, value.flat)]).reshape(value.shape[0],value.shape[1],value.shape[2])
                            except:
#                                 print('da drum drum %s' %arname)
                                temp_rD5 = copy.deepcopy(resDict5[arname])
                                for ii in range(value.shape[0]):
                                    for jj in range(value.shape[1]):
                                        for kk in range(value.shape[2]):
                                            temp_rD5[ii, jj, kk] = np.hstack((temp_rD5[ii, jj, kk], value[ii, jj, kk]))
                            resDict5[arname] = temp_rD5
#                             for ii in range(value.shape[0]):
#                                 for jj in range(value.shape[1]):
#                                     for kk in range(value.shape[2]):
#                                         resDict5_t[arname][ii, jj, kk] = np.hstack((resDict5_t[arname][ii, jj, kk], value[ii, jj, kk]))
#                                         if resDict5_t[arname][ii, jj, kk].shape[0] != 0:
#                                             if len(resDict5_t[arname][ii, jj, kk]) != len(resDict5[arname][ii, jj, kk]):
#                                                 print('wrong length')
#                                                 pdb.set_trace()
#                                             else:
#                                                 if not (np.array(resDict5_t[arname][ii, jj, kk]) == np.array(np.asarray(resDict5[arname][ii, jj, kk]))).all():
#                                                     print('wrong value')
#                                                     pdb.set_trace()
#                                             
                if options.Week == 0:
                    print('should this one be used?')
                    print('Probably just use when created one outputfile per inputfile')
                    print('intrusions as time series')
                    sys.exit()
                    SingleFiletoH5(mainDir, singlefileArea, umon, latrange, stDict, result[1], result[2], result[3])
#            job_server.print_stats()
            job_server.destroy()
            tocar = time.time()
            print(tocar - ticar)
    endtime = time.time()
    print(endtime-sttime)
    print(filename)
    print(opptionsText)
    print('Missing CLC = %i' %missingCLC)
    if missingCLC == k:
        print('No CLC files')
    else:
        h5file = h5py.File(filename, 'w')
        h5file.create_dataset('Start Latitude', data = latrange)
        for otnum, val in OTlimit.items():
            h5file.create_dataset('Optical Thicknes %i' %otnum, data = val)

        grpcf = h5file.create_group('cf')
        grpcfD = grpcf.create_group('d')
        grpcfN = grpcf.create_group('n')
        for strname, val in resDict2.items():
            if strname[0] == 'n':
                grpcfN.create_dataset(strname, data = val)
            else:
                grpcfD.create_dataset(strname, data = val)
        grprh = h5file.create_group('rh')
        grprhD = grprh.create_group('d')
        grprhN = grprh.create_group('n')
        grprhDN = grprhD.create_group('quantity')
        grprhNN = grprhN.create_group('quantity')
        grprhDH = grprhD.create_group('heating')
        grprhNH = grprhN.create_group('heating')
        for strname, val in resDict.items():
            if '_n_' in strname:
                if 'antal' in strname:
                    grprhNN.create_dataset(strname, data = val)
                else:
                    grprhNH.create_dataset(strname, data = val)
            else:
                if 'antal' in strname:
                    grprhDN.create_dataset(strname, data = val)
                else:
                    grprhDH.create_dataset(strname, data = val)
        for strname, val in resDict4.items():
            if '_n_' in strname:
                if 'antal' in strname:
                    grprhNN.create_dataset(strname, data = val)
                else:
                    grprhNH.create_dataset(strname, data = val)
            else:
                if 'antal' in strname:
                    grprhDN.create_dataset(strname, data = val)
                else:
                    grprhDH.create_dataset(strname, data = val)
                    
            
        
        grptb = h5file.create_group('TOA-BOACRE')
        grptbD = grptb.create_group('d')
        grptbN = grptb.create_group('n')
        grptbDT = grptbD.create_group('TOACRE')
        grptbNT = grptbN.create_group('TOACRE')
        grptbDB = grptbD.create_group('BOACRE')
        grptbNB = grptbN.create_group('BOACRE')
        for strname, val in resDict3.items():
            if '_n_' in strname:
                if 'tshr' in strname or 'tlhr' in strname:
                    grptbNT.create_dataset(strname, data = val)
                else:
                    grptbNB.create_dataset(strname, data = val)
            else:
                if 'tshr' in strname or 'tlhr' in strname:
                    grptbDT.create_dataset(strname, data = val)
                else:
                    grptbDB.create_dataset(strname, data = val)
        if not ((len(differentCloudTypes) == 1) and (differentCloudTypes[0] == 90)):
            grpd4 = 'd4'
            grpd4D = 'd4/d'
            for strname, val in resDict5.items():
                h5file.create_dataset('d4/shape/' + strname, data = val.shape)
                for ii in range(val.shape[0]):
                    for jj in range(val.shape[1]):
                        for kk in range(val.shape[2]):
                            if len(val[ii, jj, kk]) == 0:
                                if val[ii, jj, kk] != []:
                                    print('satan')
                                    print(strname)
                                continue
                            else:
                                if '_n_' in strname:
                                    h5file.create_dataset('d4/n/%s/%i_%i_%i' %(strname, ii, jj, kk), data = val[ii, jj, kk])
                                else:
                                    h5file.create_dataset('d4/d/%s/%i_%i_%i' %(strname, ii, jj, kk), data = val[ii, jj, kk])
        h5file.close()
        print('Time to save fil = %f' %(time.time() - endtime))
        
        #: Remove temp files
#         filelist = glob.glob('%s/*%s.npy' %(mainTemp, tempFileEnd))
#         for f in filelist:
#             os.remove(f)
        print('Done')
#        alS = sum(resDict['lhr_a_90'][:,1])
#        fssS = 0
#        spS = sum(resDict['lhr_sp_90'][:,1])
#        fsspS = 0
#        for j in [0,1,2,3,4,5,6,7,8,10]:
#            fssS = fssS + sum(resDict['lhr_a_%i' %j][:,1])
#        for k in [0,1,2,3,4,5,6,7,8,10]:
#            fsspS = fsspS + sum(resDict['lhr_sp_%i' %k][:,1])
                
                
            
        