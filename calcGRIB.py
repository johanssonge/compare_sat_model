'''
Created on 16 Feb 2018

@author: a001721
'''
import h5py
import pdb
import numpy as np
from optparse import OptionParser
import matplotlib
from calcSat import all_months_comb
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time
import datetime
import sys
import os


#: 107 108 109 110 130 133 246 247 248 54 #: Minimum 10 * 12 = 120
#: 107 108 109 110 176 177 178 179 208 209 210 211 246 247 248 130 133 54 Aven med top/bottom radiation 18 * 12 = 216
#: for mon in 01 02 03 04 05 06 07 08 09 10 11 12; do export bashFile=Submit_jobs/Plot_Grib/submit_plot_grib_RMM.sh; export radOrg=107; export radSt=$radOrg; export yearOrg=2007; export yearSt=yearOrg; for rad in 107 108 109 110 176 177 178 179 208 209 210 211 246 247 248 130 133 54; do sed -i -e "s/${radSt}/${rad}/g" $bashFile; export radSt=${rad}; echo ${radSt}; for year in 2007 2008 2009 2010; do sed -i -e "s/${yearSt}/${year}/g" $bashFile; export yearSt=${year}; echo ${yearSt}; sbatch $bashFile; done; done; sed -i -e "s/${radSt}/${radOrg}/g" $bashFile; sed -i -e "s/${yearSt}/${yearOrg}/g" $bashFile; done
#: R = H eller L, MM = monad 01 02 03 ... 12
#: for mon in 01 02 03 04 05 06 07 08 09 10 11 12; do export bashFile=Submit_jobs/Plot_Grib/submit_plot_grib_R${mon}.sh; export radOrg=107; export radSt=$radOrg; export yearOrg=2007; export yearSt=yearOrg; for rad in 107 108 109 110 130 133 246 247 248 54; do sed -i -e "s/${radSt}/${rad}/g" $bashFile; export radSt=${rad}; echo ${radSt}; for year in 2007 2008 2009 2010; do sed -i -e "s/${yearSt}/${year}/g" $bashFile; export yearSt=${year}; echo ${yearSt}; sbatch $bashFile; done; done; sed -i -e "s/${radSt}/${radOrg}/g" $bashFile; sed -i -e "s/${yearSt}/${yearOrg}/g" $bashFile; done



def utcTimeApprox(localTime, lons):
        """Returns utc from local hour approximation"""
        hours = [((localTime - datetime.timedelta(hours=(l*12/180))).hour) + (((localTime - datetime.timedelta(hours=(l*12/180))).minute) / 60.) for l in lons]
        return hours


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-r', '--Rads', type='int', default=1, help = \
                      'rads')
    parser.add_option('-m', '--Month', type='int', default = -1, help = \
                      'Season')
    parser.add_option('-l', '--Low', action='store_true', default=False, help = \
                      'resolution')
    parser.add_option('-y', '--Year', type='int', default = 1, help = \
                      'Year')
    parser.add_option('-i', '--Interactive', action='store_true', default=False, help = \
                      'interactive mode')
    options, args = parser.parse_args()
    if not options.Interactive:
        allTime_st = time.time()
    ece_ver = 'NE'
    if options.Low:
        run = 'LR%s' %ece_ver
    else:
        run = 'HR%s' %ece_ver
    model = 'Primavera'
    model = '3.3.1'
    run = 'LRC6'
    mainDir = '/nobackup/smhid14/sm_erjoh/PhD-4'
#     if 'NE' in run:
#         mainDir = '/nobackup/smhid14/sm_erjoh/PhD-4'
#     else:
#         mainDir = '/nobackup/smhid13/sm_erjoh/PhD-4'
    readDir = '%s/%s/%s' %(mainDir, model, run)
    tempDir = '%s/TempFiles' %(mainDir)
    tempDir = tempDir.replace('smhid13', 'smhid14')
        
    #---
    if options.Rads == 1:
        rads = [107, 108, 109, 110]
        rad_name = {'c_sw': 107, 'c_lw': 108, 'a_sw': 109, 'a_lw': 110}
    elif options.Rads == 2:
        rads = [208, 209, 178, 179] # Top
        rad_name = {'c_sw': 208, 'c_lw': 209, 'a_sw': 178, 'a_lw': 179}
    elif options.Rads == 3:
        rads = [210, 211, 176, 177] # Bot
        rad_name = {'c_sw': 210, 'c_lw': 211, 'a_sw': 176, 'a_lw': 177}
    elif options.Rads == 4:
        rads = [130, 133, 246, 247, 248, 54]
        rad_name = {130: 'Temp', 133: 'Spec_hum', 246: 'Spec_liq', 247: 'Spec_ice', 248: 'Cloud_Cover', 54: 'Pressure'}
    else:
        rads = [options.Rads]
        rad_name = {rads[0]: rads[0]}
    #---
    if options.Year == 1:
        years = [2007, 2008, 2009, 2010]
        year_name = 'all'
    else:
        years = [options.Year]
        year_name = '%i' %options.Year
    #---
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
        years = [2007, 2008, 2009, 2010]
        months = range(1,13)
        season = 'all'
    else:
        season = ('%02i' %options.Month)
        months = [options.Month]
    print(season)
    
    #---
#     dt_obj_s = time.strptime('2005-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
#     sec_s = time.mktime(dt_obj_s)
    
    local_time = datetime.datetime.strptime('2005-01-01 13:30:00', "%Y-%m-%d %H:%M:%S")
    avail_h = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
    res_rh_ver = {}
    res_rh_lon = {}
    res_rh = {}
    use_datline_center = False
    load_temp = True
    if not options.Interactive:
        load_temp = False
    print('load temp = %s' %load_temp)
    latb = [-30, 30]
#     lonb = [0, 360]
    if use_datline_center:
        lonb = [0,360]
    else:
        lonb = [-180, 180]
    p = -1 #: Number for rad
    a = -1 #: rad, year, month
    for pid in rads:
        print('pid = %i' %pid)
        p = p + 1
        d = -1 #: year, month
        for year in years:#[2007, 2008, 2009, 2010]:
            print('year = %i') %year
            for mon in months:#[12, 1, 3]:
                if options.Month in [-5, -6]:
                    if mon not in enso_month[str(year)]:
                        continue
                print('mon = %0.2i' %mon)
                a = a + 1
                d = d + 1
                tempname = '%s/temp_%s_%s_%i_%0.2i_%i.h5' %(tempDir, model, run, year, mon, pid)
#                 tempname = tempname.replace('.h5', '_one_file.h5')
#                 tempname = tempname.replace('.h5', '_one_day.h5')
#                 tempname = tempname.replace('.h5', '_same_time.h5')
                print(tempname)
                
                if os.path.isfile(tempname) and load_temp:
#                     if a == 0:
#                         h5file_latlon = h5py.File('', 'r')
                    h5file = h5py.File(tempname, 'r')
                    res_m = h5file['res m'].value
                    lat_t = h5file['lat'].value
                    lon = h5file['lon'].value
                    num_day = h5file['num_day'].value
                    lon180 = np.where(lon > 180, lon - 360, lon)
                    ind180 = np.argsort(lon180)
                    lon180 = lon180[ind180]
                    lonind = ((lon180 >= lonb[0]) & (lon180 <= lonb[1]))

                    def controlLat(val, lats):
                        latind = np.argsort(lats) #: South on top
                        if latind[0] == 0: #: redan sort
                            ret_lat = lats
                            retv = val
                        else:
                            ret_lat = lats[::-1]
                            retv = val[:, ::-1, :]
                        return retv, ret_lat
                    res_m, lat = controlLat(res_m, lat_t)
                    latind = ((lat >= latb[0]) & (lat <= latb[1]))
                    if a == 0:
                        
                        #: Create result files
                        utc = h5file['utc'].value
                        height = h5file['height'].value
                        height.sort()
                        num_height_lev = len(height)
                        if num_height_lev == 50:
                            height_240_mask = np.asarray(height[1:])-np.asarray(height[0:-1]) == 240
                            height_240_mask = np.concatenate((height_240_mask, [False]))
                            num_height_lev_480 = num_height_lev - int((height_240_mask).sum() / 2.)
                    h5file.close()
                else:
                    tic = time.time()
                    secToDay = 1
                    if pid in [107, 108, 109, 110, 111, 112]:
                        secToDay = (24 * 3600.)
                    else:
                        secToDay = 1.
                    yeardir = '%s/%03i' %(readDir, year-2004)
                    filename = '%s/ICMGG%s+%i%02i_%i.h5' %(yeardir, run, year, mon, pid)
                    filename_latlon = "%s/latlon_%s.h5" %(yeardir, run)
                    h5file = h5py.File(filename, 'r')
                    h5file_latlon = h5py.File(filename_latlon, 'r')
                    
                    latitude = h5file_latlon['latitude'].value
                    longitude = h5file_latlon['longitude'].value
                    h5file_latlon.close()
                    lat = latitude[:, 0]
                    lon = longitude[0, :]
                    
                    latind = np.ones(lat.shape[0]).astype('bool')#((lat >= latb[0]) & (lat <= latb[1]))
                    lon180 = np.where(lon > 180, lon - 360, lon)
                    ind180 = np.argsort(lon180)
                    lon180 = lon180[ind180]
                    lonind = np.ones(lon.shape[0]).astype('bool')# ((lon180 >= lonb[0]) & (lon180 <= lonb[1]))
                    if a == 0:
#                         if 
                        utc = utcTimeApprox(local_time, lon)
                        utc = np.asarray(utc)
                        #: Create result files
                        
                        height_lev = h5file[h5file.keys()[0]].keys()
                        if len(height_lev) == 44 and '0' in height_lev:
                            zero_place = np.where(np.array(height_lev) == '0')[0][0]
                            zero_name = height_lev.pop(zero_place)
                            if zero_name != '0':#: Sometime thee is a zero level eventhogh the first level shold be 240 
                                print('What Zero??')
                                pdb.set_trace()
                            height_lev
                        height = []
                        for h in height_lev:
                            height.append(int(h))
                        height.sort()
                        num_height_lev = len(height)
                        
                        if num_height_lev == 50:#pid <= 112:
                            height_240_mask = np.asarray(height[1:])-np.asarray(height[0:-1]) == 240
                            height_240_mask = np.concatenate((height_240_mask, [False]))
                            num_height_lev_480 = num_height_lev - int((height_240_mask).sum() / 2.)
                    res_m = np.zeros([num_height_lev, latind.sum(), lonind.sum()]) #: 31=max num days in a month
                    res2 = np.zeros([num_height_lev, latind.sum(), lonind.sum()]) #: 31=max num days in a month
                    num_day = -1
                    for ech in avail_h[::-1]:
                        if ('_one_file.h5' in tempname) or ('_same_time.h5' in tempname):
                            one_file_time = 15
                            if ech != one_file_time:
                                continue
                            utc[:] = one_file_time
                        print(ech)
                        ech_ind = np.where(np.abs(utc-ech) <= 1.5)[0]
                        ech_n = ech
                        if ech == 24:
                            ech_n = 0
                        
                        t = -1
                        for arname, levels in h5file.items():
                            if arname in ['Latitude', 'latitude', 'longitude', 'missingValue']:
                                continue
                            dt_obj = datetime.datetime.strptime(arname, "%Y-%m-%d %H:%M:%S")
                            if dt_obj.hour not in [ech_n]:
                                continue
                            #: This day do not exist for time 0 (24)
                            if (year == 2010) and (mon in [12]) and (dt_obj.day == 1):
                                continue
                            if ('_one_day.h5' in tempname):
                                if dt_obj.day != 5:# and dt_obj.day != 10:
                                    continue
                            print arname
                            t = t + 1
                            dt_obj_neg = dt_obj - datetime.timedelta(hours=(3))
                            arname_neg = dt_obj_neg.isoformat().replace('T', ' ')
                            dt_obj_pos = dt_obj + datetime.timedelta(hours=(3))
                            arname_pos = dt_obj_pos.isoformat().replace('T', ' ')
                            if arname_neg in h5file.keys():
                                levels_neg = h5file[arname_neg]
                            else:
                                levels_neg = levels
                            if arname_pos in h5file.keys():
                                levels_pos = h5file[arname_pos]
                            else:
                                levels_pos = levels
                            #: Ind for neg and pos value
                            ech_neg_ind = ech_ind[(utc-ech)[ech_ind] <= 0] 
                            ech_pos_ind = ech_ind[((utc-ech)[ech_ind] > 0) & ((utc-ech)[ech_ind] < 1.5)] #: It doesent mather if 0 or 1.5 is in pos or neg, same value
                            #: Weight for neg and pos value, for neg and pos ind
                            we_neg_2 = np.abs((utc-ech)[ech_neg_ind]) * (0.5/1.5)
                            we_pos_2 = np.abs((utc-ech)[ech_pos_ind]) * (0.5/1.5)
                            #: Weight for original value, for neg and pos ind
                            we_neg_1 = 1 - we_neg_2
                            we_pos_1 = 1 - we_pos_2
                            e = -1

                            for h in height:
                                e = e + 1
                                #: Original value, for neg and pos ind
                                val_neg_ech = levels[str(h)].value[latind, :][:, ech_neg_ind]
                                val_pos_ech = levels[str(h)].value[latind, :][:, ech_pos_ind]
                                #: Neg and pos value, for neg and pos ind
                                val_neg_ech_neg = levels_neg[str(h)].value[latind, :][:, ech_neg_ind]
                                val_pos_ech_pos = levels_pos[str(h)].value[latind, :][:, ech_pos_ind]
                                #: Weighted values for neg and pos ind
                                data_neg = we_neg_1 * val_neg_ech + we_neg_2 * val_neg_ech_neg
                                data_pos = we_pos_1 * val_pos_ech + we_pos_2 * val_pos_ech_pos
#                                 #: Add to old value. This has to be done before puting in place incase of overlap between pos and neg
                                data_neg_sum = res_m[e, :, :][:, ech_neg_ind] + data_neg * secToDay
                                data_pos_sum = res_m[e, :, :][:, ech_pos_ind] + data_pos * secToDay
                                #: Test
#                                 val_neg_echT = levels[str(h)].value[latind, :][:, ech_neg_indT]
#                                 val_pos_echT = levels[str(h)].value[latind, :][:, ech_pos_indT]
#                                 #: Neg and pos value, for neg and pos ind
#                                 val_neg_ech_negT = levels_neg[str(h)].value[latind, :][:, ech_neg_indT]
#                                 val_pos_ech_posT = levels_pos[str(h)].value[latind, :][:, ech_pos_indT]
#                                 #: Weighted values for neg and pos ind
#                                 data_negT = we_neg_1T * val_neg_echT + we_neg_2T * val_neg_ech_negT
#                                 data_posT = we_pos_1T * val_pos_echT + we_pos_2T * val_pos_ech_posT
# #                                 #: Add to old value. This has to be done before puting in place incase of overlap between pos and neg
#                                 data_neg_sumT = res2[e, :, :][:, ech_neg_indT] + data_negT * secToDay
#                                 data_pos_sumT = res2[e, :, :][:, ech_pos_indT] + data_posT * secToDay
#                                 res2[e, :, :][:, ech_neg_indT] = data_neg_sumT
#                                 res2[e, :, :][:, ech_pos_indT] = data_pos_sumT

                                #: Data put in right place in res. OBS; If 0 in both pos ind and neg ind, pos ind will ovevright neg ind. Same value though
                                res_m[e, :, :][:, ech_neg_ind] = data_neg_sum
                                res_m[e, :, :][:, ech_pos_ind] = data_pos_sum

                            if '_one_file.h5' in tempname:
                                break
                            
                    toc = time.time()
                    print('read file time = %i s' %(toc-tic))
                    if num_day == -1:
                        num_day = t + 1
                    else:
                        if num_day != (t + 1):
                            print('wrong num_day')
                            pdb.set_trace()
                    
                    if ('_one_file.h5' not in tempname) and ('_same_time.h5' not in tempname) and ('_one_day.h5' not in tempname):
                        h5file_save = h5py.File(tempname, 'w')
                        h5file_save.create_dataset('res m', data = res_m)
                        h5file_save.create_dataset('utc', data = utc)
                        h5file_save.create_dataset('height', data = height)
                        h5file_save.create_dataset('lat', data = lat)
                        h5file_save.create_dataset('lon', data = lon)
                        h5file_save.create_dataset('num_day', data = num_day)
                        h5file_save.close()
                        h5file.close()
                if options.Interactive:
                    if d == 0:
                        t_1 = num_day
                        res = res_m.copy()
                    else:
                        res = res + res_m
                        t_1 = t_1 + num_day
        #: Gors for varje pid
        if options.Interactive:
#             pdb.set_trace()
            res = np.divide(res, t_1)
            if options.Rads in [2,3]:
                #: Dela med tiden fran forra utskriften (3h) for att go fran J/m2 till W/m2
                res = np.divide(res, (3 * 3600))
            if res.shape[0] == 50:
#                 if ('_one_file.h5' not in tempname) or ('_same_time.h5' not in tempname):
#                     lon180_mask = np.ones(lon180.shape).astype(bool)
#                 else:
#                     lon180_mask = (lon180<=45) & (lon180>=-45)
                lon180_mask = np.ones(lon180.shape).astype(bool)
                height_240_mask_even = np.where(height_240_mask)[0][::2]
                height_240_mask_uneven = np.where(height_240_mask)[0][1::2]
                res_240 = np.nanmean((res[height_240_mask_even, :, :], res[height_240_mask_uneven, :, :]), axis=0)  # @UndefinedVariable
                res_480 = res[~height_240_mask, :, :]
                height_240 = np.asarray(height)[height_240_mask_even]
                height_480 = np.asarray(height)[~height_240_mask]
                res_lr = np.concatenate((res_240, res_480), axis=0)
                height_lr = np.concatenate((height_240, height_480))
                res_rh_lon.update({pid: np.mean(res_lr, axis=(1))})
                res_rh_ver.update({pid: np.mean(res_lr[:, :, lon180_mask], axis=(1,2))})
                res_rh.update({pid: res_lr})
#                 res_rh_lon[p, :] = np.mean(res_lr, axis=(1))
#                 res_rh_ver[p] = np.mean(res_lr[:, :, lon180_mask], axis=(1,2))
            elif res.shape[0] == 43:
                lon180_mask = np.ones(lon180.shape).astype(bool)
                height_lr = height
                res_rh_lon.update({pid: np.mean(res, axis=(1))})
                res_rh_ver.update({pid: np.mean(res[:, :, lon180_mask], axis=(1,2))})
                res_rh.update({pid: res})
                
            else:
                res_rh_lon.update({pid: np.mean(res, axis=0)})
                res_rh.update({pid: res})
#             print('hej')
#             pdb.set_trace()
    
    if not options.Interactive:
        allTime_end = time.time()
        print(allTime_end - allTime_st)
    if options.Interactive:
        #: Save Data
        if options.Rads in [1, 2, 3]:
            crh_c_sw = res_rh[rad_name['c_sw']][:, :, ind180]
            crh_c_lw = res_rh[rad_name['c_lw']][:, :, ind180]
            crh_a_sw = res_rh[rad_name['a_sw']][:, :, ind180]
            crh_a_lw = res_rh[rad_name['a_lw']][:, :, ind180]
            save_dict = {'c_sw': crh_c_sw, 'a_sw': crh_a_sw, \
                         'c_lw': crh_c_lw, 'a_lw': crh_a_lw}
        elif options.Rads in [4]:
            #: Caclulate WC
            rd = 287.04
            rv = 461.
            #:==================================================
            #:    Virtual temperature
            #:    Tv = T ( 1+ (Rv/Rd-1)*qv)
            #:==================================================
            Tv = res_rh[130] * (1 + (((rv / rd) - 1) * res_rh[133]))
            #:==================================================
            #:    Density
            #:    rho =  p /(Rd * Tv)
            #:==================================================
            rho = res_rh[54] / (rd * Tv)
            rho = rho * 1000.
            #:==================================================-
            #: convert cwc to cwc_gm3
            #:CLWC_gm3 = ( CLWC / CC ) * rho
            #:==================================================
            cf_nan = np.where(res_rh[248] == 0, np.nan, res_rh[248])
            try:
                CIWC_gm3 = (res_rh[247] / cf_nan) * rho
                CLWC_gm3 = (res_rh[246] / cf_nan) * rho
            except:
                pdb.set_trace()
            #: IWC devided by cf in g/m3
            CIWC_gm3 = np.where(np.isinf(CIWC_gm3), np.nan, CIWC_gm3)
            CLWC_gm3 = np.where(np.isinf(CLWC_gm3), np.nan, CLWC_gm3)
            #: IWC NOT devided by cf in g/m3
            IWC_gm3 = res_rh[247] * rho
            LWC_gm3 = res_rh[246] * rho
            save_dict = {'ciwc': CIWC_gm3[:, :, ind180], \
                         'clwc': CLWC_gm3[:, :, ind180], \
                         'iwc': IWC_gm3[:, :, ind180], \
                         'lwc': LWC_gm3[:, :, ind180], \
                         'cfd': res_rh[248][:, :, ind180]}
            
        elif options.Rads in [248]:
            cfd = res_rh[248][:, :, ind180]
            save_dict = {'cfd': cfd}
        else:
            print(options.Rads)
            pdb.set_trace()

#         rhd = {'%sre' %run.lower()[0]: res_rh[rad_name['a_sw']]}#((res_rh[rad_name['a_sw']] - res_rh[rad_name['c_sw']]) + (res_rh[rad_name['a_lw']] - res_rh[rad_name['c_lw']]))}
        

        save_dict.update({'lon180': lon180})
        save_dict.update({'lon': lon})
        save_dict.update({'lat': lat})
        if options.Rads not in [2,3]:
            save_dict.update({'height': height_lr})
        savename = 'Clim_val/%s_y-%s_s-%s_rad-%i' %(run, year_name, season, options.Rads)
#         if '_extra' in tempname:
#             savename = savename + '_extra'
        np.save(savename, [save_dict])
        sys.exit()
        #: Below is done in plotData.py
        
        
        
        
        
        if pid <= 112:
            crh_ver_c_sw = res_rh_ver[107]
            crh_ver_c_lw = res_rh_ver[108]
            crh_ver_a_sw = res_rh_ver[109]
            crh_ver_a_lw = res_rh_ver[110]
            crh_ver_c = crh_ver_c_sw + crh_ver_c_lw
            crh_ver_a = crh_ver_a_sw + crh_ver_a_lw
            crh_ver = crh_ver_a - crh_ver_c
            crh_ver_sw = crh_ver_a_sw - crh_ver_c_sw
            crh_ver_lw = crh_ver_a_lw - crh_ver_c_lw
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(crh_ver, height_lr, 'k', label= 'NET')
            ax.plot(crh_ver_sw, height_lr, 'r', label= 'SW')
            ax.plot(crh_ver_lw, height_lr, 'b', label= 'LW')
            ax.axvline(0,0,1,color='g')
            ax.legend()
            ax.set_xlabel('Cloud Radiative Heating [K / day]')
            ax.set_ylabel('Height [km]')
    #         ax.set_title('%s, Total Sky - Clear Sky' %run[0:2])
#             ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
            yticks = ax.get_yticks()
            ax.set_yticks(yticks)
            ax.set_yticklabels((yticks / 1000).astype('int'))
    #         ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])
            figname = 'Plots/%s_%s_tot-clr' %(run, mon)
            if '_one_file.h5' in tempname:
                figname = figname + '_one_file_%i' %one_file_time
            fig.savefig(figname + '.png')
            fig.show()
        elif pid == 248:
            cf_ver = res_rh_ver[248]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(cf_ver, height, 'k', label= 'CF')
    #         ax.axvline(0,0,1,color='g')
            ax.legend()
            ax.set_xlabel('Cloud Fraction')
            ax.set_ylabel('Height [km]')
    #         ax.set_title('%s, Total Sky - Clear Sky' %run[0:2])
#             ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
#             ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
            yticks = ax.get_yticks()
            ax.set_yticks(yticks)
            ax.set_yticklabels((yticks / 1000).astype('int'))
    #         ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])
            
            figname = 'Plots/%s_%s_cf' %(run, mon)
#             fig.savefig(figname + '.png')
            fig.show()
            pdb.set_trace()
            sys.exit()
        climName = 'Clim_val/res_rh_lon_clim_value_rads_%i' %options.Rads
        useClim = True
        if options.Rads in [2, 3] and options.Month in [-5, -6]:
            useClim = False
        if options.Month == -100:
            np.save(climName, [res_rh_lon])
            if options.Rads == 1:
                np.save(climName.replace('rh_lon', 'rh'), [res_rh])
            sys.exit()
        if useClim:
            clim_rh_lon = np.load(climName + '.npy')[0]
            clim_c_sw = clim_rh_lon[rad_name['c_sw']]
            clim_c_lw = clim_rh_lon[rad_name['c_lw']]
            clim_a_sw = clim_rh_lon[rad_name['a_sw']]
            clim_a_lw = clim_rh_lon[rad_name['a_lw']]
            clim_c = clim_c_sw + clim_c_lw
            clim_a = clim_a_sw + clim_a_lw
            clim = clim_a - clim_c
            clim_sw = clim_a_sw - clim_c_sw
            clim_lw = clim_a_lw - clim_c_lw
        
        crh_lon_c_sw = res_rh_lon[rad_name['c_sw']]
        crh_lon_c_lw = res_rh_lon[rad_name['c_lw']]
        crh_lon_a_sw = res_rh_lon[rad_name['a_sw']]
        crh_lon_a_lw = res_rh_lon[rad_name['a_lw']]
        crh_lon_c = crh_lon_c_sw + crh_lon_c_lw
        crh_lon_a = crh_lon_a_sw + crh_lon_a_lw
        crh_lon = crh_lon_a - crh_lon_c
        
        crh_lon_sw = crh_lon_a_sw - crh_lon_c_sw
        crh_lon_lw = crh_lon_a_lw - crh_lon_c_lw
        if useClim:
            crh_lon_dict = {'c_sw': crh_lon_c_sw - clim_c_sw, 'a_sw': crh_lon_a_sw - clim_a_sw, \
                            'sw': crh_lon_sw - clim_sw, 'c_lw': crh_lon_c_lw - clim_c_lw, \
                            'a_lw': crh_lon_a_lw - clim_a_lw, 'lw': crh_lon_lw - clim_lw, \
                            'c_crh': crh_lon_c - clim_c, 'a_crh': crh_lon_a - clim_a, \
                            'crh': crh_lon - clim}
        else:
            crh_lon_dict = {'c_sw': crh_lon_c_sw, 'a_sw': crh_lon_a_sw, 'sw': crh_lon_sw, \
                            'c_lw': crh_lon_c_lw, 'a_lw': crh_lon_a_lw, 'lw': crh_lon_lw, \
                            'c_crh': crh_lon_c, 'a_crh': crh_lon_a, 'crh': crh_lon}
        
        i_name = {0: 'a_', 1: 'c_', 2: ''}
        j_name = {0: 'sw', 1: 'lw', 2: 'crh'}
        if pid <= 112:
            crh_c_sw = res_rh[rad_name['c_sw']]
            crh_c_lw = res_rh[rad_name['c_lw']]
            crh_a_sw = res_rh[rad_name['a_sw']]
            crh_a_lw = res_rh[rad_name['a_lw']]
            crh_c = crh_c_sw + crh_c_lw
            crh_a = crh_a_sw + crh_a_lw
            rhd = crh_a - crh_c #: Borde hetat crh men osaker pa om den kan forvaxlas'
            if useClim:
                clim_rh = np.load(climName.replace('rh_lon', 'rh') + '.npy')[0]
                clim_c_sw = clim_rh[rad_name['c_sw']]
                clim_c_lw = clim_rh[rad_name['c_lw']]
                clim_a_sw = clim_rh[rad_name['a_sw']]
                clim_a_lw = clim_rh[rad_name['a_lw']]
                clim_c = clim_c_sw + clim_c_lw
                clim_a = clim_a_sw + clim_a_lw
                clim = clim_a - clim_c
                rhd = rhd - clim
            if run == 'HR60':
                aspect = 10
                aspect2 = 6
            else:
                aspect = 5
                aspect2 = 3
            yticks = [1, 10, 20, 31, 41]
            ylabels = ['0', '5', '10', '15', '20']
            ylabel = 'Height [km]'
            valminmax_tot = 2
            valminmax_sep = 4
            if useClim:
                valminmax_tot = 1
                valminmax_sep = 2
                
    
        else:
            valminmax_tot = 1000000
            valminmax_sep = 10000000
            if useClim:
                valminmax_tot = 2000000
                valminmax_sep = 2000000
            aspect = 3
            if run == 'HR60':
                yticks = [1, 42, 84, 126, 169]
            else:
                yticks = [1, 21, 42, 63, 84]
            ylabels = ['-30', '-15', '0', '15', '30']
            ylabel = 'Latitude [deg]'
#             barticks = [valminmax*-1, valminmax*-0.5, 0, valminmax*0.5, valminmax]
#         if sys.version.split()[0] == '2.7.9':
#         from mpl_toolkits.basemap import Basemap
        if run == 'HR60':
            xticks = [1, 128, 256, 384, 512, 640, 768, 896, 1023]
        else:
            xticks = [1, 64, 128, 192, 256, 320, 384, 448, 511]
        if use_datline_center:
            xlabels = ['0', '45', '90', '135', '180', '225' ,'270', '215', '360']
        else:
            xlabels = ['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180']
        f = 0
        fig = plt.figure(figsize = (18,12))
        for i in range(3):
            for j in range(3):
                f = f + 1
                if f in [1,2,3,4,5,6]:
                    valminmax = valminmax_sep
                else:
                    valminmax = valminmax_tot
                barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
                ax = fig.add_subplot(3,3,f)
                sub_name = i_name[i] + j_name[j] 
                plot_val = crh_lon_dict[sub_name]
#                 m = Basemap(projection='cyl',llcrnrlat=-30,urcrnrlat=30,\
#                             llcrnrlon=-180,urcrnrlon=180,resolution='c')
#                 m = Basemap(projection='kav7',lon_0=0,resolution='c')
                im = ax.imshow(plot_val[:, ind180], origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax*-1, vmax=valminmax)
#                 im = m.imshow(plot_val[:, ind180], origin='lower', cmap='RdBu_r', aspect=aspect, vmin=valminmax*-1, vmax=valminmax)
#                 m.drawcoastlines()
                ax.set_title(sub_name)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels)
#             ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#             plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)
                if f in [1,4,7]:
                    ax.set_ylabel(ylabel)
                if f in [7,8,9]:
                    ax.set_xlabel('Longitude [deg]')
#                 print('hej')
                if f in [3,6,9]:
#                         cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                    fig.subplots_adjust(right=0.95)
                    if f == 3:
                        cbar_ax = fig.add_axes([0.96, 0.64, 0.008, 0.24])
                    if f == 6:
                        cbar_ax = fig.add_axes([0.96, 0.365, 0.008, 0.24])
                    if f == 9:
                        cbar_ax = fig.add_axes([0.96, 0.095, 0.008, 0.24])
                    cbar = fig.colorbar(im, cax=cbar_ax)
#                     cbar = fig.colorbar(im, aspect=50)#, orientation='horizontal')#, ticks=barticks)#, fraction=0.9))#, cax=cax, ticks=[valminmax*-1, 0, valminmax])#, fraction=0.9)
                    if pid > 112:
                        cbar.formatter.set_powerlimits((0, 0))
                        cbar.update_ticks()
#         plt.tight_layout()
        figname = 'Plots/%s_y_%s_s_%s_lon_tot-clr' %(run, year_name, season)
        if options.Rads == 2:
            figname.replace('_lon_', 'Top')
        elif options.Rads == 3:
            figname.replace('_lon_', 'Bot')
        if '_one_file.h5' in tempname:
            figname = figname + '_one_file_%i' %one_file_time
        fig.show()
        if useClim:
            figname = figname + '_anom'
        fig.savefig(figname + '.png')
        if options.Rads in [2, 3]:
            pdb.set_trace()
            sys.exit()

        valminmax = valminmax_tot
        stLat = lat[latind]
        fig = plt.figure(figsize=(8,8))    
        f = 0
        for stepL in range(-30,30,15)[::-1]:
            f = f + 1
            endLat = stepL + 15
            stepInd = (stLat >= stepL) & (stLat < endLat)
            rhdLat = np.nanmean(rhd[:,stepInd,:], axis=1)  # @UndefinedVariable
            ax = fig.add_subplot(4,1,f)
            im = ax.imshow(rhdLat[:, ind180], origin='lower', cmap='RdBu_r', aspect=aspect2, vmin=valminmax*-1, vmax=valminmax)
        
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)
            ax.set_xticks(xticks)
            ax.set_title('Latitude, %i - %i' %(stepL, endLat))
            ax.set_ylabel('Height [km]')
            if f in [4]:
                ax.set_xticklabels(xlabels)
                ax.set_xlabel('Longitude [deg]')
                barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
#                 cbar_ax = fig.add_axes([0.17, 0.04, 0.7, 0.01])
                cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.01])
                cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)
    #             cbar = fig.colorbar(im, orientation='horizontal', aspect=50, ticks=barticks)
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
        figname = 'Plots/%s_y_%s_s_%s_lon_15deg-step' %(run, year_name, season)
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
            rhdLat = np.nanmean(rhd[:,stepInd,:], axis=1)  # @UndefinedVariable
            ax = fig.add_subplot(12,1,f)
            im = ax.imshow(rhdLat[:, ind180], origin='lower', cmap='RdBu_r', aspect=aspect2, vmin=valminmax*-1, vmax=valminmax)
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)
#             ax.set_yticks([1, 21, 42, 63, 83, 104])
#             ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])
            ax.set_xticks(xticks)
            ax.set_title('Latitude, %i - %i' %(stepL, endLat))
    #         ax.set_title(sub_name)
            if f in [6]:
                ax.set_ylabel('Height [km]')
            if f in [12]:
                ax.set_xticklabels(xlabels)
                ax.set_xlabel('Longitude [deg]')
                barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
#                 cbar_ax = fig.add_axes([0.17, 0.009, 0.7, 0.002])
                cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.01])
                cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)
#                 cbar = fig.colorbar(im, orientation='horizontal', aspect=50, ticks=barticks)
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
#         plt.tight_layout()
        fig.show()
        figname = 'Plots/%s_y_%s_s_%s_lon_5deg-step' %(run, year_name, season)
        if useClim:
            figname = figname + '_anom'
        fig.savefig(figname + '.png')
        
        fig = plt.figure(figsize=(8,12))    
        f = 0
        for stepL in range(-30,0,5)[::-1]:
            f = f + 1
            endLat = np.abs(stepL)
            stepInd = (stLat >= stepL) & (stLat < endLat)
            rhdLat = np.nanmean(rhd[:,stepInd,:], axis=1) # @UndefinedVariable
            ax = fig.add_subplot(6,1,f)
            im = ax.imshow(rhdLat[:, ind180], origin='lower', cmap='RdBu_r', aspect=aspect2, vmin=valminmax*-1, vmax=valminmax)
        
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)

            ax.set_xticks(xticks)
            ax.set_title('Latitude, %i - %i' %(stepL, endLat))
    #         ax.set_title(sub_name)
            if f in [3]:
                ax.set_ylabel('Height [km]')
            if f in [6]:
                ax.set_xticklabels(xlabels)
                ax.set_xlabel('Longitude [deg]')
                cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.01])
                cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=barticks)
#                 barticks = [valminmax*-1, valminmax*-0.75, valminmax*-0.5, valminmax*-0.25, 0, valminmax*0.25, valminmax*0.5, valminmax*0.75, valminmax]
#                 cbar = fig.colorbar(im, orientation='horizontal', aspect=50, ticks=barticks)
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])
#         plt.tight_layout()
        fig.show()
        figname = 'Plots/%s_y_%s_s_%s_lon_Ldeg-step' %(run, year_name, season)
        if useClim:
            figname = figname + '_anom'
        fig.savefig(figname + '.png')
        pdb.set_trace()
        
        sys.exit()
        
        
        
        
        
        
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(crh_ver_a, height, 'k', label= 'NET')
        ax.plot(res_rh_ver[2,:], height, 'r', label= 'SW')
        ax.plot(res_rh_ver[3,:], height, 'b', label= 'LW')
        ax.axvline(0,0,1,color='g')
        ax.set_title('%s, Total Sky' %run[0:2])
        ax.legend()
        figname = 'Plots/%s_tot' %run
        fig.savefig(figname + '.png')
        fig.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(crh_ver_a, height, 'k', label= 'NET')
        ax.plot(res_rh_ver[0,:], height, 'r', label= 'SW')
        ax.plot(res_rh_ver[1,:], height, 'b', label= 'LW')
        ax.axvline(0,0,1,color='g')
        ax.set_title('%s, Clear Sky' %run[0:2])
        ax.legend()
        figname = 'Plots/%s_clr' %run
        fig.savefig(figname + '.png')
        fig.show()
        
        pdb.set_trace()
    
'101_SW_heating_rates_clear_sky_accum'
'102_LW_heating_rates_clear_sky_accum'
'103_SW_heating_rates_total_accum'
'104_LW_heating_rates_total_accum'
'105_net_SW_radiative_flux_accum'
'106_net_LW_radiative_flux_accum'
'107_SW_heating_rates_clear_sky'
'108_LW_heating_rates_clear_sky'
'109_SW_heating_rates_total'
'110_LW_heating_rates_total'
'111_net_SW_radiative_flux'
'112_net_LW_radiative_flux'

# [208, 209, 178, 179] Top
# [210, 211, 176, 177] Bot
# 176    SSR    Surface solar radiation    W m-2 s
# 177    STR    Surface thermal radiation  W m-2 s
# 178    TSR    Top solar radiation    W m-2 s
# 179    TTR    Top thermal radiation    W m-2 s
# 
# 208    TSRC    Top net solar radiation, clear sky    W m-2
# 209    TTRC    Top net thermal radiation, clear sky    W m-2
# 210    SSRC    Surface net solar radiation, clear sky    W m-2
# 211    STRC    Surface net thermal radiation, clear sky    W m-2    

# 248 Cloud fraction
# 



















    