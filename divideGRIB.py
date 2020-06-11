'''
Created on 13 Feb 2018

@author: a001721
'''
from matplotlib import pyplot as plt  # @UnusedImport
from mpl_toolkits.basemap import Basemap, cm  # @UnresolvedImport @UnusedImport
import pygrib  # @UnresolvedImport
import pdb
import numpy as np
import time
import os
import h5py
from optparse import OptionParser
import sys
#for m in 01 02 03 04 05 06 07 08 09 10 11 12; do export bashFile=Submit_jobs/divedGrib/submit_divideGrib_L${m}.sh; yearSt=2007; yearOrg=${yearSt}; for year in 2007 2008 2009 2010; do sed -i -e "s/${yearSt}/${year}/g" $bashFile; yearSt=${year}; echo ${yearSt}; sbatch $bashFile; done; sed -i -e "s/${yearSt}/${yearOrg}/g" $bashFile; done
#for m in 01 02 03 04 05 06 07 08 09 10 11 12; do export bashFile=Submit_jobs/divedGrib/submit_divideGrib_L${m}.sh; yearSt=2007; yearOrg=${yearSt}; for year in 2007 2008 2009 2010; do sed -i -e "s/${yearSt}/${year}/g" $bashFile; yearSt=${year}; echo ${yearSt}; done; sed -i -e "s/${yearSt}/${yearOrg}/g" $bashFile; done
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--Month', type='int', default = 1, help = \
                      'Specify Month, deafault 1')
    parser.add_option('-y', '--Year', type='int', default = 2007, help = \
                      'Specify Year, deafault 2007')
    
    options, args = parser.parse_args()
    
#     model = 'Primavera'
    model='3.3.1'
    bi = True
    if bi:
        run = 'LRC6'
#         run = 'LRNE'
        outdir = '/nobackup/smhid14/sm_erjoh/PhD-4/%s/%s' %(model, run)
        rundir = '/nobackup/smhid13/sm_erjoh/PhD-4/run/%s/%s' %(model, run)
    else:
        run = 'LRN2'
        outdir = '/proj/bolinc/users/sm_erjoh/PhD-4/%s/%s' %(model, run)
        rundir = '/proj/bolinc/users/sm_erjoh/PhD-4/run/%s/%s' %(model, run)
#     run = 'RAD0'
    for year in [options.Year]:
        print('year = %i' %year)
        savedir = '%s/%03i' %(outdir, year-2004)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        for mon in [options.Month]:
            print('month = %i' %mon)
            ticM = time.time()
            filename_read = '%s/output/ifs/%03i/ICMGG%s+%i%02i' %(rundir, year-2004, run, year, mon)
#             filename_read = '%s/output/ifs/%03i/ICMGG%s+%i%02i' %(rundir, year-2004, run, year, mon+1)
            tic = time.time()
#             grbs = pygrib.open(filename_read)
#             grbs.seek(0)
#             ll = 0
#             for grb in grbs:
#                 ll = ll + 1
#                 if ll==1:
#                     continue
#                  
#                 if grb.day < 2:
#                     continue
#                 if grb.level > 500:
#                     continue
# #                 if grb.time > 300:
# #                     continue
#                 if grb.hour != 12:
#                     continue
# #                 if grb.paramId == 0:
# #                     pdb.set_trace()
# #                     
# #                 
#                 print(grb)
#                 print(grb.paramId)
# #             
# #             print('f')
#             pdb.set_trace()
#             sys.exit()
#             grbindx = pygrib.index(filename_read,'paramId')
            grbindx = pygrib.index(filename_read,'indicatorOfParameter')
            toc = time.time()
#             grbindx_name = pygrib.index(filename_read,'name')
            print('time = %i s' %(toc-tic))
#             for pid in [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 246, 247, 248, 54, 130, 208, 209, 210, 211, 212, 164, 169,175,176,177,178,179]: # all
            for pid in [107, 108, 109, 110, 111, 112, 246, 247, 248, 54, 130, 208, 209, 210, 211, 212, 164, 169,175,176,177,178,179, 133]: #24st 23*12=288 antal filer
                print('id = %i' %pid)
                filename_save = '%s/ICMGG%s+%i%02i_%i.h5' %(savedir, run, year, mon, pid)
                print(filename_save)
                if os.path.isfile(filename_save):
                    continue
                tic = time.time()
#                 params = grbindx.select(paramId=pid)
                params = grbindx.select(indicatorOfParameter=pid)
                toc = time.time()
                print('time = %i s' %(toc-tic))
#                 tic = time.time()
#                 t101 = grbs1(paramId=101)
#                 toc = time.time()
#                 print('time = %i s' %(toc-tic))
                h5file = h5py.File(filename_save, 'w')
                i = -1
                tic = time.time()
                for para in params:
                    i = i + 1
                    if 'missingValue' not in h5file.keys():
                        missingVal = para.missingValue
                        h5file.create_dataset('missingValue', data = missingVal)
                        if (mon == 1) and (pid == 107):
                            #: Only need to save this ones since it is same for all
                            filename_save_latlon = '%s/latlon_%s.h5' %(savedir, run)
                            h5file_latlon = h5py.File(filename_save_latlon, 'w')
                            lat, lon = para.latlons()
                            h5file_latlon.create_dataset('latitude', data = lat)
                            h5file_latlon.create_dataset('longitude', data = lon)
                            h5file_latlon.close()
                        latlon = False
#                     if para.analDate.hour in [0,3,6,18,21]:
#                         continue
                    h5file.create_dataset('%s/%i' %(para.analDate.strftime("%Y-%m-%d %H:%M:%S"), para.level), data = para.values)
                toc = time.time()
                print('time = %i s' %(toc-tic))
                h5file.close()
            tocM = time.time()
            print('Month TIME %i' %(tocM-ticM))
            
            
            
            
            
            
            
            
            
            
            