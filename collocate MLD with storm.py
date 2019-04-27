#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import os
import time
import datetime as dt
import xarray as xr
from datetime import datetime
import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
import cartopy.crs as ccrs
dir_storm_wmo='F:/data/tc_wakes/ibtracks/year/'

####################you will need to change some paths here!#####################
#list of input directories
dir_storm_info='f:/data/tc_wakes/database/info/'
dir_out='f:/data/tc_wakes/database/sst/'
dir_flux = 'F:/data/model_data/oaflux/data_v3/daily/turbulence/'
dir_cmc = 'F:/data/sst/cmc/CMC0.2deg/v2/'
dir_ccmp='F:/data/sat_data/ccmp/v02.0/Y'
##where to get the data through opendap, use these directories instead
#dir_cmc = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/CMC/CMC0.1deg/v3/'
#dir_flux = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/WHOI_OAFlux/version3/daily/lh_oaflux/'
#the latest ccmp is from www.remss.com but they do not have an opendap server so you can use this instead:
#dir_ccmp='https://podaac-opendap.jpl.nasa.gov/opendap/allData/ccmp/L3.0/flk/'

#################################################################################
import geopy.distance
from math import sin, pi
from scipy import interpolate

#functions for running storm data
import sys
sys.path.append('./subroutines/')
from storm_masking_routines import interpolate_storm_path
from storm_masking_routines import get_dist_grid
from storm_masking_routines import closest_dist
from storm_masking_routines import calculate_storm_mask


# In[ ]:


input_year=int(str(sys.argv[1]))
print ('processing year:', input_year)


# In[28]:


#this is a special version of the code above that just re-caluclates the MLD
### SPECIAL VERSION ####
date_1858 = dt.datetime(1858,11,17,0,0,0) # start date is 11/17/1958
isave_mld_year = 0 #init MLD monthly data read flag
for root, dirs, files in os.walk(dir_storm_info, topdown=False):
    if root[len(dir_storm_info):len(dir_storm_info)+1]=='.':
        continue
    for name in files:
        if not name.endswith('.nc'):
            continue
        filename=os.path.join(root, name)
        print(filename[36:39],filename[31:35])
        inum_storm=int(filename[36:39])
        iyr_storm=int(filename[31:35])

        if iyr_storm!=input_year:
            continue
#        if input_storm!=inum_storm:
#            continue

#        if iyr_storm!=2007: # or iyr_storm<2003:
#            continue
        print(name,filename)
        ds_storm_info = xr.open_dataset(filename)
        lats = ds_storm_info.lat[0,:]
        lons = ds_storm_info.lon[0,:]  #lons goes from 0 to 360
        lons = (lons + 180) % 360 - 180 #put -180 to 180
        dysince = ds_storm_info.time
        ds_storm_info.close()
#        print(ds_storm_info)
#        break
#        ds_storm_interp = interpolate_storm_path(ds_storm_info)
#        print(ds_storm_interp)
#        break

#make lat and lon of storm onto 25 km grid for below
        lons = (((lons - .125)/.25+1).astype(int)-1)*.25+.125
        lats = (((lats + 89.875)/.25+1).astype(int)-1)*.25-89.875
        
        iwrap=0
#calculate size of box to get data in
        minlon,maxlon = min(lons.values)-10, max(lons.values)+10
        minlat,maxlat = min(lats.values)-10, max(lats.values)+10

        ydim_storm = round((maxlat - minlat)/.25).astype(int)
        new_lat_storm = np.linspace(minlat, maxlat, ydim_storm)
        if (minlon<-90 and maxlon>=90) or (minlon<-180 and maxlon<0):  #this storm wraps  keep everythig 0 to 360 then wrap data at very end
            iwrap = 1
            lons2 = np.mod(lons, 360)
            minlon, maxlon = min(lons2.values)-10, max(lons2.values)+10
            xdim_storm = round((maxlon - minlon)/.25).astype(int)
            new_lon_storm = np.linspace(minlon, maxlon, xdim_storm)
        else:
            xdim_storm = round((maxlon - minlon)/.25).astype(int)
            new_lon_storm = np.linspace(minlon, maxlon, xdim_storm)

        print(iwrap,minlon,maxlon)
        print(iwrap,xdim_storm, new_lon_storm[:5],new_lon_storm[-5:])

        
        dims=lats.shape
        tdim=dims[0]
        tem_date=[0]*tdim #print(dysince.values)
        for i in range(0,tdim):
            tem_date[i]=date_1858+dt.timedelta(days=float(dysince[0,i].values))  #create new time array that can be queried for year etc
        min_date = min(tem_date)+dt.timedelta(days=-5)
#        max_date = max(tem_date)+dt.timedelta(days=5)
        minjdy = min_date.timetuple().tm_yday  #create new time array that can be queried for year etc
        minyear =min_date.year #create new time array that can be queried for year etc
        minmon =min_date.month #create new time array that can be queried for year etc
        minday =min_date.day #create new time array that can be queried for year etc
#        maxjdy = max_date.timetuple().tm_yday  #create new time array that can be queried for year etc
#        maxyear =max_date.year  #create new time array that can be queried for year etc
        print(minyear,minjdy)#,maxyear,maxjdy)
        
        dif = max(tem_date)-min(tem_date)
        tdim=int(dif.days)+45             #calculate ssts for 30 days after storm
        
        max_date = dt.datetime(minyear,minmon,minday)+dt.timedelta(days=tdim)+dt.timedelta(hours=12)

        #print(tdim,xdim,ydim)            

        #special read in of ECCO data and subset in time 
        filename = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/ECCO2/cube92/mxldepth'
        ds_ecco = xr.open_dataset(filename)
        ds_ecco.close()
        ds_ecco = ds_ecco.sel(time=slice(min_date,max_date))
        if iwrap==0:
            ds_ecco.coords['lon'] = (ds_ecco.coords['lon'] + 180) % 360 - 180
            ds_ecco = ds_ecco.sortby(ds_ecco.lon)
        
        #print('sst_out_sv',sst_out_sv.shape)
        for i in range(0,tdim):
            storm_date = dt.datetime(minyear,minmon,minday)+dt.timedelta(days=i)+dt.timedelta(hours=12)
            #print(storm_date)
            
            syr=str(storm_date.year)
            smon=str(storm_date.month)
            sdym=str(storm_date.day)
            sjdy=str(storm_date.timetuple().tm_yday)

             
#ocean mixed layer depth from monthly data GODAS NOAA, lon 0 to 360, monthly data so interp to day
#this is monthly data (all other data daily) so need to read in year before and year after
#so any storms <1/15 or greater than 12/15 in the year still get data
#dir_godas='https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/godas/'
            dir_godas = 'f:/data/model_data/godas/'
            if isave_mld_year != storm_date.year:
                filename = dir_godas + 'dbss_obml.' + str(storm_date.year-1) + '.nc'
                ds_day_mld=xr.open_dataset(filename)
                ds_day_mld['time']=ds_day_mld.time+np.timedelta64(14,'D')  #data provider gave 1st day of ave in time 
                ds_day_mld.close()
                filename = dir_godas + 'dbss_obml.' + str(storm_date.year) + '.nc'
                ds_day_mld2=xr.open_dataset(filename)
                ds_day_mld2['time']=ds_day_mld2.time+np.timedelta64(14,'D')  #data provider gave 1st day of ave in time 
                ds_day_mld2.close()
                ds_day_mld = xr.concat([ds_day_mld,ds_day_mld2],dim='time')
                filename = dir_godas + 'dbss_obml.' + str(storm_date.year+1) + '.nc'
                ds_day_mld2=xr.open_dataset(filename)
                ds_day_mld2['time']=ds_day_mld2.time+np.timedelta64(14,'D')  #data provider gave 1st day of ave in time 
                ds_day_mld2.close()
                ds_day_mld = xr.concat([ds_day_mld,ds_day_mld2],dim='time')
                if iwrap==0:
                    ds_day_mld.coords['lon'] = (ds_day_mld.coords['lon'] + 180) % 360 - 180
                    ds_day_mld = ds_day_mld.sortby(ds_day_mld.lon)
                isave_mld_year = storm_date.year
            if i==0:
                tem_storm_date = dt.datetime(minyear,minmon,minday)+dt.timedelta(days=-30)
                ds_storm = ds_day_mld.interp(time = tem_storm_date, lat = new_lat_storm,lon = new_lon_storm)
            else:
                ds_storm = ds_day_mld.interp(time = storm_date, lat = new_lat_storm,lon = new_lon_storm)
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            ds_storm = ds_storm.assign_coords(time=storm_date)
            if i==0:
                ds_storm_mld = ds_storm
            else:
                ds_storm_mld = xr.concat([ds_storm_mld,ds_storm],dim='time')            

            ds_storm = ds_ecco.interp(time = storm_date, lat = new_lat_storm,lon = new_lon_storm)
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            ds_storm = ds_storm.assign_coords(time=storm_date)
            if i==0:
                ds_storm_mld_ecco = ds_storm
            else:
                ds_storm_mld_ecco = xr.concat([ds_storm_mld_ecco,ds_storm],dim='time')            
                
#        ds_all = xr.merge([ds_storm_ccmp, ds_storm_mld, ds_storm_lhf, ds_storm_shf, ds_storm_ta, ds_storm_qa, ds_storm_sst, ds_storm_sst_clim])
        ds_all = xr.merge([ds_storm_mld,ds_storm_mld_ecco])

        if iwrap==1:
            ds_all.coords['lon'] = np.mod(ds_all['lon'], 360)
            ds_storm_info['lon'] = np.mod(ds_storm_info['lon'], 360)     
         
        filename = dir_out + str(iyr_storm) + '/' + str(inum_storm).zfill(3) + '_MLD_data_v2.nc'
        ds_all.to_netcdf(filename)
        print('out:',filename)
   

