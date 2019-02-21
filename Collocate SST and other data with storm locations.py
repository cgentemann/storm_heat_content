
# coding: utf-8

# In[16]:


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
sys.path.append('C:/Users/gentemann/Google Drive/d_drive/python/storm_heat_content/subroutines/')
from storm_masking_routines import interpolate_storm_path
from storm_masking_routines import get_dist_grid
from storm_masking_routines import closest_dist
from storm_masking_routines import calculate_storm_mask


# In[ ]:


input_year=int(str(sys.argv[1]))
print ('processing year:', input_year)


# In[ ]:


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

#        if iyr_storm!=2007: # or iyr_storm<2003:
#            continue
        print(name,filename)
        ds_storm_info = xr.open_dataset(filename)
        lats = ds_storm_info.lat[0,:]
        lons = ds_storm_info.lon[0,:]  #lons goes from 0 to 360
        lons = (lons + 180) % 360 - 180 #put -180 to 180
        dysince = ds_storm_info.time
        ds_storm_info.close()
        
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
        max_date = max(tem_date)+dt.timedelta(days=5)
        minjdy = min_date.timetuple().tm_yday  #create new time array that can be queried for year etc
        minyear =min_date.year #create new time array that can be queried for year etc
        minmon =min_date.month #create new time array that can be queried for year etc
        minday =min_date.day #create new time array that can be queried for year etc
        maxjdy = max_date.timetuple().tm_yday  #create new time array that can be queried for year etc
        maxyear =max_date.year  #create new time array that can be queried for year etc
        print(minyear,minjdy,maxyear,maxjdy)
        
        dif = max(tem_date)-min(tem_date)
        tdim=int(dif.days)+45             #calculate ssts for 30 days after storm
        
        #print(tdim,xdim,ydim)            
        
        #print('sst_out_sv',sst_out_sv.shape)
        for i in range(0,tdim):
            storm_date = dt.datetime(minyear,minmon,minday)+dt.timedelta(days=i)+dt.timedelta(hours=12)
            #print(storm_date)
            
            syr=str(storm_date.year)
            smon=str(storm_date.month)
            sdym=str(storm_date.day)
            sjdy=str(storm_date.timetuple().tm_yday)

#sst data   
            fname_tem=syr + smon.zfill(2) + sdym.zfill(2) + '120000-CMC-L4_GHRSST-SSTfnd-CMC0.2deg-GLOB-v02.0-fv02.0.nc'
            filename = dir_cmc + syr + '/' + sjdy.zfill(3) + '/' + fname_tem
            ds_day=xr.open_dataset(filename,drop_variables=['analysis_error','sea_ice_fraction'])
            if iwrap==1:  #data is -180 to 180 for sst, so need to bring to 0 to 360 when wrapped
                ds_day.coords['lon'] = np.mod(ds_day['lon'], 360)
                ds_day = ds_day.sortby(ds_day.lon)
            ds_day.close()
            ds_day = ds_day.where(ds_day['mask'] == 1.) 
            ds_storm = ds_day.interp(lat = new_lat_storm,lon = new_lon_storm)
            #ds_storm['time']=storm_date
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            if i==0:
                ds_storm_sst = ds_storm
            else:
                ds_storm_sst = xr.concat([ds_storm_sst,ds_storm],dim='time')

#sst climatology  --- this isn't used, should remove from dataset in next round
#            if storm_date.timetuple().tm_yday==366:
#                sjdy = '365'
#            filename='F:/data/sst/cmc/CMC0.2deg/v2/climatology/clim1993_2016' + sjdy.zfill(3) + '-CMC-L4_GHRSST-SSTfnd-CMC0.2deg-GLOB-v02.0-fv02.0.nc'
#            ds_day=xr.open_dataset(filename,drop_variables=['analysis_error','sea_ice_fraction','sq_sst'])
#            ds_day = ds_day.rename({'analysed_sst':'analysed_sst_clim','mask':'mask_clim'}) #, inplace = True)            
#            if iwrap==1:  #data is -180 to 180 for sst, so need to bring to 0 to 360 when wrapped
#                ds_day.coords['lon'] = np.mod(ds_day['lon'], 360)
#                ds_day = ds_day.sortby(ds_day.lon)
#            ds_day.close()
#            ds_day = ds_day.where(ds_day['mask_clim'] == 1.) 
#            ds_storm = ds_day.interp(lat = new_lat_storm,lon = new_lon_storm)
#            ds_storm = ds_storm.assign_coords(time=storm_date)
#            if iwrap==1:
#                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
#            if i==0:
#                ds_storm_sst_clim = ds_storm
#            else:
#                ds_storm_sst_clim = xr.concat([ds_storm_sst_clim,ds_storm],dim='time')           
            
#ccmp wind data, no masked data, a complete field
#            lyr, idyjl = 2015,1
#            storm_date = dt.datetime(2015,1,1)
            syr, smon, sdym, sjdy=str(storm_date.year),str(storm_date.month),str(storm_date.day),str(storm_date.timetuple().tm_yday)
            fname_tem='/CCMP_Wind_Analysis_' + syr + smon.zfill(2) + sdym.zfill(2) + '_V02.0_L3.0_RSS.nc'
            ccmp_filename = dir_ccmp + syr + '/M' + smon.zfill(2) + fname_tem      
            ds=xr.open_dataset(ccmp_filename,drop_variables=['nobs'])
            ds_day = ds.mean(dim='time')     #take average across all 6 hourly data fields
            ds_day = ds_day.rename({'longitude':'lon','latitude':'lat'}) #, inplace = True)            
            if iwrap==0:
                ds_day.coords['lon'] = (ds_day.coords['lon'] + 180) % 360 - 180
                ds_day = ds_day.sortby(ds_day.lon)
            ds.close()
            ds_storm = ds_day.interp(lat = new_lat_storm,lon = new_lon_storm)
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            ds_storm = ds_storm.assign_coords(time=storm_date)
            if i==0:
                ds_storm_ccmp = ds_storm
            else:
                ds_storm_ccmp = xr.concat([ds_storm_ccmp,ds_storm],dim='time')
              
#ocean mixed layer depth from monthly data GODAS NOAA, lon 0 to 360, monthly data so interp to day
            #dir_godas='https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/godas/'
            dir_godas = 'f:/data/model_data/godas/'
            if isave_mld_year != storm_date.year:
                filename = dir_godas + 'dbss_obml.' + syr + '.nc'
                ds_day_mld=xr.open_dataset(filename)
                if iwrap==0:
                    ds_day_mld.coords['lon'] = (ds_day_mld.coords['lon'] + 180) % 360 - 180
                    ds_day_mld = ds_day_mld.sortby(ds_day_mld.lon)
                ds_day_mld.close()
                isave_mld_year = storm_date.year
            ds_storm = ds_day_mld.interp(time = storm_date, lat = new_lat_storm,lon = new_lon_storm)
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            ds_storm = ds_storm.assign_coords(time=storm_date)
            if i==0:
                ds_storm_mld = ds_storm
            else:
                ds_storm_mld = xr.concat([ds_storm_mld,ds_storm],dim='time')            
            
#latent heat flux data, masked already set to NaN                
            filename = dir_flux + 'lh_oaflux_' + syr + '.nc';
            ds=xr.open_dataset(filename,drop_variables=['err'])
            ds_day = ds.sel(time = storm_date.timetuple().tm_yday)  #select day of year from annual file
            if iwrap==0:
                ds_day.coords['lon'] = (ds_day.coords['lon'] + 180) % 360 - 180
                ds_day = ds_day.sortby(ds_day.lon)
            ds.close()
            ds_storm = ds_day.interp(lat = new_lat_storm,lon = new_lon_storm)
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            ds_storm['time']=storm_date
            if i==0:
                ds_storm_lhf = ds_storm
            else:
                ds_storm_lhf = xr.concat([ds_storm_lhf,ds_storm],dim='time')

#sensible heat flux data , masked already set to NaN                
            filename = dir_flux + 'sh_oaflux_' + syr + '.nc';
            ds=xr.open_dataset(filename,drop_variables=['err'])
            ds_day = ds.sel(time = storm_date.timetuple().tm_yday)  #select day of year from annual file
            if iwrap==0:
                ds_day.coords['lon'] = (ds_day.coords['lon'] + 180) % 360 - 180
                ds_day = ds_day.sortby(ds_day.lon)
            ds.close()
            ds_storm = ds_day.interp(lat = new_lat_storm,lon = new_lon_storm)
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            ds_storm['time']=storm_date
            if i==0:
                ds_storm_shf = ds_storm
            else:
                ds_storm_shf = xr.concat([ds_storm_shf,ds_storm],dim='time')

#surface humid flux data   , masked already set to NaN              
            filename = dir_flux + 'qa_oaflux_' + syr + '.nc';
            ds=xr.open_dataset(filename,drop_variables=['err'])
            ds_day = ds.sel(time = storm_date.timetuple().tm_yday)  #select day of year from annual file
            if iwrap==0:
                ds_day.coords['lon'] = (ds_day.coords['lon'] + 180) % 360 - 180
                ds_day = ds_day.sortby(ds_day.lon)
            ds.close()
            ds_storm = ds_day.interp(lat = new_lat_storm,lon = new_lon_storm)
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            ds_storm['time']=storm_date
            if i==0:
                ds_storm_qa = ds_storm
            else:
                ds_storm_qa = xr.concat([ds_storm_qa,ds_storm],dim='time')

#air temp flux data   , masked already set to NaN              
            filename = dir_flux + 'ta_oaflux_' + syr + '.nc';
            ds=xr.open_dataset(filename,drop_variables=['err'])
            ds_day = ds.sel(time = storm_date.timetuple().tm_yday)  #select day of year from annual file
            if iwrap==0:
                ds_day.coords['lon'] = (ds_day.coords['lon'] + 180) % 360 - 180
                ds_day = ds_day.sortby(ds_day.lon)
            ds.close()
            ds_storm = ds_day.interp(lat = new_lat_storm,lon = new_lon_storm)
            if iwrap==1:
                ds_storm.coords['lon'] = (ds_storm.coords['lon'] + 180) % 360 - 180
            ds_storm['time']=storm_date
            if i==0:
                ds_storm_ta = ds_storm
            else:
                ds_storm_ta = xr.concat([ds_storm_ta,ds_storm],dim='time')
                
#        ds_all = xr.merge([ds_storm_ccmp, ds_storm_mld, ds_storm_lhf, ds_storm_shf, ds_storm_ta, ds_storm_qa, ds_storm_sst, ds_storm_sst_clim])
        ds_all = xr.merge([ds_storm_ccmp, ds_storm_mld, ds_storm_lhf, ds_storm_shf, ds_storm_ta, ds_storm_qa, ds_storm_sst])

        #calculate mask
#        print('caluculating mask')
#        ds_mask = calculate_storm_mask(ds_all,lats,lons)
#        ds_all['storm_mask']=ds_mask['storm_mask']
#        #dist to storm
#        print('calculating dist')
#        dist,index,stime,position = closest_dist(ds_all,ds_storm_info)
#        dtem=xr.DataArray(dist, coords={'lat': ds_mask.lat.values, 'lon':ds_mask.lon.values}, dims=('lat', 'lon'))
#        ds_all['dist_from_storm_km']=dtem
#        dtem=xr.DataArray(index, coords={'lat': ds_mask.lat.values, 'lon':ds_mask.lon.values}, dims=('lat', 'lon'))
#        ds_all['closest_storm_index']=dtem
#        dtem=xr.DataArray(stime, coords={'lat': ds_mask.lat.values, 'lon':ds_mask.lon.values}, dims=('lat', 'lon'))
#        ds_all['closest_storm_time']=dtem
#        dtem=xr.DataArray(position, coords={'lat': ds_mask.lat.values, 'lon':ds_mask.lon.values}, dims=('lat', 'lon'))
#        ds_all['side_of_storm']=dtem

        
        if iwrap==1:
            ds_all.coords['lon'] = np.mod(ds_all['lon'], 360)
            ds_storm_info['lon'] = np.mod(ds_storm_info['lon'], 360)

#        filename = dir_out + str(iyr_storm) + '/' + str(inum_storm).zfill(3) + '_combined_data.nc'
#        ds_all = xr.open_dataset(filename)
#        ds_all.close()
                
            
        #calculate mask
        print('caluculating mask')
        ds_mask = calculate_storm_mask(ds_all,lats,lons)
        ds_all['storm_mask']=ds_mask['storm_mask']
        #dist to storm
        print('calculating dist')
        dist,index,stime,position,ds_storm_interp = closest_dist(ds_all,ds_storm_info)
        dtem=xr.DataArray(dist, coords={'lat': ds_mask.lat.values, 'lon':ds_mask.lon.values}, dims=('lat', 'lon'))
        ds_all['dist_from_storm_km']=dtem
        dtem=xr.DataArray(index, coords={'lat': ds_mask.lat.values, 'lon':ds_mask.lon.values}, dims=('lat', 'lon'))
        ds_all['closest_storm_index']=dtem
        dtem=xr.DataArray(stime, coords={'lat': ds_mask.lat.values, 'lon':ds_mask.lon.values}, dims=('lat', 'lon'))
        ds_all['closest_storm_time']=dtem
        dtem=xr.DataArray(position, coords={'lat': ds_mask.lat.values, 'lon':ds_mask.lon.values}, dims=('lat', 'lon'))
        ds_all['side_of_storm']=dtem

        #find max sst 5 days before storm location
        #first create an array with the storm crossover time (from nearest point) as an array        
        #now use array of storm time to calculate prestorm sst
        ydim,xdim=ds_all.lat.shape[0], ds_all.lon.shape[0]
#        print(ydim,xdim)
#        print(ds_all)
        sdate = np.empty([ydim,xdim], dtype=dt.datetime)   

        for i in range(0,xdim):
            for j in range(0,ydim):
                tem=date_1858+dt.timedelta(days=float(ds_all.closest_storm_time[j,i])) 
                sdate[j,i]=np.datetime64(tem)
        xsdate=xr.DataArray(sdate, coords={'lat': ds_all.lat.values, 'lon':ds_all.lon.values}, dims=('lat', 'lon'))        
        sst0 = ds_all.dist_from_storm_km.copy(deep=True)
        for i in range(0,xdim):
            for j in range(0,ydim):
                sst0[j,i] = ds_all.analysed_sst[:,j,i].sel(time=slice(xsdate[j,i]-np.timedelta64(5,'D'),xsdate[j,i])).max()
        ds_all['sst_prestorm']=sst0

        
        filename = dir_out + str(iyr_storm) + '/' + str(inum_storm).zfill(3) + '_combined_data.nc'
        ds_all.to_netcdf(filename)
        print('out:',filename)
     # filename = dir_out + str(iyr_storm) + '/' + 'str(inum_storm)' + '_other_data.nc'
     #   filename = dir_out + str(iyr_storm) + '/' + str(inum_storm).zfill(3) + '_combined_masking_data.nc'
     #   ds_all.to_netcdf(filename)
     #   print('out:',filename)
        filename = dir_out + str(iyr_storm) + '/' + str(inum_storm).zfill(3) + '_interpolated_track.nc'
        ds_storm_interp.to_netcdf(filename)
        print('out:',filename)
    


