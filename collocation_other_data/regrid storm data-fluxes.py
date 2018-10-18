from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import os
import datetime as dt
import xarray as xr
from datetime import datetime
import pandas

#################################################################################
dir_in='f:/data/tc_wakes/database/info/'
dir_out='f:/data/tc_wakes/database/sst/'
####################you will need to change some paths here!#####################
#list of input files
dir_in='f:/data/tc_wakes/database/info/'
dir_out='f:/data/tc_wakes/database/sat_data/'
dir_sat_dat = 'F:/data/model_data/gmao/flux/'
#################################################################################

#read in target lat/lon (ccmp array)
ccmp_filename='F:/data/sat_data/ccmp/v02.0/Y2003/M10/CCMP_Wind_Analysis_20031009_V02.0_L3.0_RSS.nc'
ds_target=xr.open_dataset(ccmp_filename)  #don't need nobs
ds_target_lon=ds_target.longitude
ds_target_lat=ds_target.latitude
date_1858 = dt.datetime(1858,11,17,0,0,0) # start date is 11/17/1958
dx=0.25
dy=0.25
dx_offset = -179.875
dy_offset = 78.3750
for root, dirs, files in os.walk(dir_in, topdown=False):
    #print(files)
    #for ii in range(40,90): 
    #print(root[31:35])
    for name in files:
        istart_storm=0
        #name = files[ii]
        fname_in=os.path.join(root, name)
        print(name,fname_in)
        year_storm=int(root[31:35])
        num_storm=int(name[0:3])
        if year_storm<2004:
            break
        if num_storm<=76 and year_storm==2004:
            break
        dsx = xr.open_dataset(fname_in)
        lats = dsx.lat[0,:]
        lons = dsx.lon[0,:]
        dysince = dsx.time
        icycle=0
        minlon=min(lons.values)-10
        maxlon=max(lons.values)+10
        minlat=min(lats.values)-10
        maxlat=max(lats.values)+10
        if minlon<-170 and maxlon>170.:  #wrapping around meridion need to cal new min/max lon
            minlon=max(lons[lons>100].values)-10
            maxlon=min(lons[lons<-100].values)+10
            icycle=1 #set flag for wraparound
        print(icycle,minlon,maxlon,minlat,maxlat)
        dims=lats.shape
        print(dims)
        tdim=dims[0]
        tem_date=[0]*tdim #print(dysince.values)
        for i in range(0,tdim):
            tem_date[i]=date_1858+dt.timedelta(days=float(dysince[0,i].values))  #create new time array that can be queried for year etc
        minjdy = min(tem_date).timetuple().tm_yday  #create new time array that can be queried for year etc
        minyear =min(tem_date).year #create new time array that can be queried for year etc
        maxjdy = max(tem_date).timetuple().tm_yday  #create new time array that can be queried for year etc
        maxyear =max(tem_date).year  #create new time array that can be queried for year etc
        dif = max(tem_date)-min(tem_date)
        tdim=int(dif.days)  
        print(minjdy,maxjdy,minlon,maxlon,minlat,maxlat)
        ds_new=[]
        new_coord=[]
        for incr_dy in range(-10,tdim+30): #from -10 to 30 days after
            storm_date = tem_date[0]+dt.timedelta(days=incr_dy)            
            #print(storm_date)
            syr=str(storm_date.year)
            smon=str(storm_date.month)
            sdym=str(storm_date.day)
            sjdy=str(storm_date.timetuple().tm_yday)
            if storm_date.year<2011:
                fname_tem='MERRA2_300.tavg1_2d_flx_Nx.'+syr + smon.zfill(2) + sdym.zfill(2)+'.SUB.nc4'
            else:
                fname_tem='MERRA2_400.tavg1_2d_flx_Nx.'+syr + smon.zfill(2) + sdym.zfill(2)+'.SUB.nc4'                
            mur_filename = dir_sat_dat + fname_tem
            print(mur_filename)
            ds=xr.open_dataset(mur_filename,drop_variables=('NIRDF','NIRDR'))  #don't need nobs
            #interpolate 1 km mur data onto CCMP grid
            #select lat/lon region
            if icycle==0:
                ds_interp = ds.interp(lat=ds_target_lat,lon=ds_target_lon)
                ds2=ds_interp.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
            else:
                ds_rolled = ds.assign_coords(lon=(ds.lon % 360)).roll(lon=(ds.dims['lon'] // 2))
                ds_interp = ds.interp(lat=ds_target_lat,lon=ds_target_lon)
                ds2=ds_interp.sel(latitude=slice(minlat,maxlat),longitude=(sst_interp.longitude < minlon) | (sst_interp.longitude > maxlon))
            ds_new.append(ds2)
            new_coord.append(storm_date)
        combined = xr.concat(ds_new, dim='time')
        combined.coords['time'] = new_coord
        fname_out=dir_out + 'gmao' + root[31:35] + name[0:3] + '.nc'
        print(fname_out)
        combined.to_netcdf(fname_out)

