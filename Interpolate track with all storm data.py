import os
import xarray as xr
import numpy as np
import geopy.distance
from math import sin, pi
import datetime as dt


date_1858 = dt.datetime(1858,11,17,0,0,0) # start date is 11/17/1958

#still processing 2002, 2011 - onwards
for iyr_storm in range(2009,2010):
    for inum_storm in range(28,100):
        dir_out='f:/data/tc_wakes/database/sst/'
        filename = dir_out + str(iyr_storm) + '/' + str(inum_storm).zfill(3) + '_interpolated_track.nc'
        exists = os.path.isfile(filename)
        if exists:
            print(filename)
            ds_storm_info=xr.open_dataset(filename)
            ds_storm_info = ds_storm_info.sel(j2=0)
            ds_storm_info.close()

#add storm translation speed to storm information
            tdim_storm = ds_storm_info.time.size
            storm_speed = ds_storm_info.time.copy(deep=True)*np.nan    
            for i in range(0,tdim_storm-1):
                coords_1 = (ds_storm_info.lat[i], ds_storm_info.lon[i])  
                coords_2 = (ds_storm_info.lat[i+1], ds_storm_info.lon[i+1])  
                arclen_temp = geopy.distance.geodesic(coords_1, coords_2).km  #distance in km  
                storm_date1 = np.datetime64(date_1858 + dt.timedelta(days=float(ds_storm_info.time[i])))  
                storm_date2 = np.datetime64(date_1858 + dt.timedelta(days=float(ds_storm_info.time[i+1])))  
                arclen_time = storm_date2 - storm_date1
                arclen_hr = arclen_time / np.timedelta64(1, 'h')
                storm_speed[i]=arclen_temp/(arclen_hr)
            storm_speed[-1]=storm_speed[-2]
            ds_storm_info['storm_speed']=storm_speed
            
#add storm info to combined grid
            filename = dir_out + str(iyr_storm) + '/' + str(inum_storm).zfill(3) + '_combined_data.nc'
            ds_all = xr.open_dataset(filename)
            ds_all.close()
            xdim,ydim,tdim = ds_all.lon.shape[0],ds_all.lat.shape[0],ds_all.time.shape[0]
            wtem=np.empty([ydim,xdim])
            ptem=np.empty([ydim,xdim])
            stem=np.empty([ydim,xdim])
            for i in range(0,xdim):
                for j in range(0,ydim):
                    storm_index = ds_all.closest_storm_index[j,i].data
                    wtem[j,i]=ds_storm_info.wind[int(storm_index)].data
                    ptem[j,i]=ds_storm_info.pres[int(storm_index)].data
                    stem[j,i]=ds_storm_info.storm_speed[int(storm_index)].data
            xrtem=xr.DataArray(wtem, coords={'lat': ds_all.lat.values, 'lon':ds_all.lon.values}, dims=('lat', 'lon'))        
            ds_all['wmo_storm_wind']=xrtem
            xrtem=xr.DataArray(ptem, coords={'lat': ds_all.lat.values, 'lon':ds_all.lon.values}, dims=('lat', 'lon'))        
            ds_all['wmo_storm_pres']=xrtem
            xrtem=xr.DataArray(stem, coords={'lat': ds_all.lat.values, 'lon':ds_all.lon.values}, dims=('lat', 'lon'))        
            ds_all['wmo_storm_speed']=xrtem

#make subset by masking data
#subset now only has the data within 100 and 800 km of storm
            if abs(ds_all.lon[-1]-ds_all.lon[0])>180:
                ds_all.coords['lon'] = np.mod(ds_all['lon'], 360)
                ds_storm_info['lon'] = np.mod(ds_storm_info['lon'], 360)
            max_lat = ds_storm_info.lat.max()
            if max_lat<0:
                cond = ((ds_all.dist_from_storm_km<100) & (ds_all.side_of_storm<=0)) |  ((ds_all.dist_from_storm_km<800) & (ds_all.side_of_storm>0))
            else:
                cond = ((ds_all.dist_from_storm_km<800) & (ds_all.side_of_storm<0)) |  ((ds_all.dist_from_storm_km<100) & (ds_all.side_of_storm>=0))
            subset = ds_all.where(cond)

#now calculate coldwake information
          
            xdim,ydim,tdim = ds_all.lon.shape[0],ds_all.lat.shape[0],ds_all.time.shape[0]
            date_1858 = dt.datetime(1858,11,17,0,0,0) # start date is 11/17/1958
            coldwake_max=ds_all.sst_prestorm.copy(deep=True)*np.nan
            coldwake_maxindex=ds_all.sst_prestorm.copy(deep=True)*np.nan
            coldwake_hrtomaxcold=ds_all.sst_prestorm.copy(deep=True)*np.nan
            coldwake_recovery=ds_all.sst_prestorm.copy(deep=True)*np.nan
#go through entire array lat/lon dims
            for i in range(0,xdim):
                for j in range(0,ydim):
                     #calculate the storm time for the closest collocated storm point then find the combined data index for closest time
                    #this gives you the combined data storm index cross over
                    #storm index is the combined data index for the strom cross over
                    storm_date = date_1858 + dt.timedelta(days=float(ds_all.closest_storm_time[j,i]))  
                    storm_date64 = np.datetime64(storm_date)
                    if np.isnan(subset.analysed_sst[0,j,i]):  #don't process masked values
                        continue
                    time_diff = subset.time-storm_date64
                    storm_index = np.argmin(abs(time_diff)).data
                    #now look for cold wake for 1 day before strom to 5 days after strom
                    #caluclate hours to cold wake, maximum cold wake, hours until it returns to prestorm sst
                    #there is NO filter on wheither coldwake large enough here, just does all points
                    istart,iend = int(storm_index)-1,int(storm_index)+5
                    if istart<0:
                        istart=0
                    if iend>tdim:
                        iend=tdim
                    if np.isnan(subset.sst_prestorm[j,i]):
                        continue
                    coldwake_max[j,i] = (subset.sst_prestorm[j,i]-subset.analysed_sst[istart:iend,j,i]).min()
                    itmp = np.argmin(subset.sst_prestorm[j,i]-subset.analysed_sst[istart:iend,j,i]).data
                    coldwake_maxindex[j,i]=istart+itmp
                    delay = subset.time[istart+itmp].values-subset.time[istart+1].values
                    coldwake_hrtomaxcold[j,i]=delay / np.timedelta64(1, 'h')
                    for k in range(istart+itmp,tdim):
                        sst_change = subset.sst_prestorm[j,i]-subset.analysed_sst[k,j,i]
                        if sst_change>-0.2:
                            break
                    delay = subset.time[k].values-subset.time[istart+1].values
                    coldwake_recovery[j,i]=delay / np.timedelta64(1, 'h')

            ds_all['coldwake_max']=coldwake_max
            ds_all['coldwake_maxindex']=coldwake_maxindex
            ds_all['coldwake_hrtomaxcold']=coldwake_hrtomaxcold
            ds_all['coldwake_hrtorecovery']=coldwake_recovery
            print('out:',filename)
            filename = dir_out + str(iyr_storm) + '/' + str(inum_storm).zfill(3) + '_combined_data_all.nc'
            ds_all.to_netcdf(filename)
