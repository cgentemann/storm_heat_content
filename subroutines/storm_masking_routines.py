
# coding: utf-8

# In[ ]:


#functions for running storm data
#functions for running storm data
def interpolate_storm_path(dsx):
    import numpy as np
    from scipy import interpolate
    import xarray as xr
    #after calculating the distance from the storm it became clear that the storm data is every 6 hours, no matter 
    #how much it may have moved.  So if the storm moved 300 km in 6 hr, when calculating the distance to the storm
    #there were points on the storm track that showed large distances because of the separation to the 6hrly storm points
    #this subroutine interpolates the storm path onto a higher spatial resolution
    #the new storm dataset is carefully put into an identical format with i2 and j2 as dims to match the old format
    ynew = []
    tnew = []
    xnew = []
    wnew = []
    pnew = []
    bnew = []
    dsx['lon'] = (dsx.lon-180) % 360 - 180 #put -180 to 180
    for istep in range(1,dsx.lon.shape[1]):
        dif_lat = dsx.lat[0,istep]-dsx.lat[0,istep-1]
        dif_lon = dsx.lon[0,istep]-dsx.lon[0,istep-1]
        x,y,t = dsx.lon[0,istep-1:istep+1].values,dsx.lat[0,istep-1:istep+1].values,dsx.time[0,istep-1:istep+1].values
        w,p,b = dsx.wind[0,istep-1:istep+1].values,dsx.pres[0,istep-1:istep+1].values,dsx.basin[0,istep-1:istep+1].values
        x1,y1,t1 = dsx.lon[0,istep-1:istep].values,dsx.lat[0,istep-1:istep].values,dsx.time[0,istep-1:istep].values
        w1,p1,b1 = dsx.wind[0,istep-1:istep].values,dsx.pres[0,istep-1:istep].values,dsx.basin[0,istep-1:istep].values
        if abs(dif_lat)>abs(dif_lon):
            isign = np.sign(dif_lat)
            if abs(dif_lat)>0.75:
                ynew1 = np.arange(y[0], y[-1], isign.data*0.75)
                f = interpolate.interp1d(y, x, assume_sorted=False)
                xnew1 = f(ynew1)
                f = interpolate.interp1d(y, t, assume_sorted=False)
                tnew1 = f(ynew1)
                f = interpolate.interp1d(y, w, assume_sorted=False)
                wnew1 = f(ynew1)
                f = interpolate.interp1d(y, p, assume_sorted=False)
                pnew1 = f(ynew1)
                f = interpolate.interp1d(y, b, assume_sorted=False)
                bnew1 = f(ynew1)
            else:
                xnew1,ynew1,tnew1,wnew1,pnew1,bnew1 = x1,y1,t1,w1,p1,b1
            xnew,ynew,tnew = np.append(xnew,xnew1),np.append(ynew,ynew1),np.append(tnew,tnew1) 
            wnew,pnew,bnew = np.append(wnew,wnew1),np.append(pnew,pnew1),np.append(bnew,bnew1) 
        else:
            isign = np.sign(dif_lon)
            if abs(dif_lon)>0.75:
                iwrap_interp = 1
                if (x[0]<-90) & (x[-1]>90):
                    iwrap_interp = -1
                    x[0]=x[0]+360
                if (x[0]>90) & (x[-1]<-90):
                    iwrap_interp = -1
                    x[-1]=x[-1]+360
                xnew1 = np.arange(x[0], x[-1], iwrap_interp*isign.data*0.75)
                f = interpolate.interp1d(x, y, assume_sorted=False)
                ynew1 = f(xnew1)
                f = interpolate.interp1d(x, t, assume_sorted=False)
                tnew1 = f(xnew1)
                f = interpolate.interp1d(x, w, assume_sorted=False)
                wnew1 = f(xnew1)
                f = interpolate.interp1d(x, p, assume_sorted=False)
                pnew1 = f(xnew1)
                f = interpolate.interp1d(x, b, assume_sorted=False)
                bnew1 = f(xnew1)
                xnew1 = (xnew1 - 180) % 360 - 180 #put -180 to 180
            else:
                xnew1,ynew1,tnew1 = x1,y1,t1
                wnew1,pnew1,bnew1 = w1,p1,b1
            xnew,ynew,tnew = np.append(xnew,xnew1),np.append(ynew,ynew1),np.append(tnew,tnew1) 
            wnew,pnew,bnew = np.append(wnew,wnew1),np.append(pnew,pnew1),np.append(bnew,bnew1) 
#remove any repeated points
    ilen=xnew.size
    outputx,outputy,outputt,outputw,outputp,outputb=[],[],[],[],[],[]
    for i in range(ilen-1):
        if (xnew[i]==xnew[i+1]) and (ynew[i]==ynew[i+1]):
            continue
        else:
            outputx,outputy,outputt = np.append(outputx,xnew[i]),np.append(outputy,ynew[i]),np.append(outputt,tnew[i])
            outputw,outputp,outputb = np.append(outputw,wnew[i]),np.append(outputp,pnew[i]),np.append(outputb,bnew[i])
    xnew,ynew,tnew=outputx,outputy,outputt
    wnew,pnew,bnew=outputw,outputp,outputb
#put into xarray
    i2,j2=xnew.shape[0],1
    tem = np.expand_dims(xnew, axis=0)
    xx = xr.DataArray(tem.T,dims=['i2','j2'])
    tem = np.expand_dims(ynew, axis=0)
    yy = xr.DataArray(tem.T,dims=['i2','j2'])
    tem = np.expand_dims(tnew, axis=0)
    tt = xr.DataArray(tem.T,dims=['i2','j2'])
    tem = np.expand_dims(wnew, axis=0)
    ww = xr.DataArray(tem.T,dims=['i2','j2'])
    tem = np.expand_dims(pnew, axis=0)
    pp = xr.DataArray(tem.T,dims=['i2','j2'])
    tem = np.expand_dims(bnew, axis=0)
    bb = xr.DataArray(tem.T,dims=['i2','j2'])
    dsx_new = xr.Dataset({'lon':xx.T,'lat':yy.T,'time':tt.T,'wind':ww.T,'pres':pp.T,'basin':bb.T})

#add storm translation speed to storm information
    tdim_storm = dsx_new.time.size
    storm_speed = dsx_new.time.copy(deep=True)*np.nan    
    for i in range(0,tdim_storm-1):
        coords_1 = (dsx_new.lat[i], dsx_new.lon[i])  
        coords_2 = (dsx_new.lat[i+1], dsx_new.lon[i+1])  
        arclen_temp = geopy.distance.geodesic(coords_1, coords_2).km  #distance in km  
        storm_date1 = np.datetime64(date_1858 + dt.timedelta(days=float(dsx_new.time[i])))  
        storm_date2 = np.datetime64(date_1858 + dt.timedelta(days=float(dsx_new.time[i+1])))  
        arclen_time = storm_date2 - storm_date1
        arclen_hr = arclen_time / np.timedelta64(1, 'h')
        storm_speed[i]=arclen_temp/(arclen_hr)
    storm_speed[-1]=storm_speed[-2]
    dsx_new['storm_speed']=storm_speed   
    
    return dsx_new


def get_dist_grid(lat_point,lon_point,lat_grid,lon_grid):
    import geopy.distance
    from math import sin, pi
    import numpy as np
    #this routine takes a point and finds distance to all points in a grid of lat and lon
    #it is slowwwwwww
    dist_grid = np.empty(lat_grid.shape)    
    coords_1 = (lat_point, lon_point)  
    for i in range(0,lat_grid.shape[0]):
        for j in range(0,lat_grid.shape[1]):
            coords_2 = (lat_grid[i,j], lon_grid[i,j])  
            arclen_temp = geopy.distance.geodesic(coords_1, coords_2).km  #distance in km       
            dist_grid[i,j]=arclen_temp
    return dist_grid


def closest_dist(ds_in,ds_storm): 
    import xarray as xr
    import numpy as np
# m.garcia-reyes 2.4.2019, edited c.gentemann 2.4.2019
# calculate distance closest storm point
# point given as tla,tlo.... storm is in the program
# initialize distances (in km)
 #   ds_storm['lon'] = (ds_storm.lon + 180) % 360 - 180
    dsx_input = ds_storm.copy(deep=True)
    ds_storm_new = interpolate_storm_path(dsx_input)       
    tdim,xdim,ydim=ds_storm_new.lat.shape[1], ds_in.analysed_sst[0,:,0].shape[0], ds_in.analysed_sst[0,0,:].shape[0]
    dx_save=np.zeros([tdim,xdim,ydim])
    dx_grid,dy_grid = np.meshgrid(ds_in.lon.values,ds_in.lat.values)
    lon_grid,lat_grid = np.meshgrid(ds_in.lon.values,ds_in.lat.values)
    min_dist_save = np.zeros([xdim,ydim])*np.nan
    min_index_save = np.zeros([xdim,ydim])*np.nan
    min_time_save = np.zeros([xdim,ydim])*np.nan

    position = np.zeros([xdim,ydim])*np.nan
    #for each location of the storm calculate the difference for all values in box
    for ipt in range(0,ds_storm_new.lat.shape[1]):  # all storm values
        dist_tem_grid = get_dist_grid(ds_storm_new.lat[0,ipt].values,ds_storm_new.lon[0,ipt].values,lat_grid,lon_grid)
        dx_save[ipt,:,:]=dist_tem_grid       
    #now go through each value in box and find minimum storm location/day
    ds_tem = ds_in.copy(deep=True)
    for j in range(0,ds_in.lon.shape[0]):
        for i in range(0,ds_in.lat.shape[0]):
            imin = np.argmin(dx_save[:,i,j])
            min_dist_save[i,j]=dx_save[imin,i,j]
            min_index_save[i,j]=imin
            min_time_save[i,j]=ds_storm_new.time[0,imin]
            i1,i2=imin,imin+1
            if i2>=ds_storm_new.lat.shape[1]:
                i1,i2=imin-1,imin
            lonx,laty=ds_in.lon[j],ds_in.lat[i]
    #                sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
            if (ds_storm_new.lon[0,i2]<0) and (lonx>180):
                lonx=lonx-360
            position[i,j] = np.sign((ds_storm_new.lon[0,i2] - ds_storm_new.lon[0,i1]) * (laty - ds_storm_new.lat[0,i1]) 
                                 - (ds_storm_new.lat[0,i2] - ds_storm_new.lat[0,i1]) * (lonx - ds_storm_new.lon[0,i1]))

    return min_dist_save,min_index_save,min_time_save,position,ds_storm_new

def calculate_storm_mask(ds_sst,lats,lons):
    import xarray as xr
    import numpy as np
#make a mask for the storm and only keep data within -4 and 10 degrees of storm track
#this was written before I had calculated the closest_dist which is probably a better mask to use
    iwrap_mask = 0
    if (ds_sst.lon.max().values>170) & (ds_sst.lon.min().values<-170):
        iwrap_mask=1
    print(ds_sst.lon.min().values,ds_sst.lon.max().values)
#okay, now ds_storm is array with right lat/lon for storm so create mask now
    ds_mask = ds_sst.copy(deep=True)
    ds_mask['storm_mask']=ds_mask.analysed_sst*0
    ds_mask = ds_mask.fillna(0)
    ds_mask['storm_mask'] = ds_mask['storm_mask'].astype(int,copy=True)
    for i in range(0,lats.shape[0]):
        if lats[i]>0:   #northern hemi on right, southers on left
            lons1,lons2=lons[i]-4,lons[i]+10
        else:
            lons1,lons2=lons[i]-10,lons[i]+4
        lats1,lats2=lats[i]-10,lats[i]+10
        if i==0:
            print('lons1,lons2:',iwrap_mask,lons1.data,lons2.data)
        if lons1<-180:
            ds_mask['storm_mask'].loc[dict(lon=(ds_mask.lon < lons2) | (ds_mask.lon > lons1+360), lat=slice(lats1,lats2))] = -1
        elif lons2>180:
            ds_mask['storm_mask'].loc[dict(lon=(ds_mask.lon < lons2-360) | (ds_mask.lon > lons1), lat=slice(lats1,lats2))] = -1
        else:
            if iwrap_mask==1:
                ds_mask.coords['lon'] = np.mod(ds_mask['lon'], 360)
                ds_mask = ds_mask.sortby(ds_mask.lon)
                ds_mask['storm_mask'].loc[dict(lon=slice(lons1+360,lons2+360), lat=slice(lats1,lats2))] = -1
                ds_mask.coords['lon'] = (ds_mask.coords['lon'] + 180) % 360 - 180
            else:
                ds_mask['storm_mask'].loc[dict(lon=slice(lons1,lons2), lat=slice(lats1,lats2))] = -1
    return ds_mask

