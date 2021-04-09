"""
GLS data analysis
Nico Lois - August 2020
This code processes GLS data from Magellanic Penguins
"""

# import libraries
import numpy as np
import pandas as pd
import cmocean as cm
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import scipy.io
from scipy.interpolate import griddata
from sklearn.neighbors import KernelDensity
import geopandas as gpd
import earthpy as et
import descartes

directorio_datos = '/home/nicolois/Documents/Tracking_Penguins/data/'

# Load land shapefile
land = gpd.read_file(directorio_datos + "ne_10m_land/ne_10m_land.shp")

# Set range
lon_w = -74
lon_e = -52
lat_s = -57
lat_n = -37

### Create meshgrid with numpy
latitudes = np.arange(lat_s,lat_n,2)
longitudes = np.arange(lon_w,lon_e,2)
nlat = len(latitudes)
nlon = len(longitudes)
glon, glat = np.meshgrid(longitudes, latitudes)

# Load GLS data
all = pd.read_csv(directorio_datos + 'All.csv')
all['dtime'] = pd.to_datetime(all['dtime'], format='%d/%m/%Y %H:%M')
all.index = all['dtime']
grupos_mes = all.groupby('month')

# # Busco localizaciones por mes
# tabla_03 = all.iloc[all.index.month == 3]
# tabla_04 = all.iloc[all.index.month == 4]
# tabla_05 = all.iloc[all.index.month == 5]
# tabla_06 = all.iloc[all.index.month == 6]
# tabla_07 = all.iloc[all.index.month == 7]
# tabla_08 = all.iloc[all.index.month == 8]
# tabla_09 = all.iloc[all.index.month == 9]


#--- load bathy data
data_bati = xr.open_dataset(directorio_datos + 'gebco_2020_n-37.0_s-57.0_w-75.0_e-52.0.nc')
blon = data_bati.lon.values
blat = data_bati.lat.values
blonlen = len(blon)
blatlen = len(blat)
data_bati = data_bati.elevation.values

levels_bati = [-5000, -4000, -3000, -2000, -1000, -500, -200, -100, 0]


### Obtain depth value by sex

depth = []

for i in range(len(all)):
    x = all.iloc[i].loc['lon']
    y = all.iloc[i].loc['lat']
    print('Calculando profundidades', int(i/len(all) * 100), '%',end='\r')
    for ilon in range(len(blon)-1):
        lon0 = blon[ilon]
        lon1 = blon[ilon + 1]
        if lon0 < x and lon1 > x:
            for ilat in range(len(blat)-1):
                lat0 = blat[ilat]
                lat1 = blat[ilat + 1]
                if lat0 < y and lat1 > y:
                    depth1 = data_bati[ilat][ilon]
                    depth.append(depth1)

all['depth'] = depth

nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/tablas/depth.csv'
all.to_csv(nombre_salida)

## Plot depth by point
plt.close('all')
figprops = dict(figsize=(6,4), dpi=72)
fig = plt.figure(**figprops)
plt.clf()
title = ["a) Females", "b) Males"]
posiciones = [[0.1, 0.1, 0.35, 0.8], [0.6, 0.1, 0.35, 0.8]]
pos_cb = [0.1, 0.05, 0.8, 0.01]
lab_cb = 'Depth'
# cmap = [cm.cm.haline, cm.cm.haline]

sex=['F','M']
for j in range(2):
    ax = plt.axes(posiciones[j], projection=ccrs.Mercator())
    ax.set_title(title[j], loc="left", fontsize=8)
    ax.coastlines(resolution='10m', color='black', linewidths=0.4, zorder=3)
    cbati = ax.contour(blon, blat, data_bati, [-2000, -1000, -200], colors='black',
            linewidths=.25, transform=ccrs.PlateCarree())
    data = ax.scatter(all[all.sex == sex[j]].lon.values, all[all.sex == sex[j]].lat.values, marker='.', s=31,
            lw=0, c=all[all.sex == sex[j]].depth.values, cmap=cm.cm.deep_r, vmin=-3000, vmax=0, transform=ccrs.PlateCarree(), label=sex[j])
    ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
    ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
    ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#AAAAAA', zorder=2)
    #ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # ax.title('Fishing effort ' + str(int(imonth)))
    ax.tick_params(labelsize=7)

# colorbar
cax = fig.add_axes(pos_cb)
cb = fig.colorbar(data ,orientation='horizontal',cax=cax, extend = 'both')
cb.ax.set_xlabel(lab_cb,fontsize=8)
cb.ax.tick_params(labelsize=7)

# export
nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/MaleFemale_depth'
fig.savefig(nombre_salida , dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
plt.close('all')


"""
Load penguin data
"""
## Defino meses de interacción para plotear !
meses_interac = [4,5,6,7,8,9]
## Cuento puntos de penguins en grilla ##
M_males       = np.zeros((12,nlat,nlon))
M_males[:]    = np.nan
M_females     = np.zeros((12,nlat,nlon))
M_females[:]  = np.nan

for i, imonth in enumerate(grupos_mes.groups.keys()):
    print(imonth)
    for ilon in range(nlon-1):
        lon0 = longitudes[ilon]
        lon1 = longitudes[ilon+1]
        for ilat in range(nlat-1):
            lat0 = latitudes[ilat]
            lat1 = latitudes[ilat+1]

            df  = all.loc[all['lon'] > lon0]
            df2 = df.loc[df['lat'] > lat0]
            df3 = df2.loc[df2['lon'] < lon1]
            df4 = df3.loc[df3['lat'] < lat1]
            df5 = df4.iloc[df4.index.month == imonth]
            df_males   = df5.loc[df5['sex'] == "M"]
            df_females = df5.loc[df5['sex'] == "F"]

            if int(df5['id'].count()) == 0:
                M_males[int(imonth),ilat,ilon] = np.nan
                M_females[int(imonth),ilat,ilon] = np.nan
            elif int(df_females['id'].count()) == 0:
                M_females[int(imonth),ilat,ilon] = np.nan
                M_males[int(imonth),ilat,ilon] = int(df_males['id'].count())
            elif int(df_males['id'].count()) == 0:
                M_males[int(imonth),ilat,ilon] = np.nan
                M_females[int(imonth),ilat,ilon] = int(df_females['id'].count())
            else:
                M_males[int(imonth),ilat,ilon] = int(df_males['id'].count())
                M_females[int(imonth),ilat,ilon] = int(df_females['id'].count())

# genero matriz con ambos sexos juntos
M_matrix = [M_males, M_females]

# Facet plot males vs females last month
# una hoja A4 mide 11.7 y 8.3 en pulgadas!! -> figsize!
title = ["April", "May",
    "June", "July", "August",
    "September"]

letra = [ ["a) ", "b) "],
    ["c) ", "d) "],
    ["e) ", "f) "], ["g) ", "h) "],
            ["i) ", "j) "], ["k) ", "l) "] ]

#NORMALIZO pinguinos !
M_males_norm       = np.zeros((12,nlat,nlon))
M_males_norm[:]    = np.nan
M_females_norm     = np.zeros((12,nlat,nlon))
M_females_norm[:]  = np.nan

for i, imonth in enumerate(meses_interac):
    max = np.nanmax(M_males[int(imonth),:,:])
    M_males_norm[int(imonth),:,:] = M_males[int(imonth),:,:]/max

    max = np.nanmax(M_females[int(imonth),:,:])
    M_females_norm[int(imonth),:,:] = M_females[int(imonth),:,:]/max

    M_matrix_norm = [M_males_norm, M_females_norm]


#### Plotting penguins !
plt.close('all')
figprops = dict(figsize=(10,8), dpi=72)
fig = plt.figure(**figprops)
plt.clf()

posiciones = [    [    [0.1, 0.7, 0.15, 0.24],  [0.28, 0.7, 0.15, 0.24]   ],
              [    [0.1, 0.4, 0.15, 0.24], [0.28, 0.4,0.15, 0.24]   ],
              [    [0.1, 0.1, 0.15, 0.24],  [0.28, 0.1, 0.15, 0.24]   ],
              [    [0.52, 0.7, 0.15, 0.24], [0.7, 0.7,0.15, 0.24]   ],
              [    [0.52, 0.4, 0.15, 0.24],  [0.7, 0.4, 0.15, 0.24]   ],
              [    [0.52, 0.1, 0.15, 0.24], [0.7, 0.1,0.15, 0.24]   ]      ]

for i, imonth in enumerate(meses_interac):
    print (imonth)
    for j in range(2):
        ax = plt.axes(posiciones[i][j], projection=ccrs.Mercator())
        title_LM = letra[i][j] + title[i] + " males", letra[i][j] + title[i] + " females"
        ax.set_title(title_LM[j], loc="left", fontsize=8)
        ax.coastlines(resolution='10m', color='black', linewidths=0.4, zorder=3)
        # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
        #                     alpha=0.75, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
        cbati = ax.contour(blon, blat, data_bati, [-2000, -1000, -200], colors='gray',
                linewidths=.25, transform=ccrs.PlateCarree())
        data = ax.pcolormesh(glon, glat, M_matrix_norm[j][int(imonth),:,:],
                transform=ccrs.PlateCarree(), cmap='Blues')
        ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
        ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
        ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#AAAAAA', zorder=2)
        #ax.add_feature(cfeature.OCEAN, facecolor='#')
        #ax.plot(all.Lon.values, all.Lat.values, marker='.', markersize=4,
        #    lw=0.25, markeredgecolor='lightcoral', markerfacecolor='lightcoral', transform=ccrs.PlateCarree(), alpha=0.03, linestyle='')
        #ax.plot(all.LONG.values, all.LAT.values, marker='.', markersize=3,
        #    lw=0.25, markeredgecolor='lightblue', markerfacecolor='lightblue', transform=ccrs.PlateCarree(), alpha=0.03, linestyle='')
        # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
        #     alpha=0.55, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        # ax.title('Fishing effort ' + str(int(imonth)))
        if j==1:
            ax.set_yticklabels([])
        if i>=3:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

# colorbar
cax = fig.add_axes([0.325, 0.05, 0.3, 0.005])
cb = fig.colorbar(data ,orientation='horizontal',cax=cax, ticks=[0,0.2,0.4,0.6,0.8,1])
cb.ax.set_xlabel('Normalized location density',fontsize=8)
cb.ax.tick_params(labelsize=7)

# export
nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/MaleFemale_facet'
fig.savefig(nombre_salida , dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
plt.close('all')
	

### SUM Penguins by sex

M_males_sum     = np.zeros((nlat,nlon))
M_males_sum [:] = np.nan

M_females_sum     = np.zeros((nlat,nlon))
M_females_sum [:] = np.nan

M_males_sum   = np.nansum(M_males  [meses_interac,:,:], axis=0)
M_females_sum = np.nansum(M_females[meses_interac,:,:], axis=0)

M_males_sum[M_males_sum == 0]     = np.nan
M_females_sum[M_females_sum == 0] = np.nan

max = np.nanmax(M_males_sum)
M_males_sum_norm = M_males_sum/max

max = np.nanmax(M_females_sum)
M_females_sum_norm = M_females_sum/max

M_peng_summ = [M_males_sum_norm , M_females_sum_norm]


## Plot mean penguin  and SD
plt.close('all')
figprops = dict(figsize=(6,4), dpi=72)
fig = plt.figure(**figprops)
plt.clf()
title = ["a)", "b)"]
posiciones = [[0.1, 0.1, 0.35, 0.8], [0.6, 0.1, 0.35, 0.8]]
pos_cb = [[0.1, 0.05, 0.35, 0.01], [0.6, 0.05, 0.35, 0.01]]
lab_cb = ['Male normalized density', ' Female normalized density']
# cmap = [cm.cm.haline, cm.cm.haline]

for j in range(2):
    ax = plt.axes(posiciones[j], projection=ccrs.Mercator())
    ax.set_title(title[j], loc="left", fontsize=8)
    ax.coastlines(resolution='10m', color='black', linewidths=0.4, zorder=3)
    # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
    #                     alpha=0.75, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
    cbati = ax.contour(blon, blat, data_bati, [-2000, -1000, -200], colors='black',
            linewidths=.25, transform=ccrs.PlateCarree())
    data = ax.pcolormesh(glon, glat, M_peng_summ[j], #norm=True,
            transform=ccrs.PlateCarree(), cmap='Blues')
    ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
    ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
    ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#AAAAAA', zorder=2)
    # ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
    #ax.plot(all.Lon.values, all.Lat.values, marker='.', markersize=4,
    #    lw=0.25, markeredgecolor='lightcoral', markerfacecolor='lightcoral', transform=ccrs.PlateCarree(), alpha=0.03, linestyle='')
    #ax.plot(all.LONG.values, all.LAT.values, marker='.', markersize=3,
    #    lw=0.25, markeredgecolor='lightblue', markerfacecolor='lightblue', transform=ccrs.PlateCarree(), alpha=0.03, linestyle='')
    # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
    #     alpha=0.55, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # ax.title('Fishing effort ' + str(int(imonth)))
    ax.tick_params(labelsize=7)

    # colorbar
    cax = fig.add_axes(pos_cb[j])
    cb = fig.colorbar(data ,orientation='horizontal',cax=cax)
    cb.ax.set_xlabel(lab_cb[j],fontsize=8)
    cb.ax.tick_params(labelsize=7)

# export
nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/MaleFemale_Summary'
fig.savefig(nombre_salida , dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
plt.close('all')


### Plot male vs female last month
# Usamos un csv generado con las localizaciones
# del último mes pre recaptura.
LM_males       = np.zeros((nlat,nlon))
LM_males [:]   = np.nan
LM_females     = np.zeros((nlat,nlon))
LM_females [:] = np.nan

LM = pd.read_csv(directorio_datos + 'lastmonth.csv')

for ilon in range (nlon - 1):
    lon0 = longitudes[ilon]
    lon1 = longitudes[ilon + 1]
    for ilat in range (nlat - 1):
        lat0 = latitudes[ilat]
        lat1 = latitudes[ilat + 1]

        df  = LM.loc[LM['lon'] > lon0]
        df2 = df.loc[df['lat'] > lat0]
        df3 = df2.loc[df2['lon'] < lon1]
        df4 = df3.loc[df3['lat'] < lat1]
        df_males   = df4.loc[df4['sex'] == "M"]
        df_females = df4.loc[df4['sex'] == "F"]

        if int(df4['id'].count()) == 0:
            LM_males[ilat,ilon]   = np.nan
            LM_females[ilat,ilon]   = np.nan
        elif int(df_females['id'].count()) == 0:
            LM_females[ilat,ilon]   = np.nan
            LM_males[ilat,ilon] = int(df_males['id'].count())
        elif int(df_males['id'].count()) == 0:
            LM_males[ilat,ilon]   = np.nan
            LM_females[ilat,ilon] = int(df_females['id'].count())
        else:
            LM_males[ilat,ilon] = int(df_males['id'].count())
            LM_females[ilat,ilon] = int(df_females['id'].count())

#NORMALIZO
max = np.nanmax(LM_males)
LM_males = LM_males/max

max = np.nanmax(LM_females)
LM_females = LM_females/max

# genero matriz con ambos sexos juntos
LM_matrix = [LM_males , LM_females]

# Facet plot males vs females last month
# una hoja A4 mide 11.7 y 8.3 en pulgadas!! -> figsize!
plt.close('all')
figprops = dict(figsize=(7,3), dpi=72)
fig = plt.figure(**figprops)
plt.clf()

posiciones = [[0.1, 0.15, 0.35, 0.9], [0.5, 0.15, 0.35, 0.9]]
title_LM = ["a) Males", "b) Females"]

for i in range(2):
    ax = plt.axes(posiciones[i], projection=ccrs.Mercator())
    ax.set_title(title_LM[i], loc="left", fontsize=8)
    ax.coastlines(resolution='10m', color='black', linewidths=0.4, zorder=3)
    # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
                        # alpha=0.55, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
    cbati = ax.contour(blon, blat, data_bati, [-2000, -1000, -200],
            colors='black',linewidths=.25, transform=ccrs.PlateCarree())
    data = ax.pcolormesh(glon, glat, LM_matrix[i],
            transform=ccrs.PlateCarree(), cmap='Blues')
    ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
    ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
    ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='dimgray', zorder=2)
    #ax.add_feature(cfeature.OCEAN, facecolor='#')
    #ax.plot(all.Lon.values, all.Lat.values, marker='.', markersize=4,
    #    lw=0.25, markeredgecolor='lightcoral', markerfacecolor='lightcoral', transform=ccrs.PlateCarree(), alpha=0.03, linestyle='')
    #ax.plot(all.LONG.values, all.LAT.values, marker='.', markersize=3,
    #    lw=0.25, markeredgecolor='lightblue', markerfacecolor='lightblue', transform=ccrs.PlateCarree(), alpha=0.03, linestyle='')
    # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
    #     alpha=0.55, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # ax.title('Fishing effort ' + str(int(imonth)))
    ax.tick_params(labelsize=7)

# colorbar
cax = fig.add_axes([0.3, 0.05, 0.35, 0.015])
cb = fig.colorbar(data ,orientation='horizontal',cax=cax)
cb.ax.set_xlabel('Last month normalized density',fontsize=8)
cb.ax.tick_params(labelsize=7)

# export
nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/LastMonth'
fig.savefig(nombre_salida, dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
plt.close('all')


"""
Load Fishing data from The Ministry of Agriculture,
Livestock and Fisheries of Argentina database (csv or shp)

Access to this data is dependent on Ministry's clearance.
Please contact Samanta Dodino (sami.dodino@gmail.com) for further info
"""

pesca = gpd.read_file(directorio_datos + 'pesca-points/arrastre-tango.shp')
# pesca.plot()
# plt.show()

# meses de marzo a octubre
meses = pesca.groupby('mes')

# # Filtro por arte de pesca (loc)
# pesca.nom_arte.unique()
arrastre = pesca.loc[pesca["nom_arte"] == "ARRASTRE DE  FONDO"]
tango = pesca.loc[pesca["nom_arte"] == 'RED TANGONERAS']

artes = [arrastre, tango]
pesca_filt = pd.concat(artes)

arte = ['Arrastre', 'Tangoneras']
levels = np.arange(0,5000,100)
for k in range(len(artes)):
    pesca_x = artes[k].geometry.x.values
    pesca_y = artes[k].geometry.y.values
    artes[k]['x'] = pesca_x
    artes[k]['y'] = pesca_y
    points_pesca = np.array([pesca_x, pesca_y]).T
    # pesca_grid = griddata(points, pesca['horas_moni'].values, (latitudes,longitudes) )

    ## Evaluo intensidad de pesca en grilla ##
    M_pesca = np.zeros((12, nlat,nlon))
    M_pesca[:] = np.nan

    for i, imonth in enumerate(meses.groups.keys()):
        print('Procesando base de datos en mes', int(imonth))
        for ilon in range(nlon - 1):
            lon0 = longitudes[ilon]
            lon1 = longitudes[ilon + 1]
            for ilat in range(nlat - 1):
                lat0 = latitudes[ilat]
                lat1 = latitudes[ilat + 1]

                df  = artes[k].loc[artes[k]['x'] > lon0]
                df2 = df.loc[df['y'] > lat0]
                df3 = df2.loc[df2['x'] < lon1]
                df4 = df3.loc[df3['y'] < lat1]
                df5 = df4.loc[df4['mes'] == imonth]

                if df5['horas_moni'].sum() == 0:
                    M_pesca[int(imonth),ilat,ilon] = np.nan
                else:
                    M_pesca[int(imonth),ilat,ilon] = df5['horas_moni'].sum()

    # Scoring pesca !!
    M_pesca_score     = np.zeros((12, nlat,nlon))
    M_pesca_score [:] = np.nan

    M_pesca_score[M_pesca >= 6001] = 5
    M_pesca_score[M_pesca <= 6000] = 4
    M_pesca_score[M_pesca <= 800] = 3
    M_pesca_score[M_pesca <= 200] = 2
    M_pesca_score[M_pesca <= 60 ] = 1
    M_pesca_score[M_pesca == np.nan] = 0


        # M_pesca[int(imonth),:,:] = M_pesca[int(imonth),:,:]/max

    ## Facet fishing effort !!
    plt.close('all')
    figprops = dict(figsize=(6, 5.5), dpi=72)
    fig = plt.figure(**figprops)
    plt.clf()
    print('Preparando el plot de esfuerzo pesquero facetado')

    title = [  "a) April", "b) May",
            "c) June", "d) July", "e) August",
            "f) September"   ]

    posiciones = [  [0.1, 0.6, 0.25, 0.4], [0.4, 0.6, 0.25, 0.4], [0.7, 0.6, 0.25, 0.4],
                    [0.1, 0.12, 0.25, 0.4], [0.4, 0.12, 0.25, 0.4], [0.7, 0.12, 0.25, 0.4]    ]

    for i, imonth in enumerate(meses_interac):
        print(imonth)
        ax = plt.axes(posiciones[i], projection=ccrs.Mercator())
        ax.coastlines(resolution='10m', color='black', linewidths=0.4, zorder=3)
        ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
        # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
        #     alpha=0.75, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
        cbati = ax.contour(blon, blat, data_bati, [-2000, -1000, -200],
            colors='gray',linewidths=.25, transform=ccrs.PlateCarree())
        data = ax.pcolormesh(glon, glat, M_pesca_score[int(imonth),:,:],
            transform=ccrs.PlateCarree(), cmap=cm.cm.matter, vmin=0, vmax=5)
        ax.add_feature(cfeature.LAND, facecolor='#AAAAAA', zorder=2)
        ax.set_title(title[i], loc='left', fontsize=8)
        ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
        ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.set_aspect('equal', 'box')
        if i==4 or i==5 or i==7 or i==8:
            ax.set_yticklabels([])
        if i==1 or i==2:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8)

    # colorbar
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.01])
    cb = fig.colorbar(data, orientation='horizontal',cax=cax, ticks=[1,2,3,4,5])
    cb.ax.set_xlabel('Fishing effort score',fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # export
    nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/FishingEffort_'+arte[k]+'_facet'
    fig.savefig(nombre_salida, dpi=300, bbox_inches='tight')
    fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')


### SUM FISHING EFFORT AND SD

    M_pesca_mean     = np.zeros((nlat,nlon))
    M_pesca_mean [:] = np.nan

    M_pesca_SD     = np.zeros((nlat,nlon))
    M_pesca_SD [:] = np.nan

    M_pesca_mean = np.nanmean(M_pesca_score[meses_interac,:,:], axis=0)
    M_pesca_SD = np.nanstd(M_pesca_score[meses_interac,:,:], axis=0)

    M_pesca_summ = [M_pesca_mean , M_pesca_SD]

    ## Plot mean fishing effort and SD
    plt.close('all')
    figprops = dict(figsize=(6,4), dpi=72)
    fig = plt.figure(**figprops)
    plt.clf()
    title = ["a)", "b)"]
    posiciones = [[0.1, 0.1, 0.35, 0.8], [0.6, 0.1, 0.35, 0.8]]
    pos_cb = [[0.1, 0.05, 0.35, 0.01], [0.6, 0.05, 0.35, 0.01]]
    lab_cb = ['Mean fishing effort score', 'Fishing effort score SD']
    cmap = [cm.cm.matter, 'Purples']

    for j in range(2):
        print('Calculando Mean y SD fishing effort')
        ax = plt.axes(posiciones[j], projection=ccrs.Mercator())
        ax.set_title(title[j], loc="left", fontsize=8)
        ax.coastlines(resolution='10m', color='black', linewidths=0.4, zorder=3)
        # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
        #                     alpha=0.75, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
        cbati = ax.contour(blon, blat, data_bati, [-2000, -1000, -200], colors='black',
                linewidths=.25, transform=ccrs.PlateCarree())
        data = ax.pcolormesh(glon, glat, M_pesca_summ[j],
                transform=ccrs.PlateCarree(), cmap=cmap[j])
        ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
        ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
        ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='dimgray', zorder=2)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        # ax.title('Fishing effort ' + str(int(imonth)))
        ax.tick_params(labelsize=7)

        # colorbar
        cax = fig.add_axes(pos_cb[j])
        cb = fig.colorbar(data ,orientation='horizontal',cax=cax)
        cb.ax.set_xlabel(lab_cb[j],fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # export
    nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/Fishing_Summary' + arte[k]
    fig.savefig(nombre_salida , dpi=300, bbox_inches='tight')
    fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
    plt.close('all')


    #### Mido interacción entre pinguinos y pesquerias

    #### Hago la suma de las interacciones mensuales
    ## sumo meses normalizados y normalizo por el máximo total
    Int_males         = np.zeros((12,nlat,nlon))
    Int_males[:]      = np.nan
    Int_females       = np.zeros((12,nlat,nlon))
    Int_females[:]    = np.nan
    Interac = [Int_males, Int_females]

    for j in range(2):
        m = M_matrix_norm[j]
        for i, imonth in enumerate(meses_interac):
            for ilat in range(nlat):
                for ilon in range(nlon):
                    m_pesca = M_pesca_score[imonth,ilat,ilon]
                    m_ping  = m[imonth,ilat,ilon]
                    print(imonth,ilat,ilon)
                    print(m_pesca,m_ping)

                    if (np.isnan(m_pesca) and ~np.isnan(m_ping)) :
                        valor = -1
                        print("acá está")
                        print("\n")
                    else:
                        valor = m_pesca * m_ping

                    Interac[j][imonth,ilat,ilon] = valor


    # Interac_suma.append(np.nansum(Interac[j][meses_interac,:,:], axis=0))
    # Interac_suma[j][Interac_suma[j] == 0] = np.nan


    # interac_prueba = [np.reshape(ijk[meses_interac,:,:], (len(meses_interac) * nlon * nlat)) for ijk in Interac]

    data = np.empty((660,3))
    interac_prueba = [np.reshape(ijk[meses_interac,:,:], (len(meses_interac), nlon * nlat)) for ijk in Interac]

    df_interac = pd.DataFrame(data, columns=["male", "female", "month"])

    for i in range(6):

        imonthdata = np.array([interac_prueba[0],interac_prueba[1]]).T[:,i]
        imonth = np.ones(110)*(i+4) # para que arranque en abril
        idata = np.column_stack((imonthdata, imonth))
    #    df_interac.concat(idata)
        df_interac.loc[ 110*i : 110*(i+1)-1, : ] = idata


    nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/tablas/MaleFemale_interac_tot'+arte[k]+'.csv'
    df_interac.to_csv(nombre_salida)



    # Facet plot males vs females interaction
    # una hoja A4 mide 11.7 y 8.3 en pulgadas!! -> figsize!

    title = ["April", "May",
            "June", "July", "August",
            "September"]

    letra = [ ["a) ", "b) "],
            ["c) ", "d) "],
            ["e) ", "f) "], ["g) ", "h) "],
                    ["i) ", "j) "], ["k) ", "l) "] ]

    plt.close('all')
    figprops = dict(figsize=(10,8), dpi=72)
    fig = plt.figure(**figprops)
    plt.clf()

    posiciones = [    [    [0.1, 0.7, 0.15, 0.24],  [0.28, 0.7, 0.15, 0.24]   ],
                      [    [0.1, 0.4, 0.15, 0.24], [0.28, 0.4,0.15, 0.24]   ],
                      [    [0.1, 0.1, 0.15, 0.24],  [0.28, 0.1, 0.15, 0.24]   ],
                      [    [0.52, 0.7, 0.15, 0.24], [0.7, 0.7,0.15, 0.24]   ],
                      [    [0.52, 0.4, 0.15, 0.24],  [0.7, 0.4, 0.15, 0.24]   ],
                      [    [0.52, 0.1, 0.15, 0.24], [0.7, 0.1,0.15, 0.24]   ]      ]


    cmap = cm.cm.algae
    cmap.set_under('orange')

    for i, imonth in enumerate(meses_interac):
        print (imonth)
        for j in range(2):
            ax = plt.axes(posiciones[i][j], projection=ccrs.Mercator())
            title_LM = letra[i][j] + title[i] + " male", letra[i][j] + title[i] + " female"
            ax.set_title(title_LM[j], loc="left", fontsize=8)
            ax.coastlines(resolution='10m', color='black', linewidths=0.4, zorder=3)
            # cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
            #                     alpha=0.75, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
            cbati = ax.contour(blon, blat, data_bati, [-2000, -1000, -200], colors='gray',
                    linewidths=.25, transform=ccrs.PlateCarree())
            data = ax.pcolormesh(glon, glat, Interac[j][int(imonth),:,:],
                    transform=ccrs.PlateCarree(), cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
            ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
            ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='#AAAAAA', zorder=2)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            # ax.title('Fishing effort ' + str(int(imonth)))
            if j==1:
                ax.set_yticklabels([])
            if i>=3:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    # colorbar
    cax = fig.add_axes([0.325, 0.05, 0.3, 0.005])
    cb = fig.colorbar(data ,orientation='horizontal',cax=cax)
    cb.ax.set_xlabel('Interaction score',fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # export
    nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/Interaction'+arte[k]
    fig.savefig(nombre_salida , dpi=300, bbox_inches='tight')
    fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
    plt.close('all')



##########################################################
"""
Figura 1 !!
Pesca + individuos
"""
##########################################################

meses_interac = [4,5,6,7,8,9]

M_pesca_arrastre = np.zeros((12, nlat,nlon))
M_pesca_arrastre[:] = np.nan

M_pesca_tango = np.zeros((12, nlat,nlon))
M_pesca_tango[:] = np.nan

M_pesca = [ M_pesca_arrastre, M_pesca_tango ]

for k in range(len(artes)):
    pesca_x = artes[k].geometry.x.values
    pesca_y = artes[k].geometry.y.values
    artes[k]['x'] = pesca_x
    artes[k]['y'] = pesca_y
    points_pesca = np.array([pesca_x, pesca_y]).T
    # pesca_grid = griddata(points, pesca['horas_moni'].values, (latitudes,longitudes) )

    ## Evaluo intensidad de pesca en grilla ##

    for i, imonth in enumerate(meses.groups.keys()):
        print('Procesando base de datos en mes', int(imonth))
        for ilon in range(nlon - 1):
            lon0 = longitudes[ilon]
            lon1 = longitudes[ilon + 1]
            for ilat in range(nlat - 1):
                lat0 = latitudes[ilat]
                lat1 = latitudes[ilat + 1]

                df  = artes[k].loc[artes[k]['x'] > lon0]
                df2 = df.loc[df['y'] > lat0]
                df3 = df2.loc[df2['x'] < lon1]
                df4 = df3.loc[df3['y'] < lat1]
                df5 = df4.loc[df4['mes'] == imonth]

                if df5['horas_moni'].sum() == 0:
                    M_pesca[k][int(imonth),ilat,ilon] = np.nan
                else:
                    M_pesca[k][int(imonth),ilat,ilon] = df5['horas_moni'].sum()

M_pesca_mean0     = np.zeros((nlat,nlon))
M_pesca_mean0 [:] = np.nan
M_pesca_mean1     = np.zeros((nlat,nlon))
M_pesca_mean1 [:] = np.nan
M_pesca_mean      = np.zeros((nlat,nlon))
M_pesca_mean [:]  = np.nan

M_pesca_mean0 = np.nansum(M_pesca[0][meses_interac,:,:], axis=0)
M_pesca_mean1 = np.nansum(M_pesca[1][meses_interac,:,:], axis=0)
M_pesca_mean = M_pesca_mean1 + M_pesca_mean0


# Scoring pesca !!
M_pesca_score     = np.zeros((nlat,nlon))
M_pesca_score [:] = np.nan

M_pesca_score[M_pesca_mean >= 6001] = 5
M_pesca_score[M_pesca_mean <= 6000] = 4
M_pesca_score[M_pesca_mean <= 800] = 3
M_pesca_score[M_pesca_mean <= 200] = 2
M_pesca_score[M_pesca_mean <= 60 ] = 1
#M_pesca_score[M_pesca_mean == np.nan] = 0

M_pesca_score_plot     = np.zeros((nlat,nlon))
M_pesca_score_plot [:] = np.nan

M_pesca_score_plot[M_pesca_score > 2 ] = 1


## All together by sex + fisheries!
grupos = all.groupby('id')
sexo = all.groupby('sex')
plt.close('all')
figprops = dict(figsize=(7, 3.5), dpi=72)
fig = plt.figure(**figprops)
plt.clf()
ax = plt.axes([0.08, 0.15, 0.9, 0.9], projection=ccrs.Mercator())
ax.coastlines(resolution='10m', color='black', linewidths=0.4, zorder=5)
ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_aspect('equal', 'box')
plt.rcParams['legend.title_fontsize'] = 'x-small'
# cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
#         alpha=0.75, cmap=cm.cm.ice, transform=ccrs.PlateCarree())
cbati = ax.contour(blon, blat, data_bati, [-2000, -1000, -200],
        colors='gray',linewidths=.25, transform=ccrs.PlateCarree())
color_fem = [ 'orange', 'darkorange', 'lightcoral', 'brown', 'maroon']
color_males = [ 'cornflowerblue', 'deepskyblue', 'blue', 'darkblue' ]
countf = 0
countm = 0
ax.add_feature(cfeature.LAND, facecolor='dimgray', zorder=2)

data = ax.pcolormesh(glon, glat, M_pesca_score, cmap='Greens',
        transform=ccrs.PlateCarree(), alpha=0.65)

# ax.plot(saf_x, saf_y, color='blue', linestyle='-', linewidth=0.9, transform=ccrs.PlateCarree())
ax.tick_params(labelsize=6)
for i, iID in enumerate(grupos.groups.keys()):
    posiciones = grupos.get_group(iID)
    if posiciones['sex'].iloc[0] == 'F':
        c = color_fem[countf]
        countf = countf+1
        ax.scatter(posiciones.lon.values, posiciones.lat.values, marker='.', s=7,
                            lw=0, color=c, transform=ccrs.PlateCarree(), label='Female '+str(iID), alpha=1)
        ax.legend(fontsize=6, markerscale=1.5, loc="upper left")

for i, iID in enumerate(grupos.groups.keys()):
    posiciones = grupos.get_group(iID)
    if posiciones['sex'].iloc[0] == 'M':
        c = color_males[countm]
        countm = countm+1
        ax.scatter(posiciones.lon.values, posiciones.lat.values, marker='.', s=5,
                    lw=0, color=c, transform=ccrs.PlateCarree(), label='Male '+str(iID), alpha=1)
        ax.legend(fontsize=5.5, markerscale=1.5, loc="upper left")

# colorbar
cax = fig.add_axes([0.35, 0.05, 0.35, 0.01])
cb = fig.colorbar(data ,orientation='horizontal',cax=cax, ticks=[0,1,2,3,4,5])
cb.ax.set_xlabel('Fishing effort score',fontsize=7)
cb.ax.tick_params(labelsize=7)

# export
nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/Figure1-All_tracks_bySex'
fig.savefig(nombre_salida, dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
plt.close('all')



## Facet by individual colored by sex
grupos = all.groupby('id')

plt.close('all')
figprops = dict(figsize=(7, 10), dpi=72)
fig = plt.figure(**figprops)
plt.clf()

posiciones = [[0.1, 0.1, 0.25, 0.25], [0.4, 0.1, 0.25, 0.25], [0.7, 0.1, 0.25, 0.25],
                [0.1, 0.4, 0.25, 0.25], [0.4, 0.4, 0.25, 0.25], [0.7, 0.4, 0.25, 0.25],
                [0.1, 0.7, 0.25, 0.25], [0.4, 0.7, 0.25, 0.25], [0.7, 0.7, 0.25, 0.25]]

lista = [3497, 3499, 3503, 3509, 3514, 3510, 3482, 3488, 3494]

for i, iID in enumerate(lista):
    print(iID)
    loc = grupos.get_group(iID)
    if loc['sex'].iloc[0] == 'F':
        c = 'green'
    else:
        c = 'orange'
    ax = plt.axes(posiciones[i], projection=ccrs.Mercator())
    ax.plot(loc.lon.values, loc.lat.values, marker='.', markersize=3,
        lw=0.25, color=c, transform=ccrs.PlateCarree(), label=iID, alpha=1)
    # plt.axes(posiciones[i], projection=ccrs.Mercator())
    ax.coastlines(resolution='10m', color='black', linewidths=0.4)
    ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
    # ax.add_feature(cfeature.LAND, facecolor='#AAAAAA')
    cbati = ax.contourf(blon[::6], blat[::6], data_bati[::6,::6], levels_bati, extend='both',
        alpha=0.75, cmap=cm.cm.ice, transform=ccrs.PlateCarree())


    ax.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
    ax.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper left', fontsize=8)
    if i==4 or i==5 or i==7 or i==8:
        ax.set_yticklabels([])
    if i==1 or i==2:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=8)

# colorbar
cax = fig.add_axes([0.2, 0.05, 0.6, 0.01])
cb = fig.colorbar(cbati ,orientation='horizontal',cax=cax)
cb.ax.set_xlabel('Depth [m]',fontsize=8)
cb.ax.tick_params(labelsize=8)

# export
nombre_salida = '/home/nicolois/Documents/Tracking_Penguins/figs/Facet_tracks'
fig.savefig(nombre_salida, dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
plt.close('all')
