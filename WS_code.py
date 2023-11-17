# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:27:15 2023

@author: jm2842
"""

#This code is used to analyse data from the Water Security framework assessment. 

#%% Libraries 
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inequality.theil import Theil
from sklearn.neighbors import KNeighborsRegressor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
import esda
from geopandas import GeoDataFrame
from esda.moran import Moran
from pysal.lib import weights
from splot import esda as esdaplot


#%% Import dataframe with scores and geometry

filename = 'SectorsScores/SectorsScores.shp' 
#original dataframe with scores for each territorial unit 
utb_scores_org = gpd.read_file(filename)

#make a copy of the original dataframe
utb_scores =utb_scores_org.copy()


#%% Spatial interpolation

# IMPORTANT: CHANGE THE SECTOR'S ID AND THE COLUMN NAME OF THE POPULATION IN THIS PART: search for id_sector AND pop_ind_name
#Transform geometries to centroids for interpolation

utb_scores_cent = utb_scores.copy()
utb_scores_cent.geometry = utb_scores_cent["geometry"].centroid


#Check which indicators contain NaN

#initialising the lists of indicators
indList = []
indNaN  = []

k = 0
#loop to go through the indicators and get the names of all indicators and of those with NaN among the scores
for (label, data) in utb_scores.items():
    if "_id" not in label and (label.startswith("a") or label.startswith("b") or label.startswith("c") or label.startswith("d")):
        indList.append(label)
        indNaN.append(int(data.isna().sum()))
        k += 1
        
# List of indicators containing NaN:
indnanlist = [label for (i, label) in enumerate(indList) if indNaN[i] > 0]


# check if any collum has only NaN 
nan_col= (utb_scores.isnull().all()).reset_index()
nan_col.columns=['id','bool']

#add id of the indicador that has only NaN to a list

nan_all_list = nan_col[nan_col['bool']==True]
nan_all_list = nan_all_list.id.tolist()

#remove indicator with only NaN from the list of indicators containing NaN to proceed to the spatial interpolation
for i in nan_all_list: 
    indnanlist.remove(i)


# Spatial interpolation 

#create a copy of the dataframe with the scores, with only the id of each sector and the geometry

id_sector = 'utb_id'


utb_scores_fix = utb_scores[[id_sector, 'geometry']].copy()


# Get X and Y coordinates of the centroids
x_data = utb_scores_cent["geometry"].x
y_data = utb_scores_cent["geometry"].y

# Create list of XY coordinate pairs
xy_coords = np.array([list(xy) for xy in zip(x_data, y_data)])

#loop to go through the list of indicators and do the spatial interpolation. It prints the results before and after inteprolation
for label in indList:
    if label in indnanlist:
        # Get list of indicator values
        value_ind = utb_scores_cent[label]
        
        # Get only non-NaN points
        idx_nan    = utb_scores_cent.loc[ pd.isna(utb_scores_cent[label]), :].index
        idx_notnan = utb_scores_cent.loc[~pd.isna(utb_scores_cent[label]), :].index
        
        xy_coords_notnan = xy_coords[idx_notnan]
        value_ind_notnan = np.array(list(value_ind[idx_notnan]))
        
        xy_coords_nan = xy_coords[idx_nan]
        value_ind_nan = np.array(list(value_ind[idx_nan]))
        
        # Set number of neighbors to look for
        neighbors = 3
        
        # Initialize KNN regressor
        knn_regressor = KNeighborsRegressor(n_neighbors = neighbors, weights = "distance")
        
        # Fit regressor to data
        knn_regressor.fit(xy_coords_notnan, value_ind_notnan)
        
        value_ind_knn = knn_regressor.predict(xy_coords_nan)
        
        utb_scores_fix[label] = pd.concat([pd.Series(value_ind_notnan, index=idx_notnan), 
                                        pd.Series(value_ind_knn, index=idx_nan)])
        
        # Generate in-sample R^2
        in_r_squared_knn = knn_regressor.score(xy_coords_nan, value_ind_knn)
        print("KNN in-sample r-squared: {}".format(round(in_r_squared_knn, 2)))
        
        fig, ax = plt.subplots(1, 2)
        
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        utb_scores.plot(ax = ax[0], column=label, edgecolor='grey', linewidth=0.1, cmap = 'RdYlGn')
        utb_scores_fix.plot(ax = ax[1], column=label, cmap = 'RdYlGn',edgecolor='grey', linewidth=0.1, legend=True, cax=cax)
        
        ax[0].set_title('Raw', fontsize=15)
        ax[1].set_title('Interpolated', fontsize=15)
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        
        fig.suptitle(label, fontsize=16)
        
        
        plt.show()
    else:
        utb_scores_fix[label] = utb_scores[label]


#associate the population with the new interpolated dataframe

pop_ind_name = 'tot_pop' # change to the name attributed to the population collumn 
scores_df_interp= pd.merge(
    utb_scores_fix, utb_scores[[id_sector, pop_ind_name]],  on=id_sector, how="left")

#%% Summary of scores
    
#import a summary of indicators and sample sizes, acording to READ_ME file 
filename = 'Sample_size.xls'
size = pd.read_excel(filename)


#select only sectors with population>0
scores_df = (scores_df_interp[scores_df_interp[pop_ind_name] > 0]).reset_index()

#select a dataframe with only the indicators, based on the list of codes provided in the size file (Read_me file)
df_utb = scores_df[size.code].reset_index(level=0)
df_utb = df_utb.drop(['index'], axis=1)

 
# ind_list is the list of names of indicators 
ind_list = [col for col in df_utb]

# summary of scores:
scores_summary = df_utb.describe()
scores_summary = scores_summary.transpose().reset_index(level=0).rename(columns={'index': 'code'})
scores_summary['sample_size'] = size['sample_size']
scores_summary['name'] = size['name']
scores_summary['Dimension'] = size['Dim']
    
#%% Theil entropy index
 
# We calculate the Theil entroypy index only for indicators with sample size>5
#Creating a subset of indicators  to be analysed
subset_theil = df_utb[size['code']
                      [size['sample_size'] >= 5]].reset_index(level=0)
subset_theil = subset_theil.iloc[:, 1:]


#Function to compute the Theil index: 
    
def theil(column):
    theil_y = Theil(column.values)
    return theil_y.T


#Initialising 
theil_results = []
colunas = []

#Loop to compute the Theil index for the subset 
for col in subset_theil:
    y = np.asarray(subset_theil[[col]])
    theil_y = Theil(y)
    theil_results.append(theil_y.T)
    colunas.append(col)

#creating a dataframe with only the code for each indicator and the computed Theil index:
colunas_df = pd.DataFrame(colunas, columns=['code'])

theil_df = pd.DataFrame(theil_results, columns=['theil'])
theil_df['code'] = colunas_df

#Adding the dimension and category based on the code:
theil_df['Dimension'] = theil_df['code'].astype(str).str[0].str.upper()
theil_df['Category'] = theil_df['code'].astype(str).str[:2].str.upper()
theil_df['Indicator'] = theil_df['code'].astype(str).str[:2].str.upper()+'.'+theil_df['code'].astype(str).str[2:3].str.upper()

#Adding the mean score 
theil_df = pd.merge(
    theil_df, scores_summary[['code', 'mean']], on='code', how="left")

#function to annotate indicator names in the plot
def anotation(x, y, mylist):
    for i in range(len(mylist)):
        ax.annotate(mylist[i], (x[i], y[i]), textcoords='offset points', xytext=(10, 10),
                    ha='left', va='baseline')


#Plot quadrant with Theil  and score for all indicators
colors = {'A': 'tab:blue', 'B': 'tab:green', 'C': 'tab:red', 'D': 'tab:orange'}

fig, ax = plt.subplots(figsize=(10, 10))
bla = plt.scatter(theil_df['mean'],
                  theil_df['theil'],
                  c=theil_df['Dimension'].map(colors),
                  s=150,
                  alpha=0.5)
plt.axvline(0.5, c='gray', ls='--')
plt.axhline(0.2, c='gray', ls='--')
# anotation(theil_df['mean'],theil_df['theil'],
#           theil_df['id'])
texts = [plt.text(theil_df['mean'][i],
                  theil_df['theil'][i],
                  theil_df['code'][i],
                  # c=colors[theil_df['Dimension'][i]],
                  fontweight='normal',
                  fontsize=14)
         for i in range(len(theil_df))]
adjust_text(texts,
            x=list(theil_df['mean']),
            y=list(theil_df['theil']),
            precision=0.01,
            add_objects=[bla],
            expand_text=(1.2, 1.2),
            expand_points=(1.5, 1.5),
            expand_objects=(2, 2),
            autoalign=True,
            # arrowprops=dict(arrowstyle="->", color='r', lw=0.5),
            )

# giving X and Y label
plt.xlabel("X: Score (mean)", fontsize=18)
plt.ylabel("Y: Inequality index", fontsize=18)

ax.tick_params(axis='both', which='major', labelsize=16)
#Names of Indicators to put in the legend
A_selection = '\nA1.1 - Water demand \nA3.1 - Access to piped drinking water \nA3.2 - Access to wastewater collection \nA3.3 - Affordability\nA4.2 - Service reliability \nA5.1 - Water diseases \nA5.2 - Recreational opportunities'
B_selection = '\nB1.1 - Green areas \nB1.2 - Environmental diseases \nB2.3 - Wastewater treatment rate \nB3.1 - Wastewater treatment efficiency \nB4.1 - Solid waste collection'
C_selection = '\nC1.3 - Flood-prone area \nC1.4 - People leaving in hazardous zones \nC2.2 - Storm drains \nC2.3 - Pavement'
D_selection = '\nD2.1 - Literacy \nD2.2 - Population density \nD2.3 - Inequality coefficient\nD2.4 - Income\nD2.5 -  Informal dwellings \nD2.6 - Gender equality'


labels = list(colors.keys())
labels2 = ['Drinking water and \n human well-being'+A_selection,
           'Ecosystems'+B_selection,
           'Water related hazard\n and climate change'+C_selection,
           'Economic and social\n development'+D_selection]
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label], alpha=0.5)
           for label in labels]
plt.legend(handles, labels2, loc='center', title='Dimensions and indicators', fontsize=14, title_fontsize=14,
           frameon=False, bbox_to_anchor=(1.3, 0.5),
           labelspacing=1.5)

plt.show()
 
#%% Plot maps with scores results

#Create a copy of the data frame with the results you want to plot. This dataframe needs to have the geometry. 

    
my_df =utb_scores_fix.copy() 
Cdf=my_df[['geometry']+ind_list]
    
#import a base map. Territorial units basemap source: Campinas geospatial database from the Campinas Municipal Council https://informacao-didc.campinas.sp.gov.br/metadados.php  freely available to use.
filename = 'Base Map/pd2018_utb.shp' 
base_map= gpd.read_file(filename)

# check the base map 
#base_map.plot()

#check Coordinate Reference System (CRS) 
crs_basemap = base_map.crs
print(crs_basemap)
#make sure the Cdf and the base map  are in the same EPSG
Cdf = Cdf.to_crs(31983) # change acording to the CRS of the basemap 
   
   
#Function to plot the maps with scores
#input the name of the category or dimension (catname) and the  code (catID)


def plot_cat(catName,catID):
    Cdf[catName] = Cdf.loc[:, Cdf.columns.str.startswith(catID)].mean(axis=1)

    indTitle = catName
    
    fig, ax = plt.subplots(figsize  = (15, 15))
    base_map.plot(color='gainsboro', edgecolor = 'grey', lw=0.5, ax = ax)

    Cdf.plot(cmap='RdYlBu', vmin = 0, vmax =1,column=indTitle, edgecolor = 'black', ax = ax,
             legend=True,
               legend_kwds={'label': indTitle,
                            'orientation': "vertical"})

    ax.set_axis_off()
    
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=30)
    cb_ax.set_ylabel(indTitle, rotation=90, size=30)

   
    #if you want to save the figure:
    #plt.savefig(catName+'.png', dpi=300, bbox_inches='tight')  

    plt.tight_layout()
    #plt.show()
   
    plt.axis('equal')


#examples: 
plot_cat('C3_score','c3')

plot_cat('C_score','c')

plot_cat('A11 ','a11')

#%% Visualisation - Radar plot 

            
#import data for the city 
filename = 'CityScores.xls' 
df_city= pd.read_excel(filename, index_col=0).reset_index()
df_city['Dim'] = df_city.ID.str[:1]
df_city['cat'] = (df_city.ID.str[:3]).str.replace('.', '')

# data from sectors
df_sectors = scores_df_interp.copy()

#names of the dimensions
dim_names = {'Dim': ['A', 'B','C','D'],
             'Name_dim': ['Drinking water and human well-being', 
                          'Ecosystems',
                          'Water related hazards and climate change',
                          'Economic and social development']}
dim_names = pd.DataFrame(data=dim_names)


#radar per sector function 

def radar_dim(df,Dim,colors,indlabels):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,polar=True)

    df_aux = df[df['Dim']==Dim].reset_index(drop=True)

    sample = df_aux.norm_value
    N = len(sample) 

    theta = np.arange(0, 2*np.pi, 2*np.pi/N) 
    ax.bar(theta, sample.fillna(0), width=6.3/len(sample),color=df_aux['cat'].map(colors),alpha=0.8)


    for name in df_aux.Name:
        idx = df_aux.index[df_aux['Name'] == name].values[0]
        my_color = df_aux[df_aux['Name'] == name]['cat'].map(colors)
        df_sectors[name[0:4].replace('_','')]
        ax.scatter((theta[idx]).repeat(len(df_sectors[name[0:4].replace('_','')])), utb_scores[name[0:4].replace('_','')],
                   c=my_color, s= utb_scores[pop_ind_name]/200, 
                   edgecolors='black',
                   linewidths=1,
                   alpha=0.8,zorder=2)
        
    #c = ax.scatter(theta, sample, c='black', s=12, alpha=0.5,zorder=2)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.set_xticks(theta)
    ax.set_xticklabels(df_aux.ID,size=14)
    ax.yaxis.grid(True) 
    ax.set_ylim(-0.15,1.1)
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0],color="gray",size=12,ha='center')


    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha=1) for label in labels]
    legend1 = plt.legend(handles, indlabels,  loc='upper left', bbox_to_anchor=(1.1, 0.9),ncol=1,
               title=(dim_names[dim_names['Dim']==Dim].Name_dim).item(), 
               fontsize=14,title_fontsize=16, frameon=False,
               labelspacing = 1)
    # Create empty plot with blank marker containing the extra label
        
    l1, = plt.plot([],[], 'o', c='w',markeredgecolor='k', ms=(400)**(1/2))
    l2, = plt.plot([],[], 'o', c='w',markeredgecolor='k', ms=(200)**(1/2))
    l3, = plt.plot([],[], 'o', c='w',markeredgecolor='k', ms=(100)**(1/2))
    l4, = plt.plot([],[], 'o', c='w',markeredgecolor='k',ms=(5)**(1/2))
    labelspop = ['80k', '40k', '20k','1k']



    legend2 = plt.legend([l1, l2, l3,l4],labelspop,
                         ncol=4,
                         title='Population per sector', 
                         fontsize=14,title_fontsize=16, frameon=False,
                         loc='upper left', bbox_to_anchor=(1.1, 1.1),
                         columnspacing=0.8,
                         labelspacing = 1.5)
    legend2._legend_box.align = "left"
    legend1._legend_box.align = "left"

    plt.gca().add_artist(legend2)
    plt.gca().add_artist(legend1)
    
    #to save figures
    #plt.savefig('Radar'+Dim+'_'+'.png', dpi=300, bbox_inches='tight')  



    plt.show()
    
    
#info for dimensions
#definition of the name of the indicators and categories for the legend: 
    
indA1 = '\nA1.1 - Water demand\nA1.2 - Water availability\nA1.3 - Diversity of sources\nA1.4 - Storage capacity\nA1.5 - Water stress'
indA2 = '\nA2.1 - Drinking water quality'
indA3 = '\nA3.1 - Access to piped drinking water\nA3.2 - Access to wastewater collection\nA3.3 - Affordability'
indA4='\nA4.1 - Service discontinuity\nA4.2 - Service reliability\nA4.3 - Metering level\nA4.4 - Water loss'
indA5 = '\nA5.1 - Water diseases\nA5.2 - Recreational opportunities'

 
indAnames=['WATER QUANTITY '+indA1,
                     'WATER QUALITY'+indA2,
                     'ACCESSIBILITY TO SERVICES'+indA3,
                     'INFRASTRUCTURE RELIABILITY'+indA4, 
                     'PUBLIC HEALTH AND WELL-BEING '+indA5]

indBnames = ['ENVIRONMENT'+
             '\nB1.1 - Green areas\nB1.2 - Environmental safety',
             'POLLUTION CONTROL'+'\nB2.1 - Groundwater quality\nB2.2 - Surface water quality\nB2.3 - Wastewater treatment rate',
             'USAGE EFFICIENCY'+'\nB3.1 - Wastewater treatment efficiency\nB3.2 - Wastewater reuse rate',
             'SOLID WASTE'+'\nB4.1 - Solid waste collection\nB4.2 - Solid waste recycling']

indCnames = ['WATER-RELATED DISASTERS'+
             '\nC1.1 - Flood frequency\nC1.2 - Drought frequency\nC1.3 - Flood-prone areas\nC1.4 - People leaving in hazardous zones',
             'PREPAREDNESS'+ '\nC2.1 - Risk management\nC2.2 - Flood protection infrastructure (storm drain)\nC2.3 - Flood protection infrastructure (pavement)\nC2.4 - Investment in drainage',
             'CLIMATE CHANGE'+'\nC3.1 - CO2 emissions\nC3.2 - Temperature\nC3.3 - Extreme events of precipitation']

indD1 = '\nD1.1 - Communication and access\nD1.2 - Public participation\nD1.3 - Equality and non-discrimination\nD1.4 - Infrastructure investment\nD1.5 - Water self-sufficiency\nD1.6 - Regulation, planning, institutional framework'
indD2 = '\nD2.1 - Literacy\nD2.2 - Population density\nD2.3 - Inequality coefficient\nD2.4 - Income\nD2.5 - Informal dwellings\nD2.6 - Gender equality'
indD3 = '\nD3.1 - Per capita GDP\nD3.2 - Water productivity'

indDnames=['GOVERNANCE'+indD1,
           'SOCIAL ASPECTS'+indD2,
           'ECONOMIC DEVELOPMENT'+indD3]


#definition of colors: 
    
Blueshades=plt.get_cmap('Blues')
colorsA = {'A1':(Blueshades(0.98)), 
           'A2':(Blueshades(0.75)),
           'A3':(Blueshades(0.55)),
           'A4':(Blueshades(0.4)),
           'A5':(Blueshades(0.25))}

Greenshades=plt.get_cmap('Greens')
colorsB = {'B1':(Greenshades(0.98)), 
           'B2':(Greenshades(0.75)),
           'B3':(Greenshades(0.55)),
           'B4':(Greenshades(0.4))}

Redshades=plt.get_cmap('Reds')
colorsC = {'C1':(Redshades(0.98)), 
           'C2':(Redshades(0.65)),
           'C3':(Redshades(0.35))}

Orangeshades=plt.get_cmap('YlOrBr')
colorsD = {'D1':(Orangeshades(0.8)), 
           'D2':(Orangeshades(0.50)),
           'D3':(Orangeshades(0.25))}


#plots

radar_dim(df_city,'A',colorsA,indAnames)

radar_dim(df_city,'B',colorsB,indBnames)

radar_dim(df_city,'C',colorsC,indCnames)

radar_dim(df_city,'D',colorsD,indDnames)

#%% Spatial autocorrelation

#dataframe with code, geometry and indicators: 
    
db = scores_df_interp.copy()

#select your indicator for spatial autocorrelation
My_ind = 'a32'   


#spatial correlation function definition
#input: db: dataframe and VarName: variable that I want to analyse (in our case, the indicator code: My_ind)

def spatial_cor(db,VarName):
    # Generate W from the GeoDataFrame
    w = weights.distance.KNN.from_dataframe(db, k=3)
    # Row-standardization
    w.transform = 'R'
    
    #Let us first calculate the spatial lag of our variable of interest:
    db['w_'+VarName] = weights.spatial_lag.lag_spatial(w, db[VarName])
    
    #And their respective standardized versions, where we subtract the average and divide by the standard deviation:

    db[VarName+'_std'] = ( db[VarName] - db[VarName].mean() )
    db['w_'+VarName+'_std'] = ( db['w_'+VarName] - db['w_'+VarName].mean() )

    #Moran Plot
    f, ax = plt.subplots(1, figsize=(6, 6))
    sns.regplot(x=VarName+'_std', y='w_'+VarName+'_std', data=db, ci=None);
    plt.axvline(0, c='k', alpha=0.5)
    plt.axhline(0, c='k', alpha=0.5)
    # Add text labels for each quadrant
    plt.text(db[VarName+'_std'].max()/2, db['w_'+VarName+'_std'].max()/2, "HH", fontsize=25, c='r')

    plt.text(db[VarName+'_std'].min()/2, db['w_'+VarName+'_std'].min()/2, "LL", fontsize=25, c='r')
    plt.title("Moran Plot") 

    plt.show()
    
    lisa = esda.moran.Moran_Local(db[VarName], w)
    
    
    #Reference distribution
    ax = sns.kdeplot(lisa.Is)
    sns.rugplot(lisa.Is, ax=ax);
    ax.set_title('Reference distribution')
    
    
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    axs = axs.flatten()
    ax = axs[0]
    base_map.plot(ax=ax, color='none', edgecolor='black')
    db.assign(Is=lisa.Is).plot(column='Is',cmap='viridis',scheme='quantiles',k=5, 
        edgecolor='white', linewidth=0.1, alpha=0.75,legend=True,ax=ax)
    ax = axs[1]
    base_map.plot(ax=ax, color='none', edgecolor='black')
    esdaplot.lisa_cluster(lisa, db, p=1, ax=ax);
    ax = axs[2]
    base_map.plot(ax=ax, color='none', edgecolor='black')
    labels = pd.Series(1 * (lisa.p_sim < 0.05),index=db.index).map({1: 'Significant', 0: 'Non-Significant'})
    db.assign(cl=labels).plot(column='cl',categorical=True,k=2,cmap='Paired',
        linewidth=0.1,edgecolor='white',legend=True, ax=ax)
    ax = axs[3]
    base_map.plot(ax=ax, color='none', edgecolor='black')
    esdaplot.lisa_cluster(lisa, db, p=0.05, ax=ax);
    for i, ax in enumerate(axs.flatten()):
        ax.set_axis_off()
        ax.set_title(['Local Statistics- '+ VarName,'Scatterplot Quadrant- '+ VarName, 
                'Statistical Significance- '+ VarName, 'Moran Cluster Map- '+ VarName][i], y=0)
    f.tight_layout()
    plt.show()
    
    print(VarName)
    moran = Moran(db[VarName], w)
    print('Moran Index:', moran.I) # the value of Moran's I corresponds with the slope of the linear fit overlayed on top of the Moran Plot.
    print('p_sim:',moran.p_sim)
    quadrants =pd.DataFrame(lisa.q) #values indicate quandrant location 1 HH,  2 LH,  3 LL,  4 HL
    quadrants.columns = ['quadrants']
    p_val = pd.DataFrame(lisa.p_sim)
    p_val.columns = ['p_val']
    quadrants['p_val']=p_val['p_val']
    quadrants[id_sector]=db[id_sector]
    quadrants=pd.merge(quadrants, utb_scores[[id_sector,pop_ind_name,'geometry']], on=id_sector, how="left")
    quad = quadrants[(quadrants['quadrants']==3) & (quadrants['p_val']<0.05)]
    quad_geo = GeoDataFrame(quad, crs="EPSG:4326")
    
    quad_geo = quad_geo['geometry'].to_crs({'proj':'cea'}) 
    quad_geo.area.sum()
    
    utb_area =  utb_scores['geometry'].to_crs({'proj':'cea'}) 
    utb_area.area.sum()
    
    print('nub_LL =',quad[pop_ind_name].count())
    print('pop_LL =',quad[pop_ind_name].sum())
    print('%pop_LL =',((quad[pop_ind_name].sum())/utb_scores[pop_ind_name].sum())*100, '%')
    print('%urban area that is LL = ', (quad_geo.area.sum()/utb_area.area.sum())*100,'%')

     
 
    
    
    f, ax = plt.subplots(figsize=(8, 8))
    params = {'legend.fontsize': 18}
    plt.rcParams.update(params)
    #cps.plot(ax=ax, color='none', edgecolor='black')
    esdaplot.lisa_cluster(lisa, db, p=0.05,alpha = 0.7,legend=False, ax=ax);
    db.plot(column=My_ind, color='none',edgecolor='gray', k=5, legend=True, ax=ax)
    base_map.plot(ax=ax, color='none', edgecolor='grey')
    ax.set_axis_off()
    
    colors = {'Hot spot (HH)':'#d7191c', 
              'HL':'#fdaf61',
              'LH':'#abdae9',
              'Cold spot (LL)':'#2c7ab6',
              'ns':'#d3d3d3'}
    labels = list(colors.keys())
    handles = [plt.Line2D((0,0),(1,1),linestyle='none', marker='o',markersize=18, color=colors[label], alpha=0.7) for label in labels]
    plt.legend(handles, labels, loc='lower right', title='Cluster', fontsize=20,title_fontsize=20, frameon=True)
    plt.title("Significance Map") 
    
    f.tight_layout()

    
    #plt.savefig(VarName+'Moran Cluster Map'+'.png', dpi=300, bbox_inches='tight') 
    plt.show()
    

    
    sig = 1 * (lisa.p_sim < 0.05)
    HH = 1 * (sig * lisa.q == 1)
    LL = 3 * (sig * lisa.q == 3)
    LH = 2 * (sig * lisa.q == 2)
    HL = 4 * (sig * lisa.q == 4)
    cluster_x = HH + LL + LH + HL

    cluster_labels = ["ns", "HH", "LH", "LL", "HL"]
    
    labels = [cluster_labels[i] for i in cluster_x]
    
       
   
    db_assigned = db.assign(cl=labels)
    
    #Other choice of colors 
    
    colors = {'LL':'indigo',
              'LH':'mediumslateblue',
              'HL':'mediumaquamarine',
              'HH':'gold', 
              'ns':'gainsboro'}
   
   
    
    db_assigned['c'] = db_assigned['cl'].map(colors)
    
    #significance map: Other choice of colors 
    f, ax = plt.subplots(1, figsize=(9, 9))
    base_map.plot(ax=ax, color='none', edgecolor='grey')
    db_assigned.plot(column='cl', categorical=True, \
            k=2,color=db_assigned['c'], linewidth=0.1, ax=ax, \
            edgecolor='white', legend=False,legend_kwds={'loc': 'lower right', 'title':'Cluster', 'fontsize':20,'title_fontsize':20})
    db.plot(column=My_ind, color='none',edgecolor='gray', k=5, legend=False, ax=ax)
    
    colors = {'Low/Low (LL)':'indigo',
              'Low/High (LH)':'mediumslateblue',
              'High/Low score (HL)':'mediumaquamarine',
              'High/High score (HH)':'gold', 
              'Non-significant':'gainsboro'}
    
    labels = list(colors.keys())
    handles = [plt.Line2D((0,0),(1,1),linestyle='none', marker='o',markersize=18, color=colors[label]) for label in labels]
    plt.legend(handles, labels, loc = 'lower right',title="Sector/Neighbour's score", fontsize=20,title_fontsize=20, frameon=True)
    #plt.title("Significance Map") 

    ax.set_axis_off()
    f.tight_layout()

   
    #plt.savefig(VarName+'Moran Cluster Map_newcolormap'+'.png', dpi=300, bbox_inches='tight') 
    
    plt.show()
    
       
        
        
#run function   

spatial_cor(db,My_ind)  