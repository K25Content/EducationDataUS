import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pylab import plot,show

from bokeh.plotting import *
from bokeh.models.glyphs import Circle
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.browserlib import view
from bokeh.models import (
    GMapOptions, Range1d, BoxSelectionOverlay, HoverTool, BoxSelectTool, 
    WheelZoomTool, ResetTool, PreviewSaveTool, GMapPlot, PanTool
)

uni = pd.read_csv(
"/Users/elektra/Desktop/MSc in Data Science/Principles of Data Science/Report 2/output/MERGED2013_PP.csv"
)
uni = uni[(uni.HIGHDEG == 4)][(uni.CURROPER == 1)]
uni.replace('PrivacySuppressed', np.nan)

print uni.head(1)


x_range = Range1d()
y_range = Range1d()

# JSON style string taken from: https://snazzymaps.com/style/1/pale-dawn
map_options = GMapOptions(lat=30.2861, lng=-97.7394, map_type="roadmap", zoom=3, styles="""
[{"featureType":"administrative","elementType":"all","stylers":[{"visibility":"on"},{"lightness":33}]},
{"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2e5d4"}]},
{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#c5dac6"}]},
{"featureType":"poi.park","elementType":"labels","stylers":[{"visibility":"on"},
{"lightness":20}]},{"featureType":"road","elementType":"all","stylers":[{"lightness":20}]},
{"featureType":"road.highway","elementType":"geometry","stylers":[{"color":"#c5c6c6"}]},
{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#e4d7c6"}]},
{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#fbfaf7"}]},
{"featureType":"water","elementType":"all","stylers":[{"visibility":"on"},{"color":"#acbcc9"}]}]
""")

plot = GMapPlot(
    x_range=x_range, y_range=y_range,
    map_options=map_options,
    title="Map"
)

source = ColumnDataSource(data=dict(
                i=uni.INSTNM,
                x=uni.LONGITUDE,
                y=uni.LATITUDE,
                z=uni.TUITFTE
                )
)
hover = HoverTool(
            tooltips=[
                ("Institution", "@i"),
                ("Lon", "$x"),
                ("Lat", "$y"),
                ("Net tuition revenue per full-time equivalent student", "@z")
            ]
)
box_select = BoxSelectTool()

TOOLS = [
            box_select,
            WheelZoomTool(),
            ResetTool(),
            PreviewSaveTool(),
            hover, PanTool()
]
circle = Circle(x='x', y='y',
                size=7, fill_color='red', line_color="red" 
)
plot.add_glyph(source, circle)
plot.add_tools(*TOOLS)
overlay = BoxSelectionOverlay(tool=box_select)
plot.add_layout(overlay)

doc = Document()
doc.add(plot)

if __name__ == "__main__":
    filename = "maps.html"
    with open(filename, "w") as f:
        f.write(file_html(doc, INLINE, "Google Maps Example"))
    print("Wrote %s" % filename)
    view(filename)


uni09 = pd.read_csv(
"/Users/elektra/Desktop/MSc in Data Science/Principles of Data Science/Report 2/output/MERGED2009_PP.csv"
)
uni09 = uni09[(uni09.HIGHDEG == 4)][(uni09.CURROPER == 1)]
print(uni09.head(1))
uni09 = uni09.replace('PrivacySuppressed', np.nan)

uni09['mn_earn_wne_male0_p10'] = uni09['mn_earn_wne_male0_p10'].astype(float)
uni09['mn_earn_wne_male1_p10'] = uni09['mn_earn_wne_male1_p10'].astype(float)
min(uni09.mn_earn_wne_male1_p10)
max(uni09.mn_earn_wne_male1_p10)
min(uni09.mn_earn_wne_male0_p10)
max(uni09.mn_earn_wne_male0_p10)

cols = []

for col in uni09:
    try:
        uni09[col].astype(float)
        cols.append(col)
    except ValueError:
        pass

kmeansUni = uni09[cols]
  
         
kmeansUni.dtypes
kmeansUni = kmeansUni.drop(kmeansUni[[0, 1, 2, 3, 4, 5, 6, 7, 8]], 1)

for col in kmeansUni:
    kmeansUni[col] = kmeansUni[col].astype(float)

kmeansUni.replace([np.inf, -np.inf], np.nan)
kmeansUni.dropna(axis=1, how='all', inplace=True)

for col in kmeansUni:
    kmeansUni[col].fillna(kmeansUni[col].mean(), inplace=True)



#clustering
#computing K-Means with K = 4 (4 clusters)
kmeansModel = KMeans(init='random', n_clusters=4, n_init=5)
kmeansModel.fit_predict(kmeansUni)
clusterResults = kmeansModel.labels_

uni09['kmeansLabel'] = clusterResults
uni09Clustered = uni09[['INSTNM', 'kmeansLabel', 'TUITFTE']]

#PCA
pca = PCA(n_components=2)

# We first fit a PCA model to the data
pca.fit(kmeansUni)
projectedAxes = pca.transform(kmeansUni)
eigenValues = pca.explained_variance_ratio_
loadings = pca.components_
pcaDF = pd.DataFrame(columns=kmeansUni.columns.values).transpose()
pcaDF['comp1'] = loadings[0]
pcaDF['comp2'] = loadings[1]
# Use any loadings > mean loadings for each component and obtain the index
# The index will be a column/variable from the original dataframe kmeansUni
comp1 = list(pcaDF[(pcaDF['comp1']>np.mean(pcaDF['comp1']))].index)
comp2 = list(pcaDF[(pcaDF['comp2']>np.mean(pcaDF['comp2']))].index)
# Slice original dataframe to only keep comp1 columns
comp1DF = kmeansUni[comp1]
#clustering again for only comp1
#computing K-Means with K = 4 (4 clusters)
kmeansModelcomp1 = KMeans(init='random', n_clusters=4, n_init=5)
kmeansModelcomp1.fit_predict(comp1DF)
clusterResultscomp1 = kmeansModelcomp1.labels_
# Do the same for comp2
# Slice original dataframe to only keep comp1 columns
comp2DF = kmeansUni[comp2]
#clustering again for only comp1
#computing K-Means with K = 4 (4 clusters)
kmeansModelcomp2 = KMeans(init='random', n_clusters=4, n_init=5)
kmeansModelcomp2.fit_predict(comp2DF)
clusterResultscomp2 = kmeansModelcomp2.labels_

# Create new dataframe
uni09['kmeansLabelcomp1'] = clusterResultscomp1+1
uni09['kmeansLabelcomp2'] = clusterResultscomp2+1
uni09Clustered = uni09[['INSTNM', 'kmeansLabel', 'TUITFTE', 'kmeansLabelcomp1', 'kmeansLabelcomp2']]

from bokeh.charts import BoxPlot, output_file, show
from bokeh.sampledata.autompg import autompg as df
from bokeh.charts import defaults, vplot, hplot

defaults.width = 450
defaults.height = 350

# collect and display
output_file("boxplot.html")
source2 = ColumnDataSource(data=dict(
                i=uni09Clustered.INSTNM,
                z=uni09Clustered.TUITFTE
                )
)
hover2 = HoverTool(
            tooltips=[
                ("Institution", "@INSTNM"),
                ("Net tuition revenue per full-time equivalent student", "@TUITFTE")
                #("Institution", "@i"),
                #("Net tuition revenue per full-time equivalent student", "@z")
            ]
)
box_plot = BoxPlot(uni09Clustered, label=['kmeansLabelcomp1', 'kmeansLabelcomp2'], values='TUITFTE',
                    title="label=['kmeansLabelcomp1', 'kmeansLabelcomp2'], values='TUITFTE'")
box_plot.add_tools(hover2)
show(box_plot)

df = uni[['INSTNM', 'LONGITUDE', 'LATITUDE']]
# Merge dfs to get location coordinates
uni_clustered = pd.merge(uni09Clustered, df, on='INSTNM', how='inner')

map_options = GMapOptions(lat=30.2861, lng=-97.7394, map_type="roadmap", zoom=3, styles="""
[{"featureType":"administrative","elementType":"all","stylers":[{"visibility":"on"},{"lightness":33}]},
{"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2e5d4"}]},
{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#c5dac6"}]},
{"featureType":"poi.park","elementType":"labels","stylers":[{"visibility":"on"},
{"lightness":20}]},{"featureType":"road","elementType":"all","stylers":[{"lightness":20}]},
{"featureType":"road.highway","elementType":"geometry","stylers":[{"color":"#c5c6c6"}]},
{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#e4d7c6"}]},
{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#fbfaf7"}]},
{"featureType":"water","elementType":"all","stylers":[{"visibility":"on"},{"color":"#acbcc9"}]}]
""")

x_range = Range1d()
y_range = Range1d()

plot1 = GMapPlot(
    x_range=x_range, y_range=y_range,
    map_options=map_options,
    title="Map"
)

coloring = []

for i in range(len(uni_clustered)):
    if uni_clustered['kmeansLabelcomp1'][i] == 1 and uni_clustered['kmeansLabelcomp2'][i] == 1:
        coloring.append('purple')
    elif uni_clustered['kmeansLabelcomp1'][i] == 1 and uni_clustered['kmeansLabelcomp2'][i] == 2:
        coloring.append('navy')
    elif uni_clustered['kmeansLabelcomp1'][i] == 1 and uni_clustered['kmeansLabelcomp2'][i] == 3:
        coloring.append('yellow')
    elif uni_clustered['kmeansLabelcomp1'][i] == 1 and uni_clustered['kmeansLabelcomp2'][i] == 4:
        coloring.append('blue')
    elif uni_clustered['kmeansLabelcomp1'][i] == 2 and uni_clustered['kmeansLabelcomp2'][i] == 1:
       coloring.append('teal')
    elif uni_clustered['kmeansLabelcomp1'][i] == 2 and uni_clustered['kmeansLabelcomp2'][i] == 2:
       coloring.append('#6a3d9a')
    elif uni_clustered['kmeansLabelcomp1'][i] == 2 and uni_clustered['kmeansLabelcomp2'][i] == 3:
        coloring.append('brown')
    elif uni_clustered['kmeansLabelcomp1'][i] == 2 and uni_clustered['kmeansLabelcomp2'][i] == 4:
        coloring.append('red')
    elif uni_clustered['kmeansLabelcomp1'][i] == 3 and uni_clustered['kmeansLabelcomp2'][i] == 1:
        coloring.append('grey')
    elif uni_clustered['kmeansLabelcomp1'][i] == 3 and uni_clustered['kmeansLabelcomp2'][i] == 2:
        coloring.append('#a6cee3')
    elif uni_clustered['kmeansLabelcomp1'][i] == 3 and uni_clustered['kmeansLabelcomp2'][i] == 3:
        coloring.append('orange')
    elif uni_clustered['kmeansLabelcomp1'][i] == 3 and uni_clustered['kmeansLabelcomp2'][i] == 4:
        coloring.append('pink')
    elif uni_clustered['kmeansLabelcomp1'][i] == 4 and uni_clustered['kmeansLabelcomp2'][i] == 1:
        coloring.append('fuchsia')
    elif uni_clustered['kmeansLabelcomp1'][i] == 4 and uni_clustered['kmeansLabelcomp2'][i] == 2:
        coloring.append('white')
    elif uni_clustered['kmeansLabelcomp1'][i] == 4 and uni_clustered['kmeansLabelcomp2'][i] == 3:
        coloring.append('maroon')
    elif uni_clustered['kmeansLabelcomp1'][i] == 4 and uni_clustered['kmeansLabelcomp2'][i] == 4:
        coloring.append('#444444')
    else:
        coloring.append('black')

source = ColumnDataSource(data=dict(
                i=uni_clustered.INSTNM,
                x=uni_clustered.LONGITUDE,
                y=uni_clustered.LATITUDE,
                z=uni_clustered.TUITFTE,
                chroma=coloring
                )
)
hover = HoverTool(
            tooltips=[
                ("Institution", "@i"),
                ("Lon", "$x"),
                ("Lat", "$y"),
                ("Net tuition revenue per full-time equivalent student", "@z")
            ]
)
box_select = BoxSelectTool()

TOOLS = [
            box_select,
            WheelZoomTool(),
            ResetTool(),
            PreviewSaveTool(),
            hover, PanTool()
]



circle = Circle(x='x', y='y',
                size=7, fill_color='chroma', line_color="black" 
)
plot1.add_glyph(source, circle)
plot1.add_tools(*TOOLS)
overlay = BoxSelectionOverlay(tool=box_select)
plot1.add_layout(overlay)

doc = Document()
doc.add(plot1)

if __name__ == "__main__":
    filename = "maps.html"
    with open(filename, "w") as f:
        f.write(file_html(doc, INLINE, "Google Maps Example"))
    print("Wrote %s" % filename)
    view(filename)
