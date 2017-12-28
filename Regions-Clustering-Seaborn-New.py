
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import minmax_scale
from sklearn import cluster
from sklearn import decomposition
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[2]:


excel_file = pd.ExcelFile('source_data.xlsx')


# In[3]:


sheet0 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[0],
        skiprows=[0],
        header=1,
    )
sheet0 = sheet0[sheet0.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet0.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[4]:


sheet0.head()


# In[5]:


sheet1 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[1],
        skiprows=[0, 1, 2],
        header=1,
    )
sheet1 = sheet1[sheet1.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet1.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[6]:


sheet2 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[2],
        skiprows=[0, 1],
        header=1,
    )
sheet2 = sheet2[sheet2.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet2.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[7]:


sheet3 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[3],
        skiprows=[0, 1],
        header=1,
    )
sheet3 = sheet3[sheet3.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet3.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[8]:


sheet4 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[4],
        skiprows=[0, 1, 2],
        header=1,
    )
sheet4 = sheet4[sheet4.columns.drop(['Unnamed: 0', 'Unnamed: 1'])]
sheet4.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[9]:


sheet5 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[5],
        skiprows=[0, 1],
        header=1,
    )
sheet5 = sheet5[sheet5.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet5.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[10]:


sheet6 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[6],
        skiprows=[0, 1, 2, 3, 4],
        header=1,
    )
sheet6 = sheet6[sheet6.columns.drop(['Unnamed: 0', 'Unnamed: 1'])]
sheet6.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[11]:


sheet7 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[7],
        skiprows=[0, 1, 2, 3],
        header=1,
    )
sheet7 = sheet7[sheet7.columns.drop(['Unnamed: 0', 'Unnamed: 1'])]
sheet7.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[12]:


sheet8 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[8],
        skiprows=[0, 1, 2],
        header=1,
    )
sheet8 = sheet8[sheet8.columns.drop(['Unnamed: 0', 'Unnamed: 1'])]
sheet8.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[13]:


sheet9 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[9],
        skiprows=[0, 1, 2, 3],
        header=1,
    )
sheet9 = sheet9[sheet9.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet9.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[14]:


sheet10 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[10],
        skiprows=[0, 1, 2],
        header=1,
    )
sheet10 = sheet10[sheet10.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet10.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[15]:


sheet11 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[11],
        skiprows=[0, 1, 2],
        header=1,
    )
sheet11 = sheet11[sheet11.columns.drop(['Unnamed: 0', 'Unnamed: 1'])]
sheet11.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[16]:


sheet12 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[12],
        skiprows=[0, 1, 2],
        header=1,
    )
sheet12 = sheet12[sheet12.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet12.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[17]:


sheet13 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[13],
        skiprows=[0, 1],
        header=1,
    )
sheet13 = sheet13[sheet13.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet13.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[18]:


sheet14 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[14],
        skiprows=[0, 1],
        header=1,
    )
sheet14 = sheet14[sheet14.columns.drop(['Unnamed: 0', 'Unnamed: 1'])]
sheet14.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[19]:


sheet15 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[15],
        skiprows=[0, 1, 2],
        header=1,
    )
sheet15 = sheet15[sheet15.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet15.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[20]:


sheet16 = pd.read_excel(
        'source_data.xlsx',
        sheetname=excel_file.sheet_names[16],
        skiprows=[0, 1, 2],
        header=1,
    )
sheet16 = sheet16[sheet16.columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'])]
sheet16.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[21]:


sheet17 = pd.read_excel('source_data_1.xlsx')
sheet17 = sheet17[sheet17.columns.drop(['Unnamed: 11', 'ВНУТРЕННИЕ ЗАТРАТЫ НА НАУЧНЫЕ ИССЛЕДОВАНИЯ И РАЗРАБОТКИ'])]
sheet17.rename_axis({'Область /год': 'Region'} ,axis='columns', inplace=True)


# In[22]:


list_of_df = []
list_of_df.append([sheet0, 'OrganizationNum'])
list_of_df.append([sheet1, 'StaffNum'])
list_of_df.append([sheet2, 'PhDNum'])
list_of_df.append([sheet3, 'Ph.DNum'])
list_of_df.append([sheet4, 'PostgraduateNum'])
list_of_df.append([sheet5, 'DoctoralNum'])
list_of_df.append([sheet6, 'PatentNum'])
list_of_df.append([sheet7, 'UsefulPatentsNum'])
list_of_df.append([sheet8, 'CreatedTechnologyNum'])
list_of_df.append([sheet9, 'UsefulTechnologyNum'])
list_of_df.append([sheet10, 'ProportionOfOrganizationsToUseInternet'])
list_of_df.append([sheet11, 'ProportionOfInnovativeOrgainzations'])
list_of_df.append([sheet12, 'TechnologicalInnovationsCost'])
list_of_df.append([sheet13, 'AmountOfInnovativeProducts'])
list_of_df.append([sheet14, 'ProportionOfInnovativeProducts'])
list_of_df.append([sheet15, 'Population'])
list_of_df.append([sheet16, 'GrossProduct'])
list_of_df.append([sheet17, 'InternalCosts'])


# In[23]:


for i, df_container in enumerate(list_of_df):
    df = df_container[0]
    list_of_df[i][0] = df[df.Region != 'ВСЕГО']    


# In[24]:


merged_df_list = []
for year in range(2005, 2015):
    result_df, feature_name = list_of_df[0]
    result_df = result_df[['Region', year]]

    #result_df.rename_axis({year: feature_name} ,axis='columns', inplace=True)
    result_df = result_df.rename_axis({year: feature_name} ,axis='columns')
    
    merged_df_list.append(result_df)
    
    for i in range(1, 18):
        merged_df, feature_name = list_of_df[i]
        merged_df = merged_df[['Region', year]]
        #merged_df.rename_axis({year: feature_name} ,axis='columns', inplace=True)
        merged_df = merged_df.rename_axis({year: feature_name} ,axis='columns')
        merged_df_list[year - 2005] = pd.merge(
            merged_df_list[year - 2005],
            merged_df,
            on='Region')


# In[25]:


def str_to_num(s):
    if isinstance(s, str):
        s = s.replace(',', '.')
        s1 = s.replace(',', '')
        if s1.isdigit():
            return float(s)
    return s


# In[26]:


df0 = merged_df_list[0]


# In[27]:


def detect_minus(x):
    if x == '-':
        print(x)
        return True
    return False


# In[28]:


df0 = df0.applymap(str_to_num)
df0.set_value(36, 'PostgraduateNum', 0)


# In[29]:


df0_names_row_numbers_map = [(i, name) for i, name in enumerate(df0['Region'])]
df0_names_columns_numbers_map = [(i, name) for i, name in enumerate(df0.columns)]


# In[30]:


cluster_data = df0.values


# In[31]:


normalized_cluster_data = minmax_scale(cluster_data[:, 1:].astype('float'))


# In[32]:


km = cluster.KMeans(n_clusters=3)


# In[33]:


y_km = km.fit_predict(normalized_cluster_data)


# In[34]:


km.fit(normalized_cluster_data)


# In[35]:


def draw_clusters(
    clusters_num,
    axes,
    X,
    y_km,
    x_coord,
    y_coord,
    markers,
    colors,
    centres_color='red',
    centres_marker='*',
    cluster_centers=None,
    region_names=None,
    fontsize=24,
    region_s=300,
    centres_s = 200,
    xlim=(0, 1),
    ylim=(0, 1),
    legend_loc=2):
    for i in range(clusters_num):
        mask = y_km == i
        axes.scatter(
            X[mask, x_coord],
            X[mask, y_coord],
            s=300,
            c=colors[i],
            marker=markers[i],
            label="cluster %s" % i
        )
        
    if cluster_centers is not None:   
        axes.scatter(
            cluster_centers[:, x_coord],
            cluster_centers[:, y_coord],
            s=centres_s,
            marker=centres_marker,
            c=centres_color,
            label='centroids'
           )
    axes.legend(loc=legend_loc, prop={'size': fontsize})
    #plt.show()


# In[36]:


X = normalized_cluster_data

figure = plt.figure(figsize=(24, 24))

axes = figure.add_subplot(111)
axes.scatter(X[y_km == 0, 4],
            X[y_km == 0, 7],
            s=300,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
#for i in range(X[y_km == 0].shape[0]):
#   axes.annotate(i, X[i, 4], X[i, 7])

#labels = ['point{0}'.format(i) for i in range(X[y_km == 0].shape[0])]
labels = df0['Region'][y_km == 0].tolist()

#plt.subplots_adjust(bottom = 0.1)
#plt.scatter(
#    data[:, 0], data[:, 1], marker='o', c=data[:, 2], s=data[:, 3] * 1500,
#    cmap=plt.get_cmap('Spectral'))

for label, x, y in zip(labels, X[y_km == 0, 4], X[y_km == 0, 7]):
    axes.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=24)

axes.scatter(X[y_km == 1, 4],
            X[y_km == 1, 7],
            s=300,
            c='orange',
            marker='o',
            label='cluster 2'
           )
axes.scatter(X[y_km == 2, 4],
            X[y_km == 2, 7],
            s=300,
            c='lightblue',
            marker='v',
            label='cluster 3'
           )
axes.scatter(km.cluster_centers_[:, 4],
            km.cluster_centers_[:, 7],
            s=400,
            marker='*',
            c='red',
            label='centroids'
           )
axes.set_xlabel(list_of_df[4][1])
axes.set_ylabel(list_of_df[7][1])
#axes.set_xlim([0, 600])
#axes.set_ylim([0, 600])

#axes.set_xlim([0, 0.1])
#axes.set_ylim([0, 0.3])

#axes.set_xlim([0, 0.1])
#axes.set_ylim([0, 0.1])

axes.set_xlim([0, 0.03])
axes.set_ylim([0, 0.03])

#axes.set_xlim([0, 0.005])
#axes.set_ylim([0, 0.005])

#axes.set_xlim([0, 0.001])
#axes.set_ylim([0, 0.001])
for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
             axes.get_xticklabels() + axes.get_yticklabels()):
    item.set_fontsize(24)
axes.legend(loc=2, prop={'size':24})
axes.grid()

plt.show()


# In[37]:


X = normalized_cluster_data
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
           )
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3'
           )
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids'
           )
plt.legend()
plt.grid()
plt.show()


# In[38]:


X = normalized_cluster_data
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 2],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 2],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
           )
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 2],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3'
           )
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 2],
            s=250,
            marker='*',
            c='red',
            label='centroids'
           )
plt.legend()
plt.grid()
plt.show()


# In[39]:


X = normalized_cluster_data
plt.scatter(X[y_km == 0, 15],
            X[y_km == 0, 16],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
plt.scatter(X[y_km == 1, 15],
            X[y_km == 1, 16],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
           )
plt.scatter(X[y_km == 2, 15],
            X[y_km == 2, 16],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3'
           )
plt.scatter(km.cluster_centers_[:, 15],
            km.cluster_centers_[:, 16],
            s=250,
            marker='*',
            c='red',
            label='centroids'
           )
plt.legend()
plt.grid()
plt.show()


# In[40]:


X = normalized_cluster_data
plt.scatter(X[y_km == 0, 11],
            X[y_km == 0, 12],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
plt.scatter(X[y_km == 1, 11],
            X[y_km == 1, 12],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
           )
plt.scatter(X[y_km == 2, 11],
            X[y_km == 2, 12],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3'
           )
plt.scatter(km.cluster_centers_[:, 11],
            km.cluster_centers_[:, 12],
            s=250,
            marker='*',
            c='red',
            label='centroids'
           )
plt.legend()
plt.grid()
plt.show()


# In[41]:


y_km_1 = km.predict(normalized_cluster_data)


# In[42]:


df0.var()


# In[43]:


df0.std()


# In[44]:


df0_drop_regions = df0[df0.columns.drop(['Region'])]


# In[45]:


from sklearn.preprocessing import MinMaxScaler


# In[46]:


scaler = MinMaxScaler()


# In[47]:


df0_scaled = pd.DataFrame(scaler.fit_transform(df0_drop_regions), columns=df0_drop_regions.columns)


# In[48]:


df0_scaled.corr().round(2)


# In[49]:


plt.figure(figsize=(24, 24))
heatmap_fig = sns.heatmap(df0_scaled.corr().round(2), annot=True, annot_kws={"size":24}, cbar=False)
for item in heatmap_fig.get_xticklabels():
    item.set_fontsize(24)
for item in heatmap_fig.get_yticklabels():
    item.set_fontsize(24)
for item in heatmap_fig.get_label():
    item.set_fontsize(24)


# In[50]:


merged_df_list[0].set_value(36, 'PostgraduateNum', 0)


# In[51]:


merged_df_list[0].set_value(36, 'PostgraduateNum', 0)
merged_df_list[0].set_value(41, 'GrossProduct', 167139716)
merged_df_dict = {}
for i in range(0, 10):
    prepared_df = merged_df_list[i].applymap(str_to_num)
    prepared_df = prepared_df.applymap(lambda x: 0.0 if x in ['-', '0)'] else x)
    merged_df_dict[i + 2005] = prepared_df   


# In[52]:


#merged_df_dict
hier_concatenated_df = pd.concat(merged_df_dict, names=['Year', 'Region'])


# In[53]:


hier_concatenated_df.shape


# In[54]:


#hier_concatenated_df
print(hier_concatenated_df.index)
print(hier_concatenated_df.index.names)


# In[55]:


hier_concatenated_df = hier_concatenated_df.applymap(str_to_num)
hier_concatenated_df = hier_concatenated_df.applymap(lambda x: 0.0 if x in ['-', '0)'] else x)


# In[56]:


for column_name in hier_concatenated_df.columns:
    print(column_name)
    print(hier_concatenated_df.loc[hier_concatenated_df[column_name].astype(str) == '1.671.397.1.6'])


# In[57]:


print(hier_concatenated_df.loc[(2014, 41), 'GrossProduct'])


# In[58]:


hier_concatenated_df.loc[(2014, 41), 'GrossProduct'] = 167139716


# In[59]:


print(hier_concatenated_df.columns)


# In[60]:


droped_regions_hier_concatenated_df = hier_concatenated_df[hier_concatenated_df.columns.drop('Region')]


# In[61]:


droped_regions_hier_concatenated_df.head()


# In[62]:


for column_name in hier_concatenated_df.columns:
    print(column_name)
    print(hier_concatenated_df.loc[hier_concatenated_df[column_name].astype(str) == '395.700.1'])


# In[63]:


hier_concatenated_df.loc[(2014, 8), 'GrossProduct'] = 3957001


# In[64]:


droped_regions_hier_concatenated_df = hier_concatenated_df[hier_concatenated_df.columns.drop('Region')]


# In[65]:


scaled_with_dropped_regions_hier_concantenated_df = pd.DataFrame(
        scaler.fit_transform(droped_regions_hier_concatenated_df),
        columns=droped_regions_hier_concatenated_df.columns
    )


# In[66]:


prepared_df_values = scaled_with_dropped_regions_hier_concantenated_df.values
bandwidth = cluster.estimate_bandwidth(prepared_df_values, quantile=0.1, n_samples=prepared_df_values.shape[0])
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(prepared_df_values)
labels = ms.labels_
cluster_centres = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)


# In[67]:


plt.figure(figsize=(24, 24))
heatmap_fig = sns.heatmap(scaled_with_dropped_regions_hier_concantenated_df.corr().round(2), annot=True, annot_kws={"size":24}, cbar=False)
for item in heatmap_fig.get_xticklabels():
    item.set_fontsize(24)
for item in heatmap_fig.get_yticklabels():
    item.set_fontsize(24)
for item in heatmap_fig.get_label():
    item.set_fontsize(24)


# In[68]:


cross_corr_matrix = scaled_with_dropped_regions_hier_concantenated_df.corr()


# In[69]:


print(cross_corr_matrix)


# In[70]:


print(cross_corr_matrix['ProportionOfOrganizationsToUseInternet'])


# In[71]:


print(cross_corr_matrix['UsefulTechnologyNum'])


# In[72]:


print(cross_corr_matrix['ProportionOfInnovativeOrgainzations'])


# In[73]:


print(cross_corr_matrix['AmountOfInnovativeProducts'])


# In[74]:


print(cross_corr_matrix['GrossProduct'])


# In[75]:


print(cross_corr_matrix['ProportionOfInnovativeProducts'])


# In[76]:


print(cross_corr_matrix['Ph.DNum'])


# In[77]:


print(labels)


# In[78]:


len(labels)


# In[79]:


cluster_centres


# In[80]:


len(cluster_centres)


# In[81]:


prepared_for_supevised_df = scaled_with_dropped_regions_hier_concantenated_df.copy()


# In[82]:


prepared_for_supevised_df['cluster_class'] = labels


# In[83]:


prepared_for_supevised_df.head()


# In[84]:


prepared_for_supevised_df.shape


# In[85]:


def report_best_score(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[86]:


from sklearn.datasets import load_digits


# In[87]:


digits = load_digits()


# In[88]:


digits.target


# In[89]:


param_dist = {"max_depth": [3, 7],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# In[90]:


clf = RandomForestClassifier(n_estimators=20)


# In[91]:


n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)


# In[92]:


start = time()
random_search.fit(prepared_df_values, labels)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report_best_score(random_search.cv_results_)


# In[93]:


param_grid = {"max_depth": [3, 7],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)


# In[94]:


start = time()
grid_search.fit(prepared_df_values, labels)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report_best_score(grid_search.cv_results_)


# In[95]:


better_clf = RandomForestClassifier(n_estimators=20, bootstrap=True, criterion='entropy', max_depth=7, max_features=10, min_samples_leaf=1, min_samples_split=3)


# In[96]:


better_clf.fit(prepared_df_values, labels)


# In[97]:


better_clf.feature_importances_


# In[98]:


#importances = forest.feature_importances_
importances = better_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in better_clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(prepared_df_values.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(prepared_df_values.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(prepared_df_values.shape[1]), indices)
plt.xlim([-1, prepared_df_values.shape[1]])
plt.show()


# In[99]:


features = scaled_with_dropped_regions_hier_concantenated_df.columns


# In[100]:


for i in indices:
    print(features[i])


# In[101]:


clipped_scaled_with_dropped_regions_hier_concantenated_df = scaled_with_dropped_regions_hier_concantenated_df[['ProportionOfOrganizationsToUseInternet', 'UsefulTechnologyNum', 'ProportionOfInnovativeOrgainzations', 'AmountOfInnovativeProducts', 'GrossProduct', 'ProportionOfInnovativeProducts', 'Ph.DNum']]


# In[102]:


print(clipped_scaled_with_dropped_regions_hier_concantenated_df.columns)


# In[103]:


print(len(clipped_scaled_with_dropped_regions_hier_concantenated_df.columns))


# In[104]:


clipped_prepared_df_values = clipped_scaled_with_dropped_regions_hier_concantenated_df.values
bandwidth_for_clipped = cluster.estimate_bandwidth(clipped_prepared_df_values, quantile=0.1, n_samples=clipped_scaled_with_dropped_regions_hier_concantenated_df.shape[0])
ms_1 = cluster.MeanShift(bandwidth=bandwidth_for_clipped, bin_seeding=True)
ms_1.fit(clipped_prepared_df_values)
labels_for_clipped = ms_1.labels_
cluster_centres_for_clipped = ms_1.cluster_centers_
labels_unique_for_clipped = np.unique(labels_for_clipped)
n_clusters_for_clipped_ = len(labels_unique_for_clipped)
print("number of estimated clusters : %d" % n_clusters_for_clipped_)


# In[105]:


clf_1 = RandomForestClassifier(n_estimators=20)


# In[106]:


param_dist_for_clipped = {"max_depth": [3, 7],
              "max_features": sp_randint(1, 7),
              "min_samples_split": sp_randint(2, 7),
              "min_samples_leaf": sp_randint(1, 7),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# In[107]:


n_iter_search = 20
random_search_1 = RandomizedSearchCV(clf_1, param_distributions=param_dist_for_clipped,
                                   n_iter=n_iter_search)


# In[108]:


start = time()
random_search_1.fit(clipped_prepared_df_values, labels_for_clipped)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report_best_score(random_search_1.cv_results_)


# In[109]:


better_clf_for_clipped = RandomForestClassifier(n_estimators=20, bootstrap=True, criterion='gini', max_depth=7, max_features=5, min_samples_leaf=2, min_samples_split=5)


# In[110]:


better_clf_for_clipped.fit(clipped_prepared_df_values, labels_for_clipped)


# In[111]:


better_clf_for_clipped.feature_importances_


# In[112]:


importances_for_clipped = better_clf_for_clipped.feature_importances_
std = np.std([tree.feature_importances_ for tree in better_clf_for_clipped.estimators_],
             axis=0)
indices_for_clipped = np.argsort(importances_for_clipped)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(clipped_prepared_df_values.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices_for_clipped[f], importances_for_clipped[indices_for_clipped[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(clipped_prepared_df_values.shape[1]), importances_for_clipped[indices_for_clipped],
       color="r", yerr=std[indices_for_clipped], align="center")
plt.xticks(range(clipped_prepared_df_values.shape[1]), indices_for_clipped)
plt.xlim([-1, clipped_prepared_df_values.shape[1]])
plt.show()


# In[113]:


features_for_clipped = clipped_scaled_with_dropped_regions_hier_concantenated_df.columns


# In[114]:


for i in indices_for_clipped:
    print(features_for_clipped[i])


# In[115]:


print(features_for_clipped)


# In[116]:


work_df = clipped_scaled_with_dropped_regions_hier_concantenated_df


# In[117]:


print(work_df['ProportionOfOrganizationsToUseInternet'].min(), work_df['ProportionOfOrganizationsToUseInternet'].max())


# In[118]:


print(work_df['ProportionOfInnovativeOrgainzations'].min(), work_df['ProportionOfInnovativeOrgainzations'].max())


# In[119]:


y_mean_shift = ms_1.predict(clipped_prepared_df_values)


# In[120]:


print(y_mean_shift.shape)


# In[121]:


X = clipped_prepared_df_values
plt.scatter(X[y_mean_shift == 0, 0],
            X[y_mean_shift == 0, 2],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
plt.scatter(X[y_mean_shift == 1, 0],
            X[y_mean_shift == 1, 2],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
           )
plt.scatter(X[y_mean_shift == 2, 0],
            X[y_mean_shift == 2, 2],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3'
           )
plt.scatter(X[y_mean_shift == 3, 0],
            X[y_mean_shift == 3, 2],
            s=50,
            c='black',
            marker='x',
            label='cluster 4'
           )
plt.scatter(X[y_mean_shift == 4, 0],
            X[y_mean_shift == 4, 2],
            s=50,
            c='magenta',
            marker='d',
            label='cluster 5'
           )
plt.scatter(ms_1.cluster_centers_[:, 0],
            ms_1.cluster_centers_[:, 2],
            s=250,
            marker='*',
            c='red',
            label='centroids'
           )
plt.legend()
plt.grid()
plt.show()


# In[122]:


cluster_centres_for_clipped = ms_1.cluster_centers_


# In[123]:


print(cluster_centres_for_clipped.shape)


# In[124]:


print(cluster_centres.shape)


# In[125]:


y_mean_shift = ms.predict(prepared_df_values)
X = prepared_df_values
plt.scatter(X[y_mean_shift == 0, 10],
            X[y_mean_shift == 0, 11],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
plt.scatter(X[y_mean_shift == 1, 10],
            X[y_mean_shift == 1, 11],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
           )
plt.scatter(X[y_mean_shift == 2, 10],
            X[y_mean_shift == 2, 11],
            s=50,
            c='blue',
            marker='v',
            label='cluster 3'
           )
plt.scatter(X[y_mean_shift == 3, 10],
            X[y_mean_shift == 3, 11],
            s=50,
            c='black',
            marker='x',
            label='cluster 4'
           )
plt.scatter(X[y_mean_shift == 4, 10],
            X[y_mean_shift == 4, 11],
            s=50,
            c='magenta',
            marker='d',
            label='cluster 5'
           )
plt.scatter(ms.cluster_centers_[:, 10],
            ms.cluster_centers_[:, 11],
            s=250,
            marker='*',
            c='red',
            label='centroids'
           )
plt.legend()
plt.grid()
plt.show()


# In[126]:


plt.scatter(X[y_mean_shift == 0, 10],
            X[y_mean_shift == 0, 9],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
plt.scatter(X[y_mean_shift == 1, 10],
            X[y_mean_shift == 1, 9],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
           )
plt.scatter(X[y_mean_shift == 2, 10],
            X[y_mean_shift == 2, 9],
            s=50,
            c='blue',
            marker='v',
            label='cluster 3'
           )
plt.scatter(X[y_mean_shift == 3, 10],
            X[y_mean_shift == 3, 9],
            s=50,
            c='black',
            marker='x',
            label='cluster 4'
           )
plt.scatter(X[y_mean_shift == 4, 10],
            X[y_mean_shift == 4, 9],
            s=50,
            c='magenta',
            marker='d',
            label='cluster 5'
           )
plt.scatter(ms.cluster_centers_[:, 10],
            ms.cluster_centers_[:, 9],
            s=250,
            marker='*',
            c='red',
            label='centroids'
           )
plt.legend()
plt.grid()
plt.show()


# In[127]:


plt.scatter(X[y_mean_shift == 0, 11],
            X[y_mean_shift == 0, 9],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1'
           )
plt.scatter(X[y_mean_shift == 1, 11],
            X[y_mean_shift == 1, 9],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2'
           )
plt.scatter(X[y_mean_shift == 2, 11],
            X[y_mean_shift == 2, 9],
            s=50,
            c='blue',
            marker='v',
            label='cluster 3'
           )
plt.scatter(X[y_mean_shift == 3, 11],
            X[y_mean_shift == 3, 9],
            s=50,
            c='black',
            marker='x',
            label='cluster 4'
           )
plt.scatter(X[y_mean_shift == 4, 11],
            X[y_mean_shift == 4, 9],
            s=50,
            c='magenta',
            marker='d',
            label='cluster 5'
           )
plt.scatter(ms.cluster_centers_[:, 11],
            ms.cluster_centers_[:, 9],
            s=250,
            marker='*',
            c='red',
            label='centroids'
           )
plt.legend()
plt.grid()
plt.show()


# In[128]:


print(ms.cluster_all)


# In[129]:


print(ms.cluster_centers_)


# In[130]:


#print(ms.labels_)


# In[131]:


print(ms.labels_.shape)


# In[132]:


X = prepared_df_values
cluster_labels = ms.labels_
figure = plt.figure(figsize=(24, 24))
text_cluster_labels = ("1", "2", "3", "4", "5")
bool_cluster_labels = (cluster_labels == 0)
axes = figure.add_subplot(111)
axes.scatter(X[bool_cluster_labels, 10],
            X[bool_cluster_labels, 11],
            s=200,
            c='lightgreen',
            marker='s',
            label='cluster 1'
        )
bool_cluster_labels = (cluster_labels == 1)
axes.scatter(X[bool_cluster_labels, 10],
            X[bool_cluster_labels, 11],
            s=150,
            c='orange',
            marker='o',
            label='cluster 2'
        )
bool_cluster_labels = (cluster_labels == 2)
axes.scatter(X[bool_cluster_labels, 10],
            X[bool_cluster_labels, 11],
            s=150,
            c='blue',
            marker='v',
            label='cluster 3'
        )
bool_cluster_labels = (cluster_labels == 3)
axes.scatter(X[bool_cluster_labels, 10],
            X[bool_cluster_labels, 11],
            s=150,
            c='black',
            marker='^',
            label='cluster 4'
        )
bool_cluster_labels = (cluster_labels == 4)
axes.scatter(X[bool_cluster_labels, 10],
            X[bool_cluster_labels, 11],
            s=150,
            c='magenta',
            marker='d',
            label='cluster 5'
        )
axes.scatter(ms.cluster_centers_[:, 10],
            ms.cluster_centers_[:, 11],
            s=300,
            marker='*',
            c='red',
            label='centroids'
        )
for text_cluster_label, x, y in zip(
    text_cluster_labels,
    ms.cluster_centers_[:, 10],
    ms.cluster_centers_[:, 11]):
    axes.annotate(
        text_cluster_label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=24)
axes.legend(loc='upper left', prop={'size': 24})
axes.grid()
plt.show()


# In[133]:


class ClustersDrawer(object):
    
    #def __init__(self, plt, X, figure=None):
    def __init__(self, plt, X, figure=None):
        self.X = X
        if figure:
            self.figure = figure
        else:
            # Import matplotlib there?
            self.figure = plt.figure(figsize=(24, 24))

