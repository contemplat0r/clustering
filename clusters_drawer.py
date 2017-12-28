import matplotlib.pyplot as plt
import matplotlib.lines as lines


X = prepared_df_values
cluster_labels = ms.labels_
figure = plt.figure(figsize=(24, 24))
text_cluster_labels = ("1", "2", "3", "4", "5")
axes = figure.add_subplot(111)
bool_cluster_labels = (cluster_labels == 0)
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
    #    self.X = X
    #    if figure:
    #        self.figure = figure
    #    else:
    #        # Import matplotlib there?
    #        self.figure = plt.figure(figsize=(24, 24))

    def __init__(self, plt, estimator, X, features_axis, feature_names=None):
        self.X = X
        self.plt = plt
        self.x_axis_index, self.y_axis_index = features_axis
        if feature_names != None:
            self.x_name = feature_names[self.x_axis_index]
            self.y_name = feature_names[self.y_axis_index]
        self.figure = plt.figure(figsize=(24, 24))
        self.axes = figure.add_subplot(111)
        self.cluster_labels = estimator.labels_
        self.cluster_centres = estimator.cluster_centers_
        #self.cluster_numbers = [str(i + 1) for i in range(len(self.cluster_centres))]
        self.cluster_numbers = list(range(len(self.cluster_centres)))
        #self.text_cluster_labels = [str(i + 1) for i in range(len(self.cluster_centres))]
        self.text_cluster_labels = [str(i + 1) for i in self.cluster_numbers]
        self.markers = []
        for m in lines.Line2D.markers:
            try:
                if len(m) == 1 and m != ' ':
                    self.markers.append(m)
            except TypeError:
                pass

    def draw():
        for i, text_label in zip(self.cluster_numbers, self.text_cluster_labels):
            bool_cluster_labels = (self.cluster_labels == i)
            self.axes.scatter(
                    X[bool_cluster_labels, self.x_axis_index],
                    X[bool_cluster_labels, self.y_axis_index],
                    s=200,
                    c='lightgreen',
                    #marker='s',
                    marker=self.marker[i],
                    label='cluster {}'.format(text_label)
                )

        axes.scatter(
                self.cluster_centers_[:, self.x_axis_index],
                self.cluster_centers_[:, self.y_axis_index],
                s=300,
                marker='*',
                c='red',
                label='centroids'
            )

        for text_cluster_label, x, y in zip(
            text_cluster_labels,
            self.cluster_centers[:, self.x_axis_index],
            self.cluster_centers[:, self.y_axis_index]):
            axes.annotate(
                text_cluster_label,
                xy=(x, y),
                xytext=(-20, 20),
                textcoords='offset points',
                ha='right',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
                fontsize=24
            )
