import matplotlib.pyplot as plt
import matplotlib.lines as lines


class ClustersDrawer(object):
    
    def __init__(self, plt, estimator, X, features_axis, feature_names=None):
        self._X = X
        self._plt = plt
        self._x_axis_index, self._y_axis_index = features_axis
        if feature_names != None:
            self._x_name = feature_names[self.x_axis_index]
            self._y_name = feature_names[self.y_axis_index]
        self._figsize = (24, 24)
        self._data_point_size = 200
        self._cluster_centers_size = 300
        self._fontsize = 24
        self._figure = plt.figure(figsize=self._figsize)
        self._axes = figure.add_subplot(111)
        self._cluster_labels = estimator.labels_
        self._cluster_centers = estimator.cluster_centers_
        self._cluster_numbers = list(range(len(self.cluster_centers)))
        self._text_cluster_labels = [str(i + 1) for i in self.cluster_numbers]
        self._markers = []
        for m in lines.Line2D.markers:
            try:
                if len(m) == 1 and m != ' ':
                    self._markers.append(m)
            except TypeError:
                pass

    def draw(self):
        self._draw_data()
        self._draw_cluster_centers()
        self._draw_cluster_centers_labels()

    def _draw_data(self):
        for i, text_label in zip(self._cluster_numbers, self._text_cluster_labels):
            bool_cluster_labels = (self._cluster_labels == i)
            self._axes.scatter(
                    self._X[bool_cluster_labels, self._x_axis_index],
                    self._X[bool_cluster_labels, self._y_axis_index],
                    s=self._data_point_size,
                    c='lightgreen',
                    marker=self._marker[i],
                    label='cluster {}'.format(text_label)
                )
    
    def _draw_cluster_centers(self):
        self._axes.scatter(
                self._cluster_centers[:, self._x_axis_index],
                self._cluster_centers[:, self._y_axis_index],
                s=self._cluster_centers_size,
                marker='*',
                c='red',
                label='centroids'
            )

    def _draw_cluster_centers_labels(self):
        for text_cluster_label, x, y in zip(
            self._text_cluster_labels,
            self._cluster_centers[:, self._x_axis_index],
            self._cluster_centers[:, self._y_axis_index]):
            self._axes.annotate(
                text_cluster_label,
                xy=(x, y),
                xytext=(-20, 20),
                textcoords='offset points',
                ha='right',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
                fontsize=self._fontsize
            )

    @property
    def figsize(self):
        return self._figsize

    @figsize.setter
    def figsize(self, width, height):
       self._figsize = (width, height)

    @figsize.deleter
    def figsize(self):
        raise ValueError("Invalid operation")

    @property
    def data_point_size(self):
        return self._data_point_size

    @data_point_size.setter
    def data_point_size(self, size):
        self._data_point_size = size

    @data_point_size.deleter
    def data_point_size(self):
        raise ValueError("Invalid operation")

    @property
    def cluster_centers_size(self):
        return self._cluster_centers_size

    @cluster_centers_size.setter
    def cluster_centers_size(self, size):
        self._cluster_centers_size = size

    @cluster_centers_size.deleter
    def cluster_centers_size(self):
        raise ValueError("Invalid operation")

    @property
    def fontsize(self):
        return self._fontsize

    @fontsize.setter
    def fontsize(self, size):
        self._fontsize = fontsize

    @fontsize.deleter
    def fontsize(self):
        raise ValueError("Invalid operation")
