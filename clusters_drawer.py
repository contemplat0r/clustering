import itertools
import matplotlib.pyplot as plt
import matplotlib.lines as lines


class ClustersDrawer(object):
    
    DEFAULT_FONT_SIZE = 24
    DEFAULT_FIGURE_SIZE = (24, 24)
    DEFAULT_DATA_POINT_SIZE = 200
    DEFAULT_CLUSTER_CENTERS_SIZE = 300
    DEFAULT_COLOR_STEP = 40
    DEFAULT_CMAP_NAME = 'gnuplot'
    DEFAULT_CLUSTER_CENTERS_COLOR='red'
    DEFAULT_CLUSTER_CENTERS_MARKER='*'
    DEFAULT_CLUSTER_CENTERS_TEXT_LABEL='centroids'
    
    def __init__(
            self,
            plt,
            estimator,
            X,
            features_axis,
            features_names=None,
            fontsize=None,
            figsize=None,
            data_point_size=None,
            color_step=None,
            cmap_name=None,
            data_colors=None,
            cluster_labels=None,
            cluster_centers_size=None,
            cluster_centers_color=None,
            cluster_centers_marker=None,
            cluster_centers_text_label=None
        ):
        self._X = X
        self._plt = plt
        self._features_axis = features_axis
        self._proections_indexses = self.pair_list(features_axis)
        if features_names != None:
            self._features_names = features_names
        if figsize:
            self._figsize = figsize
        else:
            self._figsize = self.DEFAULT_FIGURE_SIZE
        if data_point_size:
            self._data_point_size = data_point_size
        else:
            self._data_point_size = self.DEFAULT_DATA_POINT_SIZE
        if cluster_centers_size:
            self._cluster_centers_size = cluster_centers_size
        else:
            self._cluster_centers_size = self.DEFAULT_CLUSTER_CENTERS_SIZE
        if cluster_centers_marker:
            self._cluster_centers_marker = cluster_centers_marker
        else:
            self._cluster_centers_marker = self.DEFAULT_CLUSTER_CENTERS_MARKER
        if cluster_centers_color:
            self._cluster_centers_color = cluster_centers_color
        else:
            self._cluster_centers_color = self.DEFAULT_CLUSTER_CENTERS_COLOR
        if cluster_centers_text_label:
            self._cluster_centers_text_label = cluster_centers_text_label
        else:
            self._cluster_centers_text_label = self.DEFAULT_CLUSTER_CENTERS_TEXT_LABEL
        if fontsize:
            self._fontsize = fontsize
        else:
            self._fontsize = self.DEFAULT_FONT_SIZE
        self._cluster_labels = estimator.labels_
        self._cluster_centers = estimator.cluster_centers_
        self._cluster_numbers = list(range(len(self._cluster_centers)))
        #self._number_of_clusters = len(self._cluster_centers)
        self._text_cluster_labels = [str(i + 1) for i in self._cluster_numbers]
        #self._text_cluster_labels = (str(i + 1) for i in range(self._number_of_clusters))
        self._markers = []
        for m in lines.Line2D.markers:
            try:
                if len(m) == 1 and m != ' ' and m != '.' and m != ',':
                    self._markers.append(m)
            except TypeError:
                pass
        if color_step:
            self._color_step = color_step
        else:
            self._color_step = self.DEFAULT_COLOR_STEP
        if cmap_name:
            self._cmap = plt.get_cmap(cmap_name)
        else:
            self._cmap = plt.get_cmap(self.DEFAULT_CMAP_NAME)
        if data_colors:
            self._colors = data_colors
        else:
            self._colors = [self._cmap(i * self._color_step) for i in self._cluster_numbers]
        '''
        self._colors = (
                self._cmap(i) for i in range(
                    0,
                    self._color_step,
                    self._number_of_clusters * self._color_step
                )
            )
        '''

    def pair_list(self, lst):
        return itertools.combinations(lst, 2)
    
    def draw_2d_proection(self, axes, x_axis_index, y_axis_index):
        self._draw_data(axes, x_axis_index, y_axis_index)
        self._draw_cluster_centers(axes, x_axis_index, y_axis_index)
        self._draw_cluster_centers_labels(axes, x_axis_index, y_axis_index)
        axes.legend(loc='best', prop={'size': self._fontsize})
        if self._features_names:
            axes.set_xlabel(self._features_names[x_axis_index], fontsize=14)
            axes.set_ylabel(self._features_names[y_axis_index], fontsize=14)
        axes.grid()
        self._plt.show()

    def draw(self):
        for i, proection_indexes in enumerate(self._proections_indexses):
            figure = plt.figure(figsize=self._figsize)
            axes = figure.add_subplot(111)
            x_axis_index, y_axis_index = proection_indexes
            print("\t", i)
            self.draw_2d_proection(axes, x_axis_index, y_axis_index)

    def _draw_data(self, axes, x_axis_index, y_axis_index):
        for i, color, text_label in zip(self._cluster_numbers, self._colors, self._text_cluster_labels):
            bool_cluster_labels = (self._cluster_labels == i)
            axes.scatter(
                    self._X[bool_cluster_labels, x_axis_index],
                    self._X[bool_cluster_labels, y_axis_index],
                    s=self._data_point_size,
                    c=color,
                    marker=self._markers[i],
                    label='cluster {}'.format(text_label)
                )

    def _draw_cluster_centers(self, axes, x_axis_index, y_axis_index):
        axes.scatter(
                self._cluster_centers[:, x_axis_index],
                self._cluster_centers[:, y_axis_index],
                s=self._cluster_centers_size,
                marker=self._cluster_centers_marker,
                c=self._cluster_centers_color,
                label=self._cluster_centers_text_label
            )

    def _draw_cluster_centers_labels(self, axes, x_axis_index, y_axis_index):
        for text_cluster_label, x, y in zip(
            self._text_cluster_labels,
            self._cluster_centers[:, x_axis_index],
            self._cluster_centers[:, y_axis_index]):
            axes.annotate(
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
    def cluster_centers_color(self):
        return self._cluster_centers_color

    @cluster_centers_color.setter
    def cluster_centers_color(self, color_name):
        self._cluster_centers_color = color_name

    @cluster_centers_color.deleter
    def cluster_centers_color(self):
        raise ValueError("Invalid operation")

    @property
    def cluster_centers_marker(self):
        return self._cluster_centers_marker

    @cluster_centers_marker.setter
    def cluster_centers_marker(self, marker):
        self._cluster_centers_marker = marker

    @cluster_centers_marker.deleter
    def cluster_centers_marker(self):
        raise ValueError("Invalid operation")

    @property
    def cluster_centers_text_label(self):
        return self._cluster_centers_text_label

    @cluster_centers_text_label.setter
    def cluster_centers_text_label(self, text_label):
        self._cluster_centers_text_label = text_label


    @cluster_centers_text_label.deleter
    def cluster_centers_text_label(self):
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

    @property
    def data_markers(self):
        return self._markers

    @data_markers.setter
    def data_markers(self, markers_list):
        self._markers = markers_list

    @data_markers.deleter
    def data_markers(self):
        raise ValueError("Invalid operation")

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors_list):
        self._colors = colors_list

    @colors.deleter
    def colors(self):
        raise ValueError("Invalid operation")

    @property
    def colormap(self):
        return self._cmap

    @colormap.setter
    def colormap(self, cmapname):
        self._cmap = plt.get_cmap(cmapname)
        self._colors = [self._cmap(i * self._color_step) for i in self._cluster_numbers]

    @colormap.deleter
    def colormap(self):
        raise ValueError("Invalid operation")

    @property
    def colorstep(self):
        return self._color_step

    @colorstep.setter
    def colorstep(self, color_step):
        self._color_step = color_step
        self._colors = [self._cmap(i * self._color_step) for i in self._cluster_numbers]

    @colorstep.deleter
    def colorstep(self):
        raise ValueError("Invalid operation")
