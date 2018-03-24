import shapefile
import numpy

rus_adm_units = shapefile.Reader('unzip/RUS_adm1')

rus_adm_units_records = rus_adm_units.records()
rus_adm_units_shapes = rus_adm_units.shapes()
print(type(rus_adm_units_shapes))
print(type(rus_adm_units_shapes[0]))
print(dir(rus_adm_units_shapes[0]))
print(rus_adm_units_shapes[0].bbox)
print(rus_adm_units_shapes[0].parts)
#print(rus_adm_units_shapes[0].points)
print(rus_adm_units_shapes[0].shapeType)

points = numpy.array(rus_adm_units_shapes[0].points)
print(points.shape)
x_coords = points[:, 0]
y_coords = points[:, 1]
print(x_coords.shape)
print(y_coords.shape)
