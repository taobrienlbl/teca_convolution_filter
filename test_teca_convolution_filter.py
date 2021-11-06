
from mpi4py import MPI # pre-initialize MPI
import teca
import teca_convolution_filter

regex = "./linked_inputs/e5.*\.nc"
output_filename = "test_%t%.nc"
x_axis_variable = "longitude"
y_axis_variable = "latitude"
z_axis_variable = "level"
variable = "T"
be_verbose = True
temporal_resolution = 1/24

# add a reader
reader = teca.teca_cf_reader.New()
reader.set_verbose(int(be_verbose))
reader.set_x_axis_variable(x_axis_variable)
reader.set_y_axis_variable(y_axis_variable)
reader.set_z_axis_variable(z_axis_variable)
reader.set_files_regex(regex)

# add the convolution filter
filter = teca_convolution_filter.teca_convolution_filter.New()
filter.set_point_arrays(variable)
filter.set_verbose(int(be_verbose))
filter.set_temporal_resolution(temporal_resolution)
filter.set_input_connection(reader.get_output_port())

# executive
exe = teca.teca_index_executive.New()
exe.set_verbose(int(be_verbose))

# set up the writer
writer = teca.teca_cf_writer.New()
writer.set_executive(exe)
writer.set_verbose(int(be_verbose))
writer.set_thread_pool_size(1)
writer.set_point_arrays(filter.get_point_array_names())
writer.set_input_connection(filter.get_output_port())
writer.set_file_name(output_filename)

# run the pipeline
writer.update()