from mpi4py import MPI

#import teca
from teca import *
import numpy as np
import datetime as dt
import sys

def gaussian_weights(
    ndays = 2,
    interval = 0.25,
    highpass = True,
    expansion_factor = 12,
    ):
    """ Calculates gaussian weights for a high-pass filter with a FWHM damping at frequencies lower than a period of ndays.

        input:
        ------

            ndays               : the cutoff period of the filter (2*pi/ndays is the FWHM cutoff frequency)

            interval            : the interval of the data

            highpass            : flag whether to invert the filter to be a high-pass filtered

            expansion_factor    : for a gaussian of width $\sigma$ and expansion factor $N$, the window 
                                  width is $N \sigma$.  This should be large enough that the gaussian
                                  approximately goes to zero by the end of the window.
            
        output:
        -------

            Returns a symmetric, normalized set of weights suitable for use in a convolution filter.
    
    """
    # determine the filter width (sigma) in units of number of points
    filter_sigma = (ndays/interval)/(2*np.pi)


    # calculate the width of the filter in number of points
    filter_width_points = int(np.ceil((expansion_factor*filter_sigma)))
    # make sure the filter has an odd number of points
    if filter_width_points % 2 == 0:
        filter_width_points += 1


    # set the index points of the gaussian
    nmid = int((filter_width_points-1)/2)
    x = np.array(range(filter_width_points)) - nmid

    # calculate the gaussian
    gauss = np.exp(-(x**2)/(2*filter_sigma**2))

    # normalize the weights
    gauss /= gauss.sum()

    if highpass:
        # invert the filter to turn it into a high-pass
        gauss = -gauss
        gauss[nmid] += 1

    # return
    return gauss

def uniform_weights(
    ndays = 2,
    interval = 0.25,
    highpass = True,
    ):
    """ Calculates uniform weights for a high-pass filter with a FWHM damping at frequencies lower than a period of ndays.

        input:
        ------

            ndays               : the width of the filter

            interval            : the interval of the data

            highpass            : flag whether to invert the filter to be a high-pass filtered
                        
            
        output:
        -------

            Returns a symmetric, normalized set of weights suitable for use in a convolution filter.
    
    """
    # calculate the width of the filter in number of points
    filter_width_points = int(np.ceil((ndays/interval)))
    # make sure the filter has an odd number of points
    if filter_width_points % 2 == 0:
        filter_width_points += 1

    # set the weights
    weights = np.ones(filter_width_points)

    # normalize the weights
    weights /= weights.sum()

    if highpass:
        # invert the filter to turn it into a high-pass
        weights = -weights
        nmid = int((filter_width_points-1)/2)
        weights[nmid] += 1

    # return
    return weights



class teca_convolution_filter(teca_python_algorithm):

    postfix = "_filtered"
    filter_cutoff = 5
    weight_type = "highpass_gaussian"
    valid_weight_types = ["highpass_gaussian", "highpass_uniform", "lowpass_gaussian", "lowpass_uniform"]
    window_length = None
    weights = None
    point_arrays = None
    verbose = 0

    def set_verbose(self, verbose):
        """ Flags whether to be verbose. """
        self.verbose = verbose

    def get_verbose(self):
        """ Returns the verbosity flag. """
        return self.verbose

    def vprint(self, *msg):
        """ Prints only if the verbose flag is set. """
        if self.get_verbose():
            print(*msg)

    def set_point_arrays(self, arrays):
        """ Sets the list of arrays to filter. """
        if type(arrays) is str:
            arrays = [arrays]

        self.point_arrays = arrays

        return

    def get_point_arrays(self):
        """ Gets the list of arrays to filter. """
        return self.point_arrays

    def set_postfix(self, postfix):
        """ Sets the postfix for the downstream variable(s). """
        self.postfix = postfix

    def get_postfix(self):
        """ Gets the postfix for the downstream variable(s). """
        return self.postfix

    def get_point_array_names(self):
        """" Gets the output point array names for the downstream variables. """
        point_array_names = []
        for var in self.point_arrays:
            point_array_names.append(f"{var}{self.get_postfix()}")

        return point_array_names


    def set_weight_type(self, weight_type):
        """ Sets the weight type (see teca_convolution_filter.valid_weight_types). """
        if weight_type.lower() not in self.valid_weight_types:
            raise NotImplementedError(f"Weight type {weight_type.lower()} has not been implemented.  Valid weight types are {self.valid_weight_types}")

        self.weight_type = weight_type.lower()

    def get_weight_type(self):
        """ Returns the weight type """
        return self.weight_type

    def set_filter_cutoff(self, cutoff):
        """" Sets the cutoff period of the filter (in units of days). """
        self.filter_cutoff = cutoff

    def get_filter_cutoff(self):
        """ Returns the filter cutoff period (in units of days). """
        return self.filter_cutoff

    def report(self, o_port, reports_in):
        # add the variable we produce to the report
        rep = teca_metadata(reports_in[0])

        # extract the temporal resolution of the data and convert it to days
        delta_t_str = str(rep["attributes"]["time"]["delta_t"]).split()[1]
        _tmpdate = dt.datetime.strptime(delta_t_str,"%H:%M:%S")
        delta_t = dt.timedelta(hours=_tmpdate.hour, minutes=_tmpdate.minute, seconds=_tmpdate.second)
        delta_t_days = delta_t.seconds/86400

        # get the filter type
        weight_type = self.get_weight_type()
        filter_type = weight_type.split("_")[1]
        highpass = "highpass" in weight_type

        # get the filter width
        ndays = self.get_filter_cutoff()

        # get the weights
        if filter_type == "gaussian":
            weight_function = gaussian_weights
        elif filter_type == "uniform":
            weight_function = uniform_weights
        else:
            sys.stderr.write(f"ERROR: unsupported weight type {weight_type}\n")
            return rep

        # get the weights
        self.weights = weight_function(
                ndays,
                interval = delta_t_days,
                highpass = highpass
                )

        self.window_length = len(self.weights)

        # print a status message
        self.vprint(f"teca_convolution_filter:report(): set up a {weight_type} filter with {self.window_length} weights:\n{self.weights}")

        # get the list of variables and attributes
        input_variables = rep["variables"]
        input_attributes = rep["attributes"]

        # initialize the list of output variable names and attributes
        output_names = list(input_variables)
        output_atts = teca_metadata()

        # set the list of arrays to work on
        arrays = self.get_point_arrays()

        # remove upstream variables
        for var in arrays:
            # remove the upstream variable from the variable list
            output_names.remove(var)

        if arrays is None:
            sys.stderr.write("ERROR: point arrays have not been set; set_point_arrays() needs to be called.\n")
            return rep

        # copy the metadata of dimension variables and other upstream vars that aren't being removed
        for var in output_names:
            # set the metadata
            output_atts[var] = input_attributes[var]

        # set the output variable attributes
        postfix = self.get_postfix()
        for var in arrays:
            # set the output variable name
            output_name = f"{var}{postfix}"
            output_names.append(output_name)

            # set the metadata
            output_atts[output_name] = input_attributes[var]

        # report on the arrays we are creating
        self.vprint(f"teca_convolution_filter:report(): creating new arrays {output_names}")

        # remove the requested arrays from the report
        rep["variables"] = output_names
        rep["attributes"] = output_atts

        return rep

    def request(self, o_port, md_in, request_in):

        # get the input metadata
        md = md_in[0]

        # get the key for the timestep variable
        request_key_in = request_in["index_request_key"]
        # get the current timestep request
        req_id = request_in[request_key_in]

        # report on the arrays we are creating
        self.vprint(f"teca_convolution_filter:request(): working on timestep {req_id}")

        # get the key for the timestep variable in the metadata (should be the same)
        request_key = md["index_request_key"]

        # get the number of timesteps
        init_key = md["index_initializer_key"]
        num_timesteps = md[init_key]

        # request the number of indicies required to run the filter
        nmid = (self.window_length - 1)/2
        if req_id < nmid:
            req_len = req_id*2 + 1

            # report on the arrays we are creating
            self.vprint(f"teca_convolution_filter:request(): left boundary adjustment; changing the window length from {self.window_length} to {req_len}")
        elif req_id > (num_timesteps - 1 - nmid):
            nleft = num_timesteps - 1 - req_id
            req_len = nleft*2 + 1
            # report on the arrays we are creating
            self.vprint(f"teca_convolution_filter:request(): right boundary adjustment; changing the window length from {self.window_length} to {req_len}")
        else:
            req_len = self.window_length

        # recalculate the midpoint offset with the new window length
        nmid = int((req_len - 1)/2)
        
        # request a symmetric set of indices surrounding the current request id
        indices = list(range(req_id-nmid, req_id + nmid + 1))

        up_req = []
        for i in indices:
            req = teca_metadata(request_in)
            req['arrays'] = self.get_point_arrays()
            req[request_key] = i
            up_req.append(req)
        return up_req

    def execute(self, o_port, data_in, request_in):
        """ Implements a convolution-based temporal filter. """

        # get the number of timesteps available
        ntime = len(data_in)

        # get the weights
        weights = np.array(self.weights)

        # determine if we need to reduce the length of the weights
        if ntime < self.window_length:
            # get the difference in weight sizes
            size_difference = self.window_length - ntime

            # check that the number is even
            if size_difference % 2 != 0:
                sys.stderr.write("ERROR: The request length is not an odd number.")
                raise RuntimeError("The request length must be an odd number")

            # remove points from the beginning and end of the weights
            nremove = int(size_difference/2)
            weights = weights[nremove:-nremove]

            # renormalize the weights
            if "highpass" in self.get_weight_type():
                # change to lowpass, renormalize, and re-invert
                # change the weights back to low pass
                nmid = int((ntime - 1) / 2)
                weights[nmid] -= 1
                weights = -weights

                # renormalize the weights
                weights /= weights.sum()

                # change back to high pass
                weights = -weights
                weights[nmid] += 1
            else:
                weights /= weights.sum()


            # report on the arrays we are creating
            self.vprint(f"teca_convolution_filter:execute(): boundary adjustment; new weights are {weights}")

        # build the output mesh
        in_mesh = as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        output_arrays = out_mesh.get_point_arrays()

        data_out = []

        # apply the weighted sum
        for var in self.get_point_arrays():
            for n in range(ntime):
                # get the input array
                in_mesh = as_teca_cartesian_mesh(data_in[n])

                # get the array
                in_arrays = in_mesh.get_point_arrays()

                # set the output variable name
                outvar = f"{var}{self.get_postfix()}"

                # updated the weighted sum
                if n == 0:
                    output_arrays[outvar] = weights[n] * in_arrays[var]
                else:
                    output_arrays[outvar] += weights[n] * in_arrays[var]

                # remove the upstream variable from the output mesh
                #del(output_arrays[var])

        return out_mesh

if __name__ == "__main__":

    regex = "/N/project/obrienta_startup/regcm_input_data/NNRP1/2021/air.*\.nc"
    be_verbose = True


    reader = teca_cf_reader.New()
    reader.set_verbose(int(be_verbose))
    reader.set_z_axis_variable("level")
    reader.set_files_regex(regex)

    filter = teca_convolution_filter.New()
    filter.set_point_arrays("air")
    filter.set_verbose(int(be_verbose))
    filter.set_input_connection(reader.get_output_port())

    # executive
    exe = teca_index_executive.New()
    exe.set_verbose(int(be_verbose))

    # set up the writer
    writer = teca_cf_writer.New()
    writer.set_executive(exe)
    writer.set_verbose(int(be_verbose))
    writer.set_thread_pool_size(1)
    writer.set_point_arrays(filter.get_point_array_names())
    writer.set_input_connection(filter.get_output_port())
    #writer.set_point_arrays(["air"])
    #writer.set_input_connection(reader.get_output_port())
    writer.set_file_name("test_%t%.nc")

    # TODO: remove these lines
    #writer.set_first_step(0)
    #writer.set_last_step(120)

    writer.update()
