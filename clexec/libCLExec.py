import random
import yaml
import numpy as np
import pyopencl as cl
import pyopencl.cltypes
from tabulate import tabulate

cl_type_to_numpy_type = {
    "bool": np.bool,
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "long": np.int64,
    "ulong": np.uint64,
    "float": np.float32,
    "double": np.float64,
    "half": np.float16
}

name_to_cl_vector_type = {
    "float4" : cl.cltypes.float4,
    "float3" : cl.cltypes.float3,
    "float2" : cl.cltypes.float2,
    "int4" : cl.cltypes.int4,
    "int3" : cl.cltypes.int3,
    "int2" : cl.cltypes.int2,
    "uint4":cl.cltypes.uint4,
    "uint3":cl.cltypes.uint3,
    "uint2":cl.cltypes.uint2
}


def read_file(file_name):
    f = open(file_name, "r")
    result = []
    for line in f:
        if line[-1] == "\n" and len(line) > 1:
            result.append(line[:-1])
        else:
            result.append(line)
    f.close()
    return result


def read_file_as_string(file_name):
    f = open(file_name, 'r')
    s = "".join(f.readlines())
    return s


def write_array_to_file(array, filename):
    f = open(filename, 'w')
    f.write(str(len(array)) + "\n")
    for i in array:
        f.write(str(i) + "\n")
    f.close()


def write_variable_to_file(variable, filename):
    f = open(filename, 'w')
    f.write("1\n" + str(variable) + "\n")
    f.close()


def write_image_to_file(array, filename):
    f = open(filename, 'w')
    f.write("1" + "\n")
    for i in array:
        f.write(str(i) + "\n")
    f.close()


def compare2file(file1_name,file2_name):
    num = 0
    diff = 0
    with open(file1_name) as file01:
        for line in file01:
            num = num+1

    with open(file1_name,'r') as x ,open(file2_name,'r') as y:
        line1=x.readlines()
        line2=y.readlines()
        output_filename = file1_name +"_compare_with_"+file2_name+".txt"
        for i in range(num):
            if line1[i]==line2[i]:
                output_content = str(i+1)+":same"
                #print(output_content)
                #if(save):
                #    write_variable_to_file(output_content,output_filename)
                #continue
            else:
                var=str(i+1)
                #print(file1_name)
                #print(file2_name)
                output_content = "Line "+var+" is different.\nFile_1:"+ str(line1[i])+ "File_2:"+str(line2[i])
                #write_variable_to_file(output_content,output_filename)
                diff = 1
                #print(output_content)
        return diff


def read_array_from_file(filename, data_type):
    if type(data_type) == str:
        data_type = cl_type_to_numpy_type[data_type]
    f = open(filename, 'r')
    size = int(f.readline())
    array = []
    for i in range(size):
        line = f.readline()
        if line[-1] == "\n" and len(line) > 1:
            array.append(data_type(line[:-1]))
        else:
            array.append(data_type(line))
    f.close()
    return array


def read_float_vector_array_from_file(filename, data_type):
    f = open(filename, 'r')
    size = int(f.readline())
    array = np.empty((size,), dtype=name_to_cl_vector_type[data_type])
    for i in range(size):
        line = f.readline()
        line_1 =line[1:-2].split(", ")

        for j in range(len(line_1)):
            line_1[j] = np.float32((line_1[j]))
        line_1 = tuple(line_1)
        array[i] = line_1
    #print(array)
    f.close()
    return array


def read_int_vector_array_from_file(filename, data_type):
    f = open(filename, 'r')
    size = int(f.readline())
    array = np.empty((size,), dtype=name_to_cl_vector_type[data_type])
    for i in range(size):
        line = f.readline()
        line_1 =line[1:-2].split(", ")

        for j in range(len(line_1)):
            line_1[j] = np.int32((line_1[j]))
        line_1 = tuple(line_1)
        array[i] = line_1
    #print(array)
    f.close()
    return array


def output_diff(ref_list, cmp_list, absolute_tolerance=.0, relative_tolerance=.0):
    if ref_list.shape != cmp_list.shape:
        return False
    abstol = absolute_tolerance * max([abs(x) for x in ref_list])
    for (r, c) in zip(ref_list, cmp_list):
        diff = abs(r - c)
        if not diff <= abstol or diff < relative_tolerance * abs(r):
            return False
    return True


def validate_output(outputs, absolute_tolerance=.0, relative_tolerance=.0):
    results = []
    num_output = len(outputs)
    if num_output < 2:
        return [False]
    expected_output = outputs[0]
    for i in range(1, num_output):
        result = output_diff(expected_output, outputs[i], absolute_tolerance, relative_tolerance)
        results.append(result)
    return results


def read_struct_array_from_file(filename, data_name_array, data_type_array):
    #for i in range(len(data_type_array)):
    #    data_type_array[i] = cl_type_to_numpy_type[data_type_array[i]]
    f = open(filename, 'r')
    size = int(f.readline())
    my_struct_list = []
    for i in range(len(data_type_array)):
        t = (data_name_array[i],cl_type_to_numpy_type[data_type_array[i]])
        my_struct_list.append(t)
    #my_struct_list = [("field1", np.int32), ("field2", np.float32), ("field3", np.float32)]
    my_struct = np.dtype(my_struct_list)
    array = np.empty(size, my_struct)

    for i in range(len(data_type_array)):
        data_type_array[i] = cl_type_to_numpy_type[data_type_array[i]]
    if len(data_type_array) == 1:
        for i in range(size):
            line = f.readline()
            line_1 =line[1:-3].split()
            for j in range(len(line_1)):
                line_1[j] = data_type_array[j](line_1[j])
            line_1 = tuple(line_1)
            #print(line_1)
            array[i] = line_1
    else:
        for i in range(size):
            line = f.readline()
            line_1 =line[1:-2].split(", ")
            for j in range(len(line_1)):
                line_1[j] = data_type_array[j](line_1[j])
            line_1 = tuple(line_1)
        #print(line_1)
            array[i] = line_1
    #print(array)
    #print(array.shape)
    f.close()
    return array


def read_variable_from_file(filename, data_type):
    if type(data_type) == str:
        data_type = cl_type_to_numpy_type[data_type]
    f = open(filename, 'r')
    f.readline()
    line = f.readline()
    if line[-1] == "\n" and len(line) > 1:
        return data_type(line[:-1])
    else:
        return data_type(line)


def produce_full_range_rand_value(type_str):
    lower_bound = 0
    upper_bound = 0
    if type_str == 'bool':
        return random.choice([True, False])
    elif type_str == 'char':
        lower_bound = - 2 ** 7
        upper_bound = 2 ** 7 - 1
    elif type_str == 'uchar':
        lower_bound = 0
        upper_bound = 2 ** 8 - 1
    elif type_str == 'short':
        lower_bound = -2 ** 15
        upper_bound = 2 ** 15 - 1
    elif type_str == 'ushort':
        lower_bound = 0
        upper_bound = 2 ** 16 - 1
    elif type_str == 'int':
        lower_bound = - 2 ** 31
        upper_bound = 2 ** 31 - 1
    elif type_str == 'uint':
        lower_bound = 0
        upper_bound = 2 ** 32 - 1
    elif type_str == 'long':
        lower_bound = - 2 ** 63
        upper_bound = 2 ** 63 - 1
    elif type_str == 'ulong':
        lower_bound = 0
        upper_bound = 2 ** 64 - 1
    if type_str != 'float' and type_str != 'double' and type_str != 'halt':
        return random.randint(lower_bound, upper_bound)
    elif type_str == 'float':
        lower_bound = 3.4E-38
        upper_bound = 3.4E38
    elif type_str == 'double':
        lower_bound = 1.7E-308
        upper_bound = 1.7E308
    elif type_str == 'half':
        lower_bound = 3.2E-10
        upper_bound = 3.2E10
    return random.uniform(lower_bound, upper_bound)


def produce_noinit_array(type_str, size):
    if type_str == 'bool':
        return [False] * size
    elif type_str == 'char':
        return ['\0'] * size
    elif type_str == 'uchar':
        return [0] * size
    elif type_str == 'short':
        return [0] * size
    elif type_str == 'ushort':
        return [0] * size
    elif type_str == 'int':
        return [0] * size
    elif type_str == 'uint':
        return [0] * size
    elif type_str == 'long':
        return [0] * size
    elif type_str == 'ulong':
        return [0] * size
    else:
        return [0.0] * size


def produce_rand_array(type_str, length):
    result = []
    for i in range(length):
        result.append(produce_rand_value(type_str))
    return result


char_low_bound = - 2 ** 7
char_high_bound = 2 ** 7 - 1
uchar_low_bound = 0
uchar_high_bound = 2 ** 8 - 1
short_low_bound = -2 ** 15
short_high_bound = 2 ** 15 - 1
ushort_low_bound = 0
ushort_high_bound = 2 ** 16 - 1
int_low_bound = - 2 ** 31
int_high_bound = 2 ** 31 - 1
uint_low_bound = 0
uint_high_bound = 2 ** 32 - 1
long_low_bound = - 2 ** 63
long_high_bound = 2 ** 63 - 1
ulong_low_bound = 0
ulong_high_bound = 2 ** 64 - 1
float_low_bound = -3.4E10
float_high_bound = 3.4E10
double_low_bound = -1.7E20
double_high_bound = 1.7E20
half_low_bound = -3.2E5
half_high_bound = 3.2E5


def produce_rand_value(type_str):
    if type_str == 'bool':
        return random.choice([True, False])
    elif type_str == 'char':
        return random.randint(char_low_bound, char_high_bound)
    elif type_str == 'uchar':
        return random.randint(uchar_low_bound, uchar_high_bound)
    elif type_str == 'short':
        return random.randint(short_low_bound, short_high_bound)
    elif type_str == 'ushort':
        return random.randint(ushort_low_bound, ushort_high_bound)
    elif type_str == 'int':
        return random.randint(int_low_bound, int_high_bound)
    elif type_str == 'uint':
        return random.randint(uint_low_bound, uint_high_bound)
    elif type_str == 'long':
        return random.randint(long_low_bound, long_high_bound)
    elif type_str == 'ulong':
        return random.randint(ulong_low_bound, ulong_high_bound)
    elif type_str == 'float':
        return random.uniform(float_low_bound, float_high_bound)
    elif type_str == 'double':
        return random.uniform(double_low_bound, double_high_bound)
    elif type_str == 'half':
        return random.uniform(half_low_bound, half_high_bound)

    
def produce_rand_value_custom_range(type_str, low, high):
    if type_str == 'bool':
        return random.choice([True, False])
    elif type_str == 'char':
        return random.randint(low, high)
    elif type_str == 'uchar':
        return random.randint(low, high)
    elif type_str == 'short':
        return random.randint(low, high)
    elif type_str == 'ushort':
        return random.randint(low, high)
    elif type_str == 'int':
        return random.randint(low, high)
    elif type_str == 'uint':
        return random.randint(low, high)
    elif type_str == 'long':
        return random.randint(low, high)
    elif type_str == 'ulong':
        return random.randint(low, high)
    elif type_str == 'float':
        return random.uniform(low, high)
    elif type_str == 'double':
        return random.uniform(low, high)
    elif type_str == 'half':
        return random.uniform(low, high)


def generate_random_array(size, type_str):
    if type_str == 'bool':
        return np.random.choice([True, False], size=size)
    elif type_str == 'char':
        return np.random.randint(low=char_low_bound, high=char_high_bound + 1, size=size)
    elif type_str == 'uchar':
        return np.random.randint(low=uchar_low_bound, high=uchar_high_bound + 1, size=size)
    elif type_str == 'short':
        return np.random.randint(low=short_low_bound, high=short_high_bound + 1, size=size)
    elif type_str == 'ushort':
        return np.random.randint(low=ushort_low_bound, high=ushort_high_bound + 1, size=size)
    elif type_str == 'int':
        return np.random.randint(low=int_low_bound, high=int_high_bound + 1, size=size)
    elif type_str == 'uint':
        return np.random.randint(low=uint_low_bound, high=uint_high_bound + 1, size=size)
    elif type_str == 'long':
        return np.random.randint(low=long_low_bound, high=long_high_bound + 1, size=size)
    elif type_str == 'ulong':
        return np.random.randint(low=ulong_low_bound, high=ulong_high_bound + 1, size=size, dtype=np.uint64)
    elif type_str == 'float':
        return np.random.uniform(float_low_bound, float_high_bound, size)
    elif type_str == 'double':
        return np.random.uniform(double_low_bound, double_high_bound, size)
    elif type_str == 'half':
        return np.random.uniform(half_low_bound, half_high_bound, size)


def generate_random_array_custom_range(size, type_str, low_, high_):
    if type_str == 'bool':
        return np.random.choice([True, False], size=size)
    elif type_str == 'char':
        return np.random.randint(low=low_, high=high_ + 1, size=size)
    elif type_str == 'uchar':
        return np.random.randint(low=low_, high=high_ + 1, size=size)
    elif type_str == 'short':
        return np.random.randint(low=low_, high=high_ + 1, size=size)
    elif type_str == 'ushort':
        return np.random.randint(low=low_, high=high_ + 1, size=size)
    elif type_str == 'int':
        return np.random.randint(low=low_, high=high_ + 1, size=size)
    elif type_str == 'uint':
        return np.random.randint(low=low_, high=high_ + 1, size=size)
    elif type_str == 'long':
        return np.random.randint(low=low_, high=high_ + 1, size=size)
    elif type_str == 'ulong':
        return np.random.randint(low=low_, high=high_ + 1, size=size, dtype=np.uint64)
    elif type_str == 'float':
        return np.random.uniform(low_, high_, size)
    elif type_str == 'double':
        return np.random.uniform(low_, high_, size)
    elif type_str == 'half':
        return np.random.uniform(low_, high_, size)


def parse_kernel_info(info_filename):
    yaml_reader = open(info_filename, 'r')
    kernel_info = yaml.safe_load(yaml_reader)
    yaml_reader.close()
    return kernel_info


def get_kernel_name_from_info(kernel_info):
    for item in list(kernel_info.keys()):
        if item not in ["Cov", "global", "local", "dim", "Barriers", "Branches", "Loops", "device_ID", "num_parameters",
                        "num_tests", "platform_ID", "structure_data_filename", "time_stamp"]:
            return item


def test_generation(target_kernel_name, kernel_info, num_tests, base_num=0):
    generated_tests = []
    target_kernel_info = kernel_info[target_kernel_name]
    for i in range(num_tests):
        test_id_str = str(i + base_num).zfill(6)
        current_test = {}
        for parameter in target_kernel_info:
            parameter_info = target_kernel_info[parameter]
            if parameter_info["cl_scope"] == "local":
                generated_input_data = []
            else:
                if parameter_info["fuzzing"] == "random":
                    if parameter_info["pointer"]:
                        if parameter_info["cl_type"] in cl_type_to_numpy_type:
                            generated_input_data = generate_random_array(parameter_info["size"], parameter_info["cl_type"])
                        elif parameter_info["cl_type"] in name_to_cl_vector_type:
                            generated_input_data = np.empty((parameter_info["size"],), dtype=name_to_cl_vector_type[parameter_info["cl_type"]])
                        else:
                            my_struct_list = []
                            types = []
                            for j in range((len(parameter_info)-10)//2):
                                t = (parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_name"],
                                cl_type_to_numpy_type[parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_type"]])
                                my_struct_list.append(t)
                                types.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_type"])
                            my_struct = np.dtype(my_struct_list)
                            generated_input_data = np.empty(parameter_info["size"], my_struct)
                            for array_id in range(parameter_info["size"]):
                                child_id = 0
                                for current_type in types:
                                    generated_input_data[array_id][child_id] = produce_rand_value_custom_range(current_type, 0, 100)
                                    child_id = child_id + 1
                    else:
                        if parameter_info["cl_type"] =="image2d_t":
                            generated_input_data = np.random.uniform(low=float_low_bound, high=float_high_bound, size=(1200, 1600, 4))
                        elif parameter_info["cl_type"] in name_to_cl_vector_type:
                            generated_input_data = np.empty((parameter_info["size"],), dtype=name_to_cl_vector_type[parameter_info["cl_type"]])
                        else:
                            generated_input_data = produce_rand_value(parameter_info["cl_type"])
                    #parameter_info["fuzzing"] = "generated_file"
                    #parameter_info["generated_file"] = target_kernel_name + "_" + parameter + "_" + kernel_info["time_stamp"] + "_"
                elif parameter_info["fuzzing"] == "init_file":
                    filename = parameter_info["init_file"]
                    if parameter_info["cl_type"] in cl_type_to_numpy_type:
                        array = read_array_from_file(filename, parameter_info["cl_type"])
                        generated_input_data = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])
                    elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or \
                            parameter_info["cl_type"] == "float2":
                        generated_input_data = read_float_vector_array_from_file(filename, parameter_info["cl_type"])
                    elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info[
                        "cl_type"] == "uint4":
                        generated_input_data = read_int_vector_array_from_file(filename, parameter_info["cl_type"])
                    # struct
                    else:
                        struct_data_type_array = []
                        struct_data_name_array = []
                        for j in range((len(parameter_info) - 10) // 2):
                            struct_data_name_array.append(
                                parameter_info[parameter_info["cl_type"] + "_key" + str(j) + "_name"])
                            struct_data_type_array.append(
                                parameter_info[parameter_info["cl_type"] + "_key" + str(j) + "_type"])
                        generated_input_data = read_struct_array_from_file(filename, struct_data_name_array, struct_data_type_array)
                elif parameter_info["fuzzing"] == "initial_value":
                    generated_input_data = parameter_info["initial_value"]
                else:
                    generated_input_data = np.asarray(produce_noinit_array(parameter_info["cl_type"],parameter_info["size"]))
            current_test[parameter] = generated_input_data
        file_name = target_kernel_name + "_test" + test_id_str
        generated_tests.append(file_name + ".npy")
        np.save(file_name, current_test)
    return generated_tests


def generate_random_schedule(global_size, local_size):
    total_num_groups = 1
    dimension = 0
    groups = []

    for i in range(3):
        if global_size[i] != 1:
            dimension = dimension + 1
            num_groups = global_size[i] / local_size[i]
            if global_size[i] % local_size[i] != 0:
                num_groups = num_groups + 1
            total_num_groups = total_num_groups * num_groups
            groups.append(int(num_groups))
        else:
            break
    total_num_groups = int(total_num_groups)
    groupIDMap = np.empty((total_num_groups,), dtype=cl.cltypes.uint3)
    if dimension == 1:
        for x in range(groups[0]):
            groupIDMap[x][0] = x
    elif dimension == 2:
        for y in range(groups[1]):
            for x in range(groups[0]):
                loc = y * groups[0] + x
                groupIDMap[loc][0] = x
                groupIDMap[loc][1] = y
    else:
        for z in range(groups[2]):
            for y in range(groups[1]):
                for x in range(groups[0]):
                    loc = x + groups[0] * (y + groups[1] * z)
                    groupIDMap[loc][0] = x
                    groupIDMap[loc][1] = y
                    groupIDMap[loc][2] = z
    np.random.shuffle(groupIDMap)
    return groupIDMap


def execute_kernel(target_kernel_name, kernel_filename, kernel_info, test_file_name, time_stamp, measure_cov=False, with_schedule=None):
    target_kernel_info = kernel_info[target_kernel_name]
    #test_id_str = str(test_id).zfill(6)
    arguments = {}
    real_num_parameters = kernel_info["num_parameters"]
    real_kernel_filename = kernel_filename
    if measure_cov:
        real_kernel_filename = kernel_filename[:-3] + "_cov.cl"
        if kernel_info["Barriers"] != 0:
            real_num_parameters = real_num_parameters + 1
        if kernel_info["Branches"] != 0:
            real_num_parameters = real_num_parameters + 1
        if kernel_info["Loops"] != 0:
            real_num_parameters = real_num_parameters + 1
    elif with_schedule is not None:
        real_kernel_filename = kernel_filename[:-3] + "_schedule.cl"
        real_num_parameters = real_num_parameters + 1

    ctx = cl.Context(dev_type=cl.device_type.GPU)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg_source = read_file_as_string(real_kernel_filename)
    prg = cl.Program(ctx, prg_source).build()

    kernel_scalar_arg_dtypes = [None] * real_num_parameters
    kernel = getattr(prg, target_kernel_name)

    #test_file_name = target_kernel_name + "_test" + str(test_id_str).zfill(6) + ".npy"
    test_input_data = np.load(test_file_name)

    for parameter in target_kernel_info:
        parameter_info = target_kernel_info[parameter]
        if parameter_info["cl_scope"] == "local":
            buffer = cl.LocalMemory(parameter_info["size"])
            arguments[int(parameter_info["pos"])] = {"type": "array", "buffer": buffer}
        else:
            if parameter_info["pointer"]:
                if parameter_info["fuzzing"] != "noinit":
                    nparray = test_input_data.item().get(parameter)
                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
                else:
                    array = produce_noinit_array(parameter_info["cl_type"], parameter_info["size"])
                    if parameter_info["cl_type"] in cl_type_to_numpy_type:
                        nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])
                    else:
                        nparray = np.zeros(parameter_info["size"], dtype=name_to_cl_vector_type[parameter_info["cl_type"]])
                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    print(nparray)
                    print(buffer)
                    arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
            else:
                if parameter_info["cl_type"] == "image2d_t":
                    nparray = test_input_data.item().get(parameter)
                    image_buf = cl.image_from_array(ctx, nparray, 4)
                    arguments[int(parameter_info["pos"])] = {"type": "variable", "variable": image_buf}
                elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or parameter_info[
                    "cl_type"] == "float2":
                    nparray = test_input_data.item().get(parameter)
                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
                elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info[
                    "cl_type"] == "uint4":
                    nparray = test_input_data.item().get(parameter)
                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
                else:
                    kernel_scalar_arg_dtypes[parameter_info["pos"]] = cl_type_to_numpy_type[parameter_info["cl_type"]]
                    data_type = cl_type_to_numpy_type[parameter_info["cl_type"]]
                    variable =  data_type(test_input_data.item().get(parameter))
                    arguments[int(parameter_info["pos"])] = {"type": "variable", "variable": variable}
            """
            if parameter_info["pointer"]:
                if parameter_info["fuzzing"] == "random":
                    if parameter_info["cl_type"] in cl_type_to_numpy_type:
                        nparray = test_input_data[parameter]
                    elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or \
                            parameter_info["cl_type"] == "float2":
                        nparray = test_input_data[parameter]
                    elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info[
                        "cl_type"] == "uint4":
                        nparray = test_input_data[parameter]
                    # struct
                    else:
                        struct_data_type_array = []
                        struct_data_name_array = []
                        for j in range((len(parameter_info) - 10) // 2):
                            struct_data_name_array.append(
                                parameter_info[parameter_info["cl_type"] + "_key" + str(j) + "_name"])
                            struct_data_type_array.append(
                                parameter_info[parameter_info["cl_type"] + "_key" + str(j) + "_type"])
                        nparray = read_struct_array_from_file(filename, struct_data_name_array, struct_data_type_array)

                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}

                elif parameter_info["fuzzing"] == "init_file":

                    filename = parameter_info["init_file"]

                    #
                    if parameter_info["cl_type"] in cl_type_to_numpy_type:

                        array = read_array_from_file(filename, parameter_info["cl_type"])
                        nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])
                        # my_struct_list = []

                    elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or \
                            parameter_info["cl_type"] == "float2":
                        nparray = read_float_vector_array_from_file(filename, parameter_info["cl_type"])

                    elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info[
                        "cl_type"] == "uint4":
                        nparray = read_int_vector_array_from_file(filename, parameter_info["cl_type"])


                    # struct
                    else:
                        struct_data_type_array = []
                        struct_data_name_array = []
                        for j in range((len(parameter_info) - 10) // 2):
                            struct_data_name_array.append(
                                parameter_info[parameter_info["cl_type"] + "_key" + str(j) + "_name"])
                            struct_data_type_array.append(
                                parameter_info[parameter_info["cl_type"] + "_key" + str(j) + "_type"])
                        nparray = read_struct_array_from_file(filename, struct_data_name_array, struct_data_type_array)
                    #

                    # array = read_array_from_file(filename, parameter_info["cl_type"])
                    # nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])

                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}

                elif parameter_info["fuzzing"] == "noinit":

                    array = produce_noinit_array(parameter_info["cl_type"], parameter_info["size"])
                    nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])

                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
            # image
            else:
                if parameter_info["cl_type"] == "image2d_t":
                    if parameter_info["fuzzing"] == "random":
                        filename = target_kernel_name + "_" + parameter + "_test" + str(test_id_str).zfill(6) + ".txt.npy"
                        array_image = np.load(filename)
                        image_buf = cl.image_from_array(ctx, array_image, 4)
                        # buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image_buf)
                        arguments[int(parameter_info["pos"])] = {"type": "variable", "variable": image_buf}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        array_image = np.load(filename)
                        image_buf = cl.image_from_array(ctx, array_image, 4)
                        # buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image_buf)
                        arguments[int(parameter_info["pos"])] = {"type": "variable", "variable": image_buf}
                elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or parameter_info[
                    "cl_type"] == "float2":
                    if parameter_info["fuzzing"] == "random":
                        filename = target_kernel_name + "_" + parameter + "_test" + str(test_id_str).zfill(6) + ".txt"
                        nparray = read_float_vector_array_from_file(filename, parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
                        # arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        nparray = read_float_vector_array_from_file(filename, parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
                elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info[
                    "cl_type"] == "uint4":
                    if parameter_info["fuzzing"] == "random":
                        filename = target_kernel_name + "_" + parameter + "_test" + str(test_id_str).zfill(6) + ".txt"
                        nparray = read_int_vector_array_from_file(filename, parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
                        # arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        nparray = read_int_vector_array_from_file(filename, parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[int(parameter_info["pos"])] = {"type": "array", "array": nparray, "buffer": buffer}
                        # arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}

                else:
                    kernel_scalar_arg_dtypes[parameter_info["pos"]] = cl_type_to_numpy_type[parameter_info["cl_type"]]
                    if parameter_info["fuzzing"] == "random":
                        filename = target_kernel_name + "_" + parameter + "_test" + str(test_id_str).zfill(6) + ".txt"
                        variable = read_variable_from_file(filename, parameter_info["cl_type"])
                        arguments[int(parameter_info["pos"])] = {"type": "variable", "variable": variable}
                    elif parameter_info["fuzzing"] == "initial_value":
                        variable = parameter_info["initial_value"]
                        arguments[int(parameter_info["pos"])] = {"type": "variable", "variable": variable}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        variable = read_variable_from_file(filename, parameter_info["cl_type"])
                        arguments[int(parameter_info["pos"])] = {"type": "variable", "variable": variable}
                        """

    # Set kernel parameters
    kernel_argument_list = [queue, kernel_info["global"], kernel_info["local"]]
    for i in range(len(arguments)):
        current_argument = arguments[i]
        if current_argument["type"] == "variable":
            kernel_argument_list.append(current_argument["variable"])
        else:
            kernel_argument_list.append(current_argument["buffer"])

    np_branch_recorder = None
    np_barrier_recorder = None
    np_loop_recorder = None
    buffer_branch_recorder = None
    buffer_barrier_recorder = None
    buffer_loop_recorder = None

    if measure_cov:
        if kernel_info["Branches"] != 0:
            branch_recorder = produce_noinit_array("int", kernel_info["Branches"])
            np_branch_recorder = np.asarray(branch_recorder, dtype=np.int32)
            buffer_branch_recorder = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_branch_recorder)
            kernel_argument_list.append(buffer_branch_recorder)
        if kernel_info["Barriers"] != 0:
            barrier_recorder = produce_noinit_array("int", kernel_info["Barriers"])
            np_barrier_recorder = np.asarray(barrier_recorder, dtype=np.int32)
            buffer_barrier_recorder = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_barrier_recorder)
            kernel_argument_list.append(buffer_barrier_recorder)
        if kernel_info["Loops"] != 0:
            loop_recorder = produce_noinit_array("int", kernel_info["Loops"])
            np_loop_recorder = np.asarray(loop_recorder, dtype=np.int32)
            buffer_loop_recorder = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_loop_recorder)
            kernel_argument_list.append(buffer_loop_recorder)
    elif with_schedule is not None:
        buffer_schedule = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=with_schedule)
        kernel_argument_list.append(buffer_schedule)
    # Execute kernel
    kernel(*kernel_argument_list)
    print(kernel_argument_list)
    queue.finish()

    for parameter in target_kernel_info:
        parameter_info = target_kernel_info[parameter]
        if parameter_info["result"]:
            array = arguments[int(parameter_info["pos"])]["array"]
            buffer = arguments[int(parameter_info["pos"])]["buffer"]
            cl.enqueue_copy(queue, array, buffer)
            test_id_str = test_file_name[-10:-4]
            file_name = "result_" + target_kernel_name + "_" + parameter + "_out_" + kernel_filename + "_" + \
                        time_stamp + "_" + test_id_str + ".txt"
            #input_file_name = target_kernel_name + "_" + parameter + "_" + time_stamp + "_" + test_id_str + ".txt"
            queue.finish()
            write_array_to_file(array, file_name)
            # if (compare2file(file_name,input_file_name)):
            #    print(parameter+": input and output is different!!!!!!!!!!!!!")
            #    print(parameter+": input and output is different!!!!!!!!!!!!!")

    cov = {}
    if measure_cov:
        if kernel_info["Branches"] != 0:
            cl.enqueue_copy(queue, np_branch_recorder, buffer_branch_recorder)
            cov["branch"] = np_branch_recorder
        if kernel_info["Barriers"] != 0:
            cl.enqueue_copy(queue, np_barrier_recorder, buffer_barrier_recorder)
            cov["barrier"] = np_barrier_recorder
        if kernel_info["Loops"] != 0:
            cl.enqueue_copy(queue, np_loop_recorder, buffer_loop_recorder)
            cov["loop"] = np_loop_recorder
    # cov = {"branch": np_branch_recorder, "barrier": np_barrier_recorder, "loop": np_loop_recorder}
    return cov


def pretty_print_cov(cov, mode="simple"):
    branch = cov["branch"] if "branch" in cov else None
    loop = cov["loop"] if "loop" in cov else None
    barrier = cov["barrier"] if "barrier" in cov else None

    num_branches = branch.size if branch is not None else 0
    num_loops = loop.size if loop is not None else 0
    num_barriers = barrier.size if barrier is not None else 0


    covered_branches = np.count_nonzero(branch == 1) if branch is not None else 0
    covered_loops = np.count_nonzero(loop != 0) if loop is not None else 0
    covered_barriers = np.count_nonzero(barrier == 0) if barrier is not None else 0
    if num_branches != 0: print("Branch coverage: %0.2f%%" % (covered_branches / num_branches * 100))
    if num_loops != 0: print("Loop coverage: %0.2f%%" % (covered_loops / num_loops * 100))
    if num_barriers != 0: print("Barrier coverage: %0.2f%%" % (covered_barriers/ num_barriers * 100))

    if mode != "simple":
        if num_branches != 0:
            print("Detailed branch coverage")
            header = ['Branch #', 'True', 'False']
            rows = []
            for i in range(num_branches // 2):
                row = [str(i + 1)]
                if branch[i * 2]:
                    row.append("Covered")
                else:
                    row.append("Not Covered")
                if branch[i * 2 + 1]:
                    row.append("Covered")
                else:
                    row.append("Not Covered")
                rows.append(row)
            table = tabulate(rows, headers=header, tablefmt='grid')

            print(table)

        if num_loops != 0:
            print("Detailed loop coverage")
            header = ['Loop #', '0 iteration', '1 iteration', '+1 iterations', 'boundary reached']
            rows = []
            for i in range(num_loops):
                row = [str(i + 1)]
                if loop[i] & 1 == 1:
                    row.append("Y")
                else:
                    row.append("N")
                if loop[i] & 2 == 2:
                    row.append("Y")
                else:
                    row.append("N")
                if loop[i] & 4 == 4:
                    row.append("Y")
                else:
                    row.append("N")
                if loop[i] & 8 == 8:
                    row.append("Y")
                else:
                    row.append("N")
                rows.append(row)
            table = tabulate(rows, headers=header, tablefmt='grid')

            print(table)

        if num_barriers != 0:
            print("Detailed barrier coverage")
            header = ['Barrier #', 'Covered']
            rows = []
            for i in range(num_barriers):
                row = [str(i + 1)]
                if barrier[i]:
                    row.append("N")
                else:
                    row.append("Y")
                rows.append(row)
            table = tabulate(rows, headers=header, tablefmt='grid')

            print(table)


def aggregate_cov(cov):
    example_line = cov[list(cov.keys())[0]]
    num_branches = example_line["branch"].size if "branch" in example_line else 0
    num_loops = example_line["loop"].size if "loop" in example_line else 0
    num_barriers = example_line["barrier"].size if "barrier" in example_line else 0
    branch = np.zeros(num_branches, dtype=int)
    loop = np.zeros(num_loops, dtype=int)
    barrier = np.zeros(num_barriers, dtype=int)
    for test_id in cov:
        if num_branches != 0:
            current_branch_cov = cov[test_id]["branch"]
            for i in range(num_branches):
                branch[i] = branch[i] | current_branch_cov[i]
        if num_loops != 0:
            current_loop_cov = cov[test_id]["loop"]
            for i in range(num_loops):
                loop[i] = loop[i] | current_loop_cov[i]
        if num_barriers != 0:
            current_barrier_cov = cov[test_id]["barrier"]
            for i in range(num_barriers):
                barrier[i] = barrier[i] & current_barrier_cov[i]
    final_cov = {}
    if num_barriers != 0:
        final_cov["barrier"] = barrier
    if num_loops != 0:
        final_cov["loop"] = loop
    if num_branches:
        final_cov["branch"] = branch
    return final_cov

