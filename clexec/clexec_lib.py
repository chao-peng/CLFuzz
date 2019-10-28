import random
import yaml
import numpy as np
import pyopencl as cl
import pyopencl.cltypes


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
    #"ufloat4" : cl.cltypes.ufloat4,
    #"ufloat3" : cl.cltypes.ufloat3,
    #"ufloat2" : cl.cltypes.ufloat2,
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


def produce_rand_value(type_str):
    lower_bound = 0
    upper_bound = 0
    if type_str == 'bool':
        return random.choice([True, False])
    elif type_str == 'char':
        lower_bound = - 2 ** 3
        upper_bound = 2 ** 3 - 1
    elif type_str == 'uchar':
        lower_bound = 0
        upper_bound = 2 ** 4 - 1
    elif type_str == 'short':
        lower_bound = -2 ** 7
        upper_bound = 2 ** 7 - 1
    elif type_str == 'ushort':
        lower_bound = 0
        upper_bound = 2 ** 8 - 1
    elif type_str == 'int':
        lower_bound = - 2 ** 15
        upper_bound = 2 ** 15 - 1
    elif type_str == 'uint':
        lower_bound = 0
        upper_bound = 2 ** 16 - 1
    elif type_str == 'long':
        lower_bound = - 2 ** 31
        upper_bound = 2 ** 31 - 1
    elif type_str == 'ulong':
        lower_bound = 0
        upper_bound = 2 ** 32 - 1
    if type_str != 'float' and type_str != 'double' and type_str != 'halt':
        return random.randint(lower_bound, upper_bound)
    elif type_str == 'float':
        lower_bound = 3.4E-10
        upper_bound = 3.4E10
    elif type_str == 'double':
        lower_bound = 1.7E-20
        upper_bound = 1.7E20
    elif type_str == 'half':
        lower_bound = 3.2E-5
        upper_bound = 3.2E5
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


def parse_kernel_info(info_filename):
    yaml_reader = open(info_filename, 'r')
    kernel_info = yaml.safe_load(yaml_reader)
    yaml_reader.close()
    return kernel_info


def preparation(target_kernel_name, kernel_info, num_tests):
    target_kernel_info = kernel_info[target_kernel_name]
    num_parameters = 0
    for parameter in target_kernel_info:
        parameter_info = target_kernel_info[parameter]
        num_parameters = num_parameters + 1
        if parameter_info["fuzzing"] == "random":
            if parameter_info["pointer"]:
                for i in range(num_tests):
                    if parameter_info["cl_type"] in cl_type_to_numpy_type:
                        #print(222222222)
                        random_array = produce_rand_array(parameter_info["cl_type"], parameter_info["size"])

                    elif parameter_info["cl_type"] in name_to_cl_vector_type:
                        random_array = np.empty((parameter_info["size"],), dtype=name_to_cl_vector_type[parameter_info["cl_type"]])

                    #elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2":
                    #    random_array = np.empty((parameter_info["size"],), dtype=name_to_cl_vector_type[parameter_info["cl_type"]])
                    #struct#
                    else:
                        #print(1111111111)
                        my_struct_list = []
                        for j in range((len(parameter_info)-10)//2):
                            #print(j)
                            t = (parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_name"],
                            cl_type_to_numpy_type[parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_type"]])
                            my_struct_list.append(t)
                        #print(my_struct_list)
                        #    #my_struct_list = [("field1", np.int32), ("field2", np.float32), ("field3", np.float32)]
                        my_struct = np.dtype(my_struct_list)
                        #my_struct = np.dtype([("x", np.int32), ("y", np.int32)])
                        random_array = np.empty(parameter_info["size"], my_struct)
                        #print(random_array)
                    file_name = target_kernel_name + "_" + parameter + "_" + kernel_info["time_stamp"] + "_" + str(i).zfill(3) + ".txt"
                    write_array_to_file(random_array, file_name)
                parameter_info["generated_file"] = target_kernel_name + "_" + parameter + "_" + kernel_info["time_stamp"] + "_"
                parameter_info["fuzzing"] = "generated_file"
            else:
                for i in range(num_tests):
                    if  parameter_info["cl_type"] =="image2d_t":
                        #a=np.random.random((1200,1600,4))
                        #b=np.random.randint(-2 ** 31+1,2 ** 31 - 2,size=(1200,1600,4))
                        #random_image = a+b
                        random_image = np.empty((1200,1600,4), np.dtype(np.float32))
                        [x,y,z] = random_image.shape

                        for l in range(x):
                            for j in range(y):
                                for k in range(z):
                                    random_image[l,j,k] = produce_rand_value("float")

                        file_name = target_kernel_name + "_" + parameter + "_" + kernel_info["time_stamp"] + "_" + str(i).zfill(3) + ".txt"
                        write_image_to_file(random_image, file_name)
                        np.save(file_name,random_image)
                    elif parameter_info["cl_type"] in name_to_cl_vector_type:
                        random_array = np.empty((parameter_info["size"],), dtype=name_to_cl_vector_type[parameter_info["cl_type"]])
                        file_name = target_kernel_name + "_" + parameter + "_" + kernel_info["time_stamp"] + "_" + str(i).zfill(3) + ".txt"
                        write_array_to_file(random_array, file_name)

                    else:
                        random_value = produce_rand_value(parameter_info["cl_type"])
                        file_name = target_kernel_name + "_" + parameter + "_" + kernel_info["time_stamp"] + "_" + str(i).zfill(3) + ".txt"
                        write_variable_to_file(random_value, file_name)
                parameter_info["fuzzing"] = "generated_file"
                parameter_info["generated_file"] = target_kernel_name + "_" + parameter + "_" + kernel_info["time_stamp"] + "_"
    kernel_info["num_parameters"] = num_parameters




def execute_kernel(target_kernel_name, kernel_filename, kernel_info, test_id):
    target_kernel_info = kernel_info[target_kernel_name]
    test_id_str = str(test_id).zfill(3)
    arguments = {}

    ctx = cl.Context(dev_type=cl.device_type.GPU)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg_source = read_file_as_string(kernel_filename)
    prg = cl.Program(ctx, prg_source).build()
    real_num_parameters = kernel_info["num_parameters"]
    if kernel_info["Cov"]:
        if kernel_info["Barriers"] != 0:
            real_num_parameters = real_num_parameters + 1
        if kernel_info["Branches"] != 0:
            real_num_parameters = real_num_parameters + 1
        if kernel_info["Loops"] != 0:
            real_num_parameters = real_num_parameters + 1
    kernel_scalar_arg_dtypes = [None] * real_num_parameters
    kernel = getattr(prg, target_kernel_name)
    for parameter in target_kernel_info:
        parameter_info = target_kernel_info[parameter]
        if parameter_info["cl_scope"] == "local":
            buffer = cl.LocalMemory(parameter_info["size"])
            arguments[parameter_info["pos"]] = {"type": "array", "buffer": buffer}
        else:
            if parameter_info["pointer"]:
                if parameter_info["fuzzing"] == "generated_file":
                    filename = parameter_info["generated_file"] + test_id_str + ".txt"

                    if parameter_info["cl_type"] in cl_type_to_numpy_type:

                        array = read_array_from_file(filename, parameter_info["cl_type"])
                        nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])
                        #my_struct_list = []

                    elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or parameter_info["cl_type"] == "float2":
                        nparray = read_float_vector_array_from_file(filename,parameter_info["cl_type"])

                    elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info["cl_type"] == "uint4":
                        nparray = read_int_vector_array_from_file(filename,parameter_info["cl_type"])


                    #struct
                    else:
                        struct_data_type_array = []
                        struct_data_name_array = []
                        for j in range((len(parameter_info)-10)//2):
                            struct_data_name_array.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_name"])
                            struct_data_type_array.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_type"])
                        nparray = read_struct_array_from_file(filename,struct_data_name_array,struct_data_type_array)



                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}

                elif parameter_info["fuzzing"] == "init_file":

                    filename = parameter_info["init_file"]

#
                    if parameter_info["cl_type"] in cl_type_to_numpy_type:

                        array = read_array_from_file(filename, parameter_info["cl_type"])
                        nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])
                        #my_struct_list = []

                    elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or parameter_info["cl_type"] == "float2":
                        nparray = read_float_vector_array_from_file(filename,parameter_info["cl_type"])

                    elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info["cl_type"] == "uint4":
                        nparray = read_int_vector_array_from_file(filename,parameter_info["cl_type"])


                    #struct
                    else:
                        struct_data_type_array = []
                        struct_data_name_array = []
                        for j in range((len(parameter_info)-10)//2):
                            struct_data_name_array.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_name"])
                            struct_data_type_array.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_type"])
                        nparray = read_struct_array_from_file(filename,struct_data_name_array,struct_data_type_array)
#

                    #array = read_array_from_file(filename, parameter_info["cl_type"])
                    #nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])

                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}

                elif parameter_info["fuzzing"] == "noinit":

                    array = produce_noinit_array(parameter_info["cl_type"], parameter_info["size"])
                    nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])

                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
            #image
            else:
                if parameter_info["cl_type"] == "image2d_t":
                    if parameter_info["fuzzing"] == "generated_file":
                        filename = parameter_info["generated_file"] + test_id_str + ".txt.npy"
                        array_image = np.load(filename)
                        image_buf = cl.image_from_array(ctx, array_image, 4)
                        #buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image_buf)
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": image_buf}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        array_image = np.load(filename)
                        image_buf = cl.image_from_array(ctx, array_image, 4)
                        #buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image_buf)
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": image_buf}
                elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or parameter_info["cl_type"] == "float2":
                    if parameter_info["fuzzing"] == "generated_file":
                        filename = parameter_info["generated_file"] + test_id_str + ".txt"
                        nparray = read_float_vector_array_from_file(filename,parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
                        #arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        nparray = read_float_vector_array_from_file(filename,parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
                elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info["cl_type"] == "uint4":
                    if parameter_info["fuzzing"] == "generated_file":
                        filename = parameter_info["generated_file"] + test_id_str + ".txt"
                        nparray = read_int_vector_array_from_file(filename,parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
                        #arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        nparray = read_int_vector_array_from_file(filename,parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
                        #arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}

                else:
                    kernel_scalar_arg_dtypes[parameter_info["pos"]] = cl_type_to_numpy_type[parameter_info["cl_type"]]
                    if parameter_info["fuzzing"] == "generated_file":
                        filename = parameter_info["generated_file"] + test_id_str + ".txt"
                        variable = read_variable_from_file(filename, parameter_info["cl_type"])
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": variable}
                    elif parameter_info["fuzzing"] == "initial_value":
                        variable = parameter_info["initial_value"]
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": variable}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        variable = read_variable_from_file(filename, parameter_info["cl_type"])
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": variable}
    # Set kernel parameters
    kernel.set_scalar_arg_dtypes(kernel_scalar_arg_dtypes)
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

    if kernel_info["Cov"]:
        if kernel_info["Branches"] != 0:
            branch_recorder = produce_noinit_array("int", kernel_info["Branches"] * 2)
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
    # Execute kernel
    kernel(*kernel_argument_list)



    for parameter in target_kernel_info:
        parameter_info = target_kernel_info[parameter]
        if parameter_info["result"]:
            array = arguments[parameter_info["pos"]]["array"]
            buffer = arguments[parameter_info["pos"]]["buffer"]
            cl.enqueue_copy(queue, array, buffer)
            file_name = "result_"+target_kernel_name + "_" + parameter + "_out_" + kernel_filename+"_" + kernel_info["time_stamp"] + "_" + test_id_str + ".txt"
            input_file_name = target_kernel_name + "_" + parameter +"_" + kernel_info["time_stamp"] + "_" + test_id_str + ".txt"

            write_array_to_file(array, file_name)
            #if (compare2file(file_name,input_file_name)):
            #    print(parameter+": input and output is different!!!!!!!!!!!!!")
            #    print(parameter+": input and output is different!!!!!!!!!!!!!")



    if kernel_info["Cov"]:
        if kernel_info["Branches"] != 0:
            cl.enqueue_copy(queue, np_branch_recorder, buffer_branch_recorder)

        if kernel_info["Barriers"] != 0:
            cl.enqueue_copy(queue, np_barrier_recorder, buffer_barrier_recorder)

        if kernel_info["Loops"] != 0:
            cl.enqueue_copy(queue, np_loop_recorder, buffer_loop_recorder)

    cov = {"branch": np_branch_recorder, "barrier": np_barrier_recorder, "loop": np_loop_recorder}
    return cov


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


def execute_kernel_with_schedule(target_kernel_name, kernel_filename, kernel_info, test_id, schedule):
    target_kernel_info = kernel_info[target_kernel_name]
    test_id_str = str(test_id).zfill(3)
    arguments = {}

    ctx = cl.Context(dev_type=cl.device_type.GPU)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg_source = read_file_as_string(kernel_filename)
    prg = cl.Program(ctx, prg_source).build()
    real_num_parameters = kernel_info["num_parameters"] + 1

    kernel_scalar_arg_dtypes = [None] * real_num_parameters
    kernel = getattr(prg, target_kernel_name)
    for parameter in target_kernel_info:
        parameter_info = target_kernel_info[parameter]
        if parameter_info["cl_scope"] == "local":
            buffer = cl.LocalMemory(parameter_info["size"])
            arguments[parameter_info["pos"]] = {"type": "array", "buffer": buffer}
        else:
            if parameter_info["pointer"]:
                if parameter_info["fuzzing"] == "generated_file":
                    filename = parameter_info["generated_file"] + test_id_str + ".txt"

                    if parameter_info["cl_type"] in cl_type_to_numpy_type:

                        array = read_array_from_file(filename, parameter_info["cl_type"])
                        nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])
                        #my_struct_list = []

                    elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or parameter_info["cl_type"] == "float2":
                        nparray = read_float_vector_array_from_file(filename,parameter_info["cl_type"])

                    elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info["cl_type"] == "uint4":
                        nparray = read_int_vector_array_from_file(filename,parameter_info["cl_type"])


                    #struct
                    else:
                        struct_data_type_array = []
                        struct_data_name_array = []
                        for j in range((len(parameter_info)-10)//2):
                            struct_data_name_array.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_name"])
                            struct_data_type_array.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_type"])
                        nparray = read_struct_array_from_file(filename,struct_data_name_array,struct_data_type_array)



                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}

                elif parameter_info["fuzzing"] == "init_file":

                    filename = parameter_info["init_file"]

#
                    if parameter_info["cl_type"] in cl_type_to_numpy_type:

                        array = read_array_from_file(filename, parameter_info["cl_type"])
                        nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])
                        #my_struct_list = []

                    elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or parameter_info["cl_type"] == "float2":
                        nparray = read_float_vector_array_from_file(filename,parameter_info["cl_type"])

                    elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info["cl_type"] == "uint4":
                        nparray = read_int_vector_array_from_file(filename,parameter_info["cl_type"])


                    #struct
                    else:
                        struct_data_type_array = []
                        struct_data_name_array = []
                        for j in range((len(parameter_info)-10)//2):
                            struct_data_name_array.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_name"])
                            struct_data_type_array.append(parameter_info[parameter_info["cl_type"]+"_key"+str(j)+"_type"])
                        nparray = read_struct_array_from_file(filename,struct_data_name_array,struct_data_type_array)
#

                    #array = read_array_from_file(filename, parameter_info["cl_type"])
                    #nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])

                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}

                elif parameter_info["fuzzing"] == "noinit":

                    array = produce_noinit_array(parameter_info["cl_type"], parameter_info["size"])
                    nparray = np.asarray(array, dtype=cl_type_to_numpy_type[parameter_info["cl_type"]])

                    buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                    arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
            #image
            else:
                if parameter_info["cl_type"] == "image2d_t":
                    if parameter_info["fuzzing"] == "generated_file":
                        filename = parameter_info["generated_file"] + test_id_str + ".txt.npy"
                        array_image = np.load(filename)
                        image_buf = cl.image_from_array(ctx, array_image, 4)
                        #buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image_buf)
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": image_buf}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        array_image = np.load(filename)
                        image_buf = cl.image_from_array(ctx, array_image, 4)
                        #buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image_buf)
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": image_buf}
                elif parameter_info["cl_type"] == "float4" or parameter_info["cl_type"] == "float3" or parameter_info["cl_type"] == "float2":
                    if parameter_info["fuzzing"] == "generated_file":
                        filename = parameter_info["generated_file"] + test_id_str + ".txt"
                        nparray = read_float_vector_array_from_file(filename,parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
                        #arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        nparray = read_float_vector_array_from_file(filename,parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
                elif parameter_info["cl_type"] == "int4" or parameter_info["cl_type"] == "uint2" or parameter_info["cl_type"] == "uint4":
                    if parameter_info["fuzzing"] == "generated_file":
                        filename = parameter_info["generated_file"] + test_id_str + ".txt"
                        nparray = read_int_vector_array_from_file(filename,parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
                        #arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        nparray = read_int_vector_array_from_file(filename,parameter_info["cl_type"])
                        buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nparray)
                        arguments[parameter_info["pos"]] = {"type": "array", "array": nparray, "buffer": buffer}
                        #arguments[parameter_info["pos"]] = {"type": "variable", "variable": nparray}

                else:
                    kernel_scalar_arg_dtypes[parameter_info["pos"]] = cl_type_to_numpy_type[parameter_info["cl_type"]]
                    if parameter_info["fuzzing"] == "generated_file":
                        filename = parameter_info["generated_file"] + test_id_str + ".txt"
                        variable = read_variable_from_file(filename, parameter_info["cl_type"])
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": variable}
                    elif parameter_info["fuzzing"] == "initial_value":
                        variable = parameter_info["initial_value"]
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": variable}
                    elif parameter_info["fuzzing"] == "init_file":
                        filename = parameter_info["init_file"]
                        variable = read_variable_from_file(filename, parameter_info["cl_type"])
                        arguments[parameter_info["pos"]] = {"type": "variable", "variable": variable}
    # Set kernel parameters
    kernel.set_scalar_arg_dtypes(kernel_scalar_arg_dtypes)
    kernel_argument_list = [queue, kernel_info["global"], kernel_info["local"]]
    for i in range(len(arguments)):
        current_argument = arguments[i]
        if current_argument["type"] == "variable":
            kernel_argument_list.append(current_argument["variable"])
        else:
            kernel_argument_list.append(current_argument["buffer"])

    buffer_schedule = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=schedule)
    kernel_argument_list.append(buffer_schedule)

    # Execute kernel
    kernel(*kernel_argument_list)



    for parameter in target_kernel_info:
        parameter_info = target_kernel_info[parameter]
        if parameter_info["result"]:
            array = arguments[parameter_info["pos"]]["array"]
            buffer = arguments[parameter_info["pos"]]["buffer"]
            cl.enqueue_copy(queue, array, buffer)
            file_name = "result_"+target_kernel_name + "_" + parameter + "_out_" + kernel_filename+"_" + kernel_info["time_stamp"] + "_" + test_id_str + ".txt"
            input_file_name = target_kernel_name + "_" + parameter +"_" + kernel_info["time_stamp"] + "_" + test_id_str + ".txt"

            write_array_to_file(array, file_name)
            #if (compare2file(file_name,input_file_name)):
            #    print(parameter+": input and output is different!!!!!!!!!!!!!")
            #    print(parameter+": input and output is different!!!!!!!!!!!!!")


def save_yaml_file(yaml_obj, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(yaml_obj, outfile, default_flow_style=False)
