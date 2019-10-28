#ifndef CLCOV_KERNEL_BRANCH_COVERAGE_CHECKER_CONSTANTS
#define CLCOV_KERNEL_BRANCH_COVERAGE_CHECKER_CONSTANTS

namespace kernel_rewriter_constants{
    const char* const GLOBAL_COVERAGE_RECORDER_NAME = "ocl_kernel_branch_triggered_recorder";
    const char* const LOCAL_COVERAGE_RECORDER_NAME = "my_ocl_kernel_branch_triggered_recorder";
    const char* const LOCAL_BARRIER_COUNTER_NAME = "ocl_kernel_barrier_count";
    const char* const GLOBAL_BARRIER_DIVERFENCE_RECORDER_NAME = "ocl_barrier_divergence_recorder";
    const char* const GLOBAL_LOOP_RECORDER_NAME = "ocl_kernel_loop_recorder";
    const char* const LOCAL_LOOP_RECORDER_NAME = "my_ocl_kernel_loop_recorder";
    const char* const PRIVATE_LOOP_ITERATION_COUNTER = "private_ocl_kernel_loop_iter_counter";
    const char* const PRIVATE_LOOP_BOUNDARY_RECORDER = "private_ocl_kernel_loop_boundary_not_reached";
    const int LOOP_NOT_EXECUTED = 1;
    const int LOOP_EXECUTED_ONCE = 2;
    const int LOOP_EXECUTED_MORE_THAN_ONCE = 4;
    const int LOOP_REACHED_BOUNDARY = 8;
    const char* const FAKE_HEADER_MACRO = "OPENCLBC_FAKE_HEADER_FOR_LIBTOOLING_";
    const char* const NEW_BARRIER_MACRO = "#define OCL_NEW_BARRIER(barrierid,arg)\\\n"\
        "{\\\n"\
        "  atom_inc(&ocl_kernel_barrier_count[barrierid]);\\\n"\
        "  barrier(arg);\\\n"\
        "  if (ocl_kernel_barrier_count[barrierid]!=ocl_get_general_size()) {\\\n"\
        "    ocl_barrier_divergence_recorder[barrierid]=1;\\\n"\
        "  }\\\n"\
        "  barrier(arg);\\\n"\
        "  ocl_kernel_barrier_count[barrierid]=0;\\\n"\
        "  barrier(arg);\\\n"\
        "}\n"\
        "int ocl_get_general_size(){\n"\
        "  int result = 1;\\\n"\
        "  for (int i=0; i<get_work_dim(); i++){\n"\
        "    result*=get_local_size(i);\n"\
        "  }\n"\
        "  return result;\n"\
        "}\n";
    const char* const SCHEDULER_WRAPPER = "int get_group_id_new(int dimindx, __global uint3* cl_schedule_map) {\n"\
        "    int id;\n"\
        "    int dimension = get_work_dim();\n"\
        "    if (dimension == 1) {\n"\
        "        id = get_group_id(0);\n"\
        "    } else if (dimension == 2) {\n"\
        "        id = get_group_id(0) + get_group_id(1) * get_num_groups(0);\n"\
        "    } else if (dimension == 3) {\n"\
        "        id = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));\n"\
        "    }\n"\
        "    if (dimindx == 0) return cl_schedule_map[id].x;\n"\
        "    else if (dimindx == 1) return cl_schedule_map[id].y;\n"\
        "    else return cl_schedule_map[id].z;\n"\
        "}\n"\
        "\n"\
        "int get_global_id_new(int dimindx, __global uint3* cl_schedule_map) {\n"\
        "    int new_group_id = get_group_id_new(dimindx, cl_schedule_map);\n"\
        "    return new_group_id * get_local_size(dimindx) + get_local_id(dimindx);\n"\
        "}\n\n";
}

namespace error_code{
    const int STATUS_OK = 0;
    const int TWO_MANY_HOST_FILE_SUPPLIED = 1;
    const int NO_HOST_FILE_SUPPLIED = 2;
    const int REMOVE_KERNEL_FAKE_HEADER_FAILED_KERNEL_DOES_NOT_EXIST = 3;
    const int KERNEL_FILE_ALREADY_HAS_FAKE_HEADER = 4;
    const int NO_NEED_TO_TEST_COVERAGE = 5;
}

#endif