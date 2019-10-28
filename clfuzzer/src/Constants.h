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
}

namespace error_code{
    const int STATUS_OK = 0;
    const int TWO_MANY_HOST_FILE_SUPPLIED = 1;
    const int NO_HOST_FILE_SUPPLIED = 2;
    const int REMOVE_KERNEL_FAKE_HEADER_FAILED_KERNEL_DOES_NOT_EXIST = 3;
    const int KERNEL_FILE_ALREADY_HAS_FAKE_HEADER = 4;
    const int NO_NEED_TO_TEST_COVERAGE = 5;
}

namespace type_name {
    const unsigned int BOOL =                   0x0000;

    const unsigned int CHAR =                   0x0010;
    const unsigned int CHAR2 =                  0x0011;
    const unsigned int CHAR4 =                  0x0012;
    const unsigned int CHAR8 =                  0x0013;
    const unsigned int CHAR16 =                 0x0014;

    const unsigned int UCHAR =                  0x1010;
    const unsigned int UCHAR2 =                 0x1011;
    const unsigned int UCHAR4 =                 0x1012;
    const unsigned int UCHAR8 =                 0x1013;
    const unsigned int UCHAR16 =                0x1014;

    const unsigned int SHORT =                  0x0020;
    const unsigned int SHORT2 =                 0x0021;
    const unsigned int SHORT4 =                 0x0022;
    const unsigned int SHORT8 =                 0x0023;
    const unsigned int SHORT16 =                0x0024;

    const unsigned int USHORT =                 0x1020;
    const unsigned int USHORT2 =                0x1021;
    const unsigned int USHORT4 =                0x1022;
    const unsigned int USHORT8 =                0x1023;
    const unsigned int USHORT16 =               0x1024;

    const unsigned int INT =                    0x0030;
    const unsigned int INT2 =                   0x0031;
    const unsigned int INT4 =                   0x0032;
    const unsigned int INT8 =                   0x0033;
    const unsigned int INT16 =                  0x0034;

    const unsigned int UINT =                   0x1030;
    const unsigned int UINT2 =                  0x1031;
    const unsigned int UINT4 =                  0x1032;
    const unsigned int UINT8 =                  0x1033;
    const unsigned int UINT16 =                 0x1034;

    const unsigned int LONG =                   0x0040;
    const unsigned int LONG2 =                  0x0041;
    const unsigned int LONG4 =                  0x0042;
    const unsigned int LONG8 =                  0x0043;
    const unsigned int LONG16 =                 0x0044;

    const unsigned int ULONG =                  0x1040;
    const unsigned int ULONG2 =                 0x1041;
    const unsigned int ULONG4 =                 0x1042;
    const unsigned int ULONG8 =                 0x1043;
    const unsigned int ULONG16 =                0x1044;

    const unsigned int FLOAT =                  0x0050;
    const unsigned int FLOAT2 =                 0x0051;
    const unsigned int FLOAT4 =                 0x0052;
    const unsigned int FLOAT8 =                 0x0053;
    const unsigned int FLOAT16 =                0x0054;

    const unsigned int HALF =                   0x0060;
    const unsigned int HALF2 =                  0x0061;
    const unsigned int HALF4 =                  0x0062;
    const unsigned int HALF8 =                  0x0063;
    const unsigned int HALF16 =                 0x0064;

    const unsigned int SIZE_T =                 0x0070;

    const unsigned int PTRDIFF_T =              0x0080;

    const unsigned int INTPTR_T =               0x0090;

    const unsigned int UINTPTR_T =              0x1090;

    const unsigned int VOID =                   0x00A0;

    const unsigned int DOUBLE =                 0x00B0;
    const unsigned int DOUBLE2 =                0x00B1;
    const unsigned int DOUBLE4 =                0x00B2;
    const unsigned int DOUBLE8 =                0x00B3;
    const unsigned int DOUBLE16 =               0x00B4;

    const unsigned int CUSTOMED =               0x00C0;

} // namespace type_name

namespace variable_scope
{
    const unsigned int DEFAULT =                0x0000;
    const unsigned int PRIVATE =                0x0010;
    const unsigned int LOCAL =                  0x0100;
    const unsigned int GLOBAL =                 0x1000;
    const unsigned int CONSTANT =               0x1001;
} // namespace variable_scope

#endif