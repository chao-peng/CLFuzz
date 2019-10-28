#ifndef CLCOV_OPENCL_KERNEL_REWRITER_HEADER_
#define CLCOV_OPENCL_KERNEL_REWRITER_HEADER_

#include <string>
#include <map>

#include "clang/Tooling/Tooling.h"
#include "UserConfig.h"


int rewriteOpenclKernel(clang::tooling::ClangTool* tool, UserConfig* userconfig);
#endif