#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include "llvm/Support/CommandLine.h"
#include "clang/Tooling/CommonOptionsParser.h"

#include "HostCodeGenerator.h"
#include "OpenCLKernelRewriter.h"
#include "Constants.h"
#include "UserConfig.h"

static llvm::cl::OptionCategory ToolCategory("OpenCL kernel branch coverage checker options");

int main(int argc, const char** argv){
    clang::tooling::CommonOptionsParser optionsParser(argc, argv, ToolCategory);

    auto it = optionsParser.getSourcePathList().begin();
    std::string kernelFileName(it->c_str());

    UserConfig userConfig;
    userConfig.generateFakeHeader(kernelFileName);

    clang::tooling::ClangTool tool(optionsParser.getCompilations(), optionsParser.getSourcePathList());

    int status = rewriteOpenclKernel(&tool, &userConfig);
    std::cout << "\x1B[32mDone. Please find rewritten kernel code in the output directory.\x1B[0m\n";

    UserConfig::removeFakeHeader(kernelFileName);
}