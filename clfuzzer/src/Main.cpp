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

static llvm::cl::opt<std::string> userConfigFileName(
    "config",
    llvm::cl::desc("Specify the user config file name"),
    llvm::cl::value_desc("filename"),
    llvm::cl::Optional // Will be empty string if not specified
);

int main(int argc, const char** argv){
    clang::tooling::CommonOptionsParser optionsParser(argc, argv, ToolCategory);

    auto it = optionsParser.getSourcePathList().begin();
    std::string kernelFileName(it->c_str());

    UserConfig userConfig(userConfigFileName.c_str());
    userConfig.generateFakeHeader(kernelFileName);

    clang::tooling::ClangTool tool(optionsParser.getCompilations(), optionsParser.getSourcePathList());

    int status = rewriteOpenclKernel(&tool, &userConfig);

    UserConfig::removeFakeHeader(kernelFileName);
}