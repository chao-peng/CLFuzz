#ifndef CLCOV_OPENCL_HOST_CODE_GENERATOR_HEADER_
#define CLCOV_OPENCL_HOST_CODE_GENERATOR_HEADER_

#include <sstream>
#include <string>
#include <map>
#include "UserConfig.h"

class HostCodeGenerator{
private:
    std::string kernelFunctionName;
    std::string branchRecorderArrayName;
    std::string barrierRecorderArrayName;
    std::string loopRecorderArrayName;
    std::string clContext;
    std::string errorCodeVariable;
    std::string clCommandQueue;
    int numConditions;
    int numBarriers;
    int numLoops;

    std::stringstream setArgumentPartHostCode;
    std::stringstream generatedHostCode;

public:
    HostCodeGenerator();

    void initialise(UserConfig* userConfig, int newNumConditions, int newNumBarriers, int newNumLoops);

    void setArgument(std::string functionName, int argumentLocation);

    void generateHostCode(std::string dataFilePath);

    bool isHostCodeComplete();

    std::string getGeneratedHostCode();
};

#endif