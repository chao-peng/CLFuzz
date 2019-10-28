#ifndef CLCOV_USER_CONFIG_HEADER_
#define CLCOV_USER_CONFIG_HEADER_

#include <string>
#include <set>

class UserConfig{
private:
    std::string userConfigFileName;
    unsigned int numAddedLines;

public:
    
    // Constructor function with config filename
    UserConfig(std::string filename);

    //Generate the fake header with macros specified by the user
    int generateFakeHeader(std::string kernelFileName);

    static int removeFakeHeader(std::string kernelFileName);

    static bool hasFakeHeader(std::string kernelFileName);

    std::set<std::string> getValues(std::string key);

    std::string getValue(std::string key);

    int getNumAddedLines();

    bool isEmpty();
};

#endif