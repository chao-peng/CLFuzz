#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "UserConfig.h"
#include "Constants.h"

UserConfig::UserConfig(std::string filename) : userConfigFileName(filename){
    numAddedLines = 0;
}

int UserConfig::generateFakeHeader(std::string kernelFileName){
    if (hasFakeHeader(kernelFileName)) {
        return error_code::KERNEL_FILE_ALREADY_HAS_FAKE_HEADER;
    }
    std::set<std::string> setMacro = this->getValues("macro");
    
    std::stringstream header;
    header << "#ifndef " << kernel_rewriter_constants::FAKE_HEADER_MACRO << "\n";
    header << "#define " << kernel_rewriter_constants::FAKE_HEADER_MACRO << "\n";
    header << "#include <opencl-c.h>\n"; // for opencl library calls
    numAddedLines += 3;

    if (!setMacro.empty()){
        for (auto it = setMacro.begin(); it != setMacro.end(); it++) {
            header << "#define " << *it << " 1\n";
            numAddedLines++;
        }
    }
    
    header << "#endif\n";
    numAddedLines++;

    std::stringstream kernelSource;
    std::fstream kernelFileStream(kernelFileName, std::ios::in);
    std::string line;
    kernelSource << header.str();
    while(std::getline(kernelFileStream, line)){
        kernelSource<<line<<"\n";
    }
    kernelFileStream.close();

    std::fstream newKernelFileStream(kernelFileName, std::ios::out);
    newKernelFileStream << kernelSource.str();
    newKernelFileStream.close();

    return error_code::STATUS_OK;
}

int UserConfig::removeFakeHeader(std::string kernelFileName){
    if (!hasFakeHeader(kernelFileName)){
        return error_code::REMOVE_KERNEL_FAKE_HEADER_FAILED_KERNEL_DOES_NOT_EXIST;
    }

    std::ifstream fileReader;
    std::ofstream fileWriter;
    std::string recoveredKernelCode;
    std::string line;
    fileReader.open(kernelFileName);

    std::string targetLine = "#ifndef ";
    targetLine.append(kernel_rewriter_constants::FAKE_HEADER_MACRO);
    std::string targetLine2 = "#endif";

    while(std::getline(fileReader, line)){
        if (line != targetLine){
            recoveredKernelCode.append(line);
            recoveredKernelCode.append("\n");
        } else {
            while(std::getline(fileReader, line)){
                if (line == targetLine2) {
                    break;
                }
            }
        }
    }
    fileReader.close();
    fileWriter.open(kernelFileName);
    fileWriter << recoveredKernelCode;
    fileWriter.close();
    return error_code::STATUS_OK;
}

bool UserConfig::hasFakeHeader(std::string kernelFileName){
    std::ifstream kernelFileStream(kernelFileName);
    std::string line;
    std::string targetLine = "#ifndef ";
    targetLine.append(kernel_rewriter_constants::FAKE_HEADER_MACRO);
    while (std::getline(kernelFileStream, line)){
        if (line.find(targetLine) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::set<std::string> UserConfig::getValues(std::string key){
    std::ifstream in(userConfigFileName);
    std::string line, line_key, line_value;
    size_t key_begin, key_length, value_begin, value_length, i, line_length;
    std::set<std::string> result;

    if (!in) {
        return result;
    }

    while(std::getline(in, line)){
        line_length = line.length();
        if (line_length == 0) continue;
        key_begin = 0;
        key_length = 0;
		while (key_begin < line_length && line.at(key_begin) == ' '){
            key_begin++;
        }
        if (key_begin >= line_length) {
            continue;
        }
		i = key_begin;
		while (i < line_length && line.at(i) != ':' && line.at(i) != ' ') {
		  i++;
		  key_length++;
		}
		if (i >= line_length)
        {
            continue;
        }
   		line_key = line.substr(key_begin, key_length);
        if (line_key != key) {
            continue;
        }

		value_begin = i;
		value_length = 0;
		while (value_begin < line_length && (line.at(value_begin) == ' ' || line.at(value_begin) == ':')) value_begin++;
		if (value_begin >= line_length){
            continue;
        }
		i = value_begin;
		while (i < line_length) {
		  i++;
		  value_length++;
		}
		line_value = line.substr(value_begin, value_length);
        result.insert(line_value);
    }
    return result;
}

std::string UserConfig::getValue(std::string key){
    std::set<std::string> tmp_value = this->getValues(key);
    std::string result;

    if (tmp_value.empty()){
        result = "";
    } else {
        result = *tmp_value.begin();
    }

    return result;
}

int UserConfig::getNumAddedLines(){
    return numAddedLines;
}

bool UserConfig::isEmpty(){
    return userConfigFileName == ""? true: false;
}
