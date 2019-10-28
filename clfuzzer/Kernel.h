#if !defined(CLCOV_KERNEL_)
#define CLCOV_KERNEL_

#include <string>
#include <vector>

typedef struct Kernel_Parameter {
    std::string name;
    unsigned int type;
    unsigned int scope;
}Kernel_Parameter;

typedef struct Kernel_Info {
    std::vector<Kernel_Parameter> parameters;
    std::string name;
    
}Kernel_Info;

#endif // CLCOV_KERNEL_
