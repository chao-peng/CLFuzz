set(LLVM_LINK_COMPONENTS
    Support
)
        
add_clang_executable(clfuzzer
    src/CFGGenerator.cpp
    src/CFGGenerator.h
    src/Constants.h
    src/HostCodeGenerator.cpp
    src/HostCodeGenerator.h
    src/Main.cpp
    src/OpenCLKernelRewriter.cpp
    src/OpenCLKernelRewriter.h
    src/UserConfig.cpp
    src/UserConfig.h
    src/Utils.cpp
    src/Utils.h)
        
target_link_libraries(clfuzzer PRIVATE
    clangAST
    clangASTMatchers
    clangBasic
    clangFrontend
    clangTooling)
