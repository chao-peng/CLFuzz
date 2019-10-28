set(LLVM_LINK_COMPONENTS
    Support
)
        
add_clang_executable(clscheduler
    src/Constants.h
    src/HostCodeGenerator.cpp
    src/HostCodeGenerator.h
    src/Main.cpp
    src/OpenCLKernelRewriter.cpp
    src/OpenCLKernelRewriter.h
    src/UserConfig.cpp
    src/UserConfig.h)
        
target_link_libraries(clscheduler PRIVATE
    clangAST
    clangASTMatchers
    clangBasic
    clangFrontend
    clangTooling)
