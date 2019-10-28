#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Basic/LLVM.h"

#include "OpenCLKernelRewriter.h"
#include "Constants.h"
#include "UserConfig.h"

using namespace clang;
using namespace clang::tooling;

std::string outputFileName;
std::string configFileName;
std::string kernelSourceFile;

// Second and main AST visitor:
// 1. Rewrite if blocks to update the local recorder array
// 2. Add the pointer to the local recorder array as the last argument to user-defined function declaration and calls
// 3. Add a loop to the end of kernel function to update the local recorder array to the global one
// 4. Add the pointer to the global recorder array as the last argument to the kernel entry function 
class RecursiveASTVisitorForKernelRewriter : public RecursiveASTVisitor<RecursiveASTVisitorForKernelRewriter> {
public:
    explicit RecursiveASTVisitorForKernelRewriter(Rewriter &r, ASTContext *astcontext) : myRewriter(r), context(astcontext){}
    
    bool VisitExpr(Expr *e) {
        if (isa<CallExpr>(e)) {
            auto begin = e->getBeginLoc();
            auto end = e->getEndLoc();
            SourceRange sr;
            sr.setBegin(begin);
            sr.setEnd(end);
            std::string functionCall = myRewriter.getRewrittenText(sr);
            std::string functionName = functionCall.substr(0, functionCall.find_first_of('('));
            if (functionName == "get_global_id" || functionName == "get_group_id") {
                auto dimStartLoc = functionCall.find_first_of('(') + 1;
                auto dimLength = functionCall.find_first_of(')') - dimStartLoc;
                std::string dimStr = functionCall.substr(dimStartLoc, dimLength);
                std::string newFunctionCall;
                if (functionName == "get_global_id") {
                    newFunctionCall = "get_global_id_new(" + dimStr + ", cl_schedule_map)";
                } else if (functionName == "get_group_id"){
                    newFunctionCall = "get_group_id_new(" + dimStr + ", cl_schedule_map)";
                }
                myRewriter.ReplaceText(sr, newFunctionCall);
            }
        }
        return true;
    }

    bool VisitFunctionDecl(FunctionDecl *f) {
        if (!f->hasBody()) {
            return true;
        }
        auto begin = f->getBeginLoc();
        auto end = f->getEndLoc();
        SourceRange sr;
        sr.setBegin(begin);
        sr.setEnd(end);
        std::string functionString = myRewriter.getRewrittenText(sr);
        std::string typeString = functionString.substr(0, 8);
        if (typeString.find("kernel") != typeString.npos){
            std::string funcFirstLine = functionString.substr(0, functionString.find_first_of('{'));
            unsigned offset = funcFirstLine.find_last_of(')');
            SourceLocation loc = f->getBeginLoc().getLocWithOffset(offset);
            myRewriter.InsertTextAfter(loc, ", __global uint3* cl_schedule_map");
        }
        return true;
    }

private:
    Rewriter &myRewriter;
    ASTContext *context;
};

class ASTConsumerForKernelRewriter : public ASTConsumer{
public:
    ASTConsumerForKernelRewriter(Rewriter &r, ASTContext* context) : visitor(r, context) {}

    bool HandleTopLevelDecl(DeclGroupRef DR) override {
        for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
            // Traverse the declaration using our AST visitor.
            visitor.TraverseDecl(*b);
            //(*b)->dump();
        }
    return true;
    }

private:
  RecursiveASTVisitorForKernelRewriter visitor;
};

class ASTFrontendActionForKernelRewriter : public ASTFrontendAction {
public:
    ASTFrontendActionForKernelRewriter(){}

    void EndSourceFileAction() override {
        const RewriteBuffer *buffer = myRewriter.getRewriteBufferFor(myRewriter.getSourceMgr().getMainFileID());
        if (buffer == NULL){
            llvm::outs() << "Rewriter buffer is null. Cannot write in file.\n";
            return;
        }
        std::string rewriteBuffer = std::string(buffer->begin(), buffer->end());
        std::string source = "";
        std::string line;
        std::istringstream bufferStream(rewriteBuffer);

        source.append(kernel_rewriter_constants::SCHEDULER_WRAPPER);

        while(getline(bufferStream, line)){
            source.append(line);
            source.append("\n");
        }

        // Write modified kernel source code
        std::ofstream fileWriter;
        fileWriter.open(outputFileName);
        fileWriter << source;
        fileWriter.close();

        if (UserConfig::hasFakeHeader(kernelSourceFile)){
            UserConfig::removeFakeHeader(kernelSourceFile);
        }

        if (UserConfig::hasFakeHeader(outputFileName)){
            UserConfig::removeFakeHeader(outputFileName);
        }
    }

    virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &ci, 
        StringRef file) override {
            std::string inputFileName = file.str();
            outputFileName = inputFileName.substr(0, inputFileName.find_last_of("."));
            outputFileName = outputFileName + "_schedule.cl";
            myRewriter.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());
            return llvm::make_unique<ASTConsumerForKernelRewriter>(myRewriter, &ci.getASTContext());
    }

private:
    Rewriter myRewriter;
};

int rewriteOpenclKernel(ClangTool* tool, UserConfig* userConfig) {

    // llvm::outs() << "Rewritting code\n";
    tool->run(newFrontendActionFactory<ASTFrontendActionForKernelRewriter>().get());

    return error_code::STATUS_OK;
}
