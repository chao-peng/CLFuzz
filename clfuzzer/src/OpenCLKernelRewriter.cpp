#include <algorithm>
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

#include "CFGGenerator.h"
#include "OpenCLKernelRewriter.h"
#include "Constants.h"
#include "UserConfig.h"
#include "Utils.h"

using namespace clang;
using namespace clang::tooling;

std::string outputFileName;
std::string configFileName;
std::string kernelSourceFile;
int numAddedLines;
int numConditions; // Used for labelling if-conditions when rewriting the kernel code
int countConditions; // Used for counting if-conditions before rewriting the kernel code
std::map<int, std::string> conditionLineMap; // Line number of each condition
std::map<int, std::string> conditionStringMap; // Details of each condition
std::set<std::string> setFunctions; // A set of user-defined functions

int numBarriers;
int countBarriers;
std::map<int, std::string> barrierLineMap;

int numLoops;
int countLoops;
std::map<int, std::string> LoopMap;

std::stringstream kernelInfoBuilder;
bool measureCoverageEnabled;

// First AST visitor: counting if-conditions and user-defined functions
class RecursiveASTVisitorForKernelInvastigator : public RecursiveASTVisitor<RecursiveASTVisitorForKernelInvastigator> {
public:
    explicit RecursiveASTVisitorForKernelInvastigator(Rewriter &r) : myRewriter(r) {}

    // count the number of if-conditions
    bool VisitStmt(Stmt *s) {
        if (isa<IfStmt>(s)){
            countConditions++;
        }else if (isa<CallExpr>(s)){
            CallExpr *functionCall = cast<CallExpr>(s);
            std::string functionName = myRewriter.getRewrittenText(functionCall->getCallee()->getSourceRange());
            if (functionName == "barrier") {
                countBarriers++;
            }
        } else if (isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s)) {
            countLoops++;
        }
        return true;
    }
    
    // record user-defined functions in a set
    bool VisitFunctionDecl(FunctionDecl *f) {
        if (!f->hasBody()) {
            return true;
        }
        SourceLocation locStart, locEnd;
        SourceRange sr;
        locStart = f->getBeginLoc();
        locEnd = f->getEndLoc();
        sr.setBegin(locStart);
        sr.setEnd(locEnd);
        std::string typeString = myRewriter.getRewrittenText(sr);
        typeString = typeString.substr(0, 8);
        if (typeString.find("kernel") == typeString.npos){
            setFunctions.insert(f->getQualifiedNameAsString());
            
        } else { // is a kernel function
            kernelInfoBuilder << "num_parameters: " << f->getNumParams() << "\n";
            kernelInfoBuilder << "kernel_name: " << f->getQualifiedNameAsString() << "\n";
            kernelInfoBuilder << f->getQualifiedNameAsString() << ":\n";
            for (unsigned int i = 0; i < f->getNumParams(); ++i) {
                auto parameter = f->getParamDecl(i);
                auto parameterLoc = parameter->getSourceRange();
                std::string originalCode = myRewriter.getRewrittenText(parameterLoc);
                std::string paramName = parameter->getNameAsString();
                kernelInfoBuilder << "  " << paramName << ":\n";
                auto loc = originalCode.rfind(paramName);
                std::string substr = originalCode.substr(0, loc);
                std::vector<std::string> strs = Utils::split(substr, " ");
                bool isPointer = false;
                std::string scope_identifier;
                std::string type_identifier;
                bool isConst = false;
                bool isUnsigned = false;
                for (auto it = strs.begin(); it != strs.end(); ++it) {
                    std::string currentStr = *it;
                    Utils::trim(currentStr);
                    if (currentStr.find("*") != currentStr.npos) {
                        isPointer = true;
                        currentStr = Utils::substr_by_edge(currentStr, "", "*");
                    } 
                    Utils::trim(currentStr);
                    if (currentStr == "global" || currentStr == "__global") {
                        scope_identifier = "global";
                    } else if (currentStr == "local" || currentStr == "__local") {
                        scope_identifier = "local";
                    } else if (currentStr == "constant" || currentStr == "__constant") {
                        scope_identifier = "constant";
                    } else if (currentStr == "private" || currentStr == "__private") {
                        scope_identifier = "private";
                    } else if (currentStr == "unsigned") {
                        isUnsigned = true;
                    } else if (currentStr == "const") {
                        isConst = true;
                    } else if (currentStr != "") {
                        type_identifier = currentStr;
                    }
                }
                if (isUnsigned) {
                    type_identifier = "u" + type_identifier;
                }
                kernelInfoBuilder << "    cl_type: " << type_identifier << "\n";
                kernelInfoBuilder << "    cl_scope: " << scope_identifier << "\n";
                if (isConst) {
                    kernelInfoBuilder << "    const: true\n";
                } else {
                    kernelInfoBuilder << "    const: false\n";
                }
                if (isPointer) {
                    kernelInfoBuilder << "    pointer: true\n";
                    kernelInfoBuilder << "    size: unset\n";
                } else {
                    kernelInfoBuilder << "    pointer: false\n";
                    kernelInfoBuilder << "    size: 1\n";
                }
                kernelInfoBuilder << "    fuzzing: random\n";
                kernelInfoBuilder << "    initial_value: unset\n";
                kernelInfoBuilder << "    init_file: unset\n";
                kernelInfoBuilder << "    result: false\n";
                kernelInfoBuilder << "    pos: " << i << "\n";
            }
            CFGNodePtr cfg = handleFunctionDefinition(f, &myRewriter);
            cfg->dump();      
        }
        return true;
    }

private:
    Rewriter &myRewriter;
};

class ASTConsumerForKernelInvastigator : public ASTConsumer{
public:
    ASTConsumerForKernelInvastigator(Rewriter &r) : visitor(r) {}

    bool HandleTopLevelDecl(DeclGroupRef DR) override {
        for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
            // Traverse the declaration using our AST visitor.
            visitor.TraverseDecl(*b);
            //(*b)->dump();
        }
    return true;
    }

private:
  RecursiveASTVisitorForKernelInvastigator visitor;
};

// Before visiting the AST, add a fake header so that clang will not complain about opencl library calls and macros
class ASTFrontendActionForKernelInvastigator : public ASTFrontendAction {
public:
    ASTFrontendActionForKernelInvastigator(){}

    void EndSourceFileAction() override {
        std::ofstream fileWriter;
        std::string YAMLFileName = kernelSourceFile.substr(kernelSourceFile.find_last_of("/") + 1, kernelSourceFile.size() - kernelSourceFile.find_last_of("/") - 1) + ".yaml";
        fileWriter.open(YAMLFileName);
        if (measureCoverageEnabled) {
            fileWriter << "Cov: true\n";
        } else {
            fileWriter << "Cov: false\n";
        }
        fileWriter << "global: [1, 1, 1]\n";
        fileWriter << "local: [1, 1, 1]\n";
        fileWriter << "dim: 1\n";
        fileWriter << "Barriers: " << countBarriers << "\n";
        fileWriter << "Branches: " << countConditions * 2 << "\n";
        fileWriter << "Loops: " << countLoops << "\n";
        fileWriter << "structure_data_filename: " << outputFileName + ".dat" << "\n";

        fileWriter << kernelInfoBuilder.str();
        fileWriter.close();
    }

    virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &ci, 
        StringRef file) override {
            kernelSourceFile = file.str();
            std::string inputFileName = file.str();
            outputFileName = inputFileName.substr(0, inputFileName.find_last_of("."));
            outputFileName = outputFileName + "_cov.cl";
            //if (!UserConfig::hasFakeHeader(kernelSourceFile)){
            //    numAddedLines = UserConfig::generateFakeHeader(configFileName, kernelSourceFile);
            //}
            myRewriter.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());
            return llvm::make_unique<ASTConsumerForKernelInvastigator>(myRewriter);
        }

private:
    Rewriter myRewriter;
};

// Second and main AST visitor:
// 1. Rewrite if blocks to update the local recorder array
// 2. Add the pointer to the local recorder array as the last argument to user-defined function declaration and calls
// 3. Add a loop to the end of kernel function to update the local recorder array to the global one
// 4. Add the pointer to the global recorder array as the last argument to the kernel entry function 
class RecursiveASTVisitorForKernelRewriter : public RecursiveASTVisitor<RecursiveASTVisitorForKernelRewriter> {
public:
    explicit RecursiveASTVisitorForKernelRewriter(Rewriter &r, Rewriter &original_r, ASTContext *astcontext) : myRewriter(r), originalRewriter (original_r), context(astcontext){}
    
    bool VisitStmt(Stmt *s) {
        if (isa<IfStmt>(s)) {
            // Deal with If
            // Record details of this condition
            IfStmt *IfStatement = cast<IfStmt>(s);
            std::string locIfStatement = IfStatement->getBeginLoc().printToString(myRewriter.getSourceMgr());
            std::string conditionIfStatement = myRewriter.getRewrittenText(IfStatement->getCond()->getSourceRange());
            SourceLocation conditionStart = myRewriter.getSourceMgr().getFileLoc(IfStatement->getCond()->getBeginLoc());
            SourceLocation conditionEnd = myRewriter.getSourceMgr().getFileLoc(IfStatement->getCond()->getEndLoc());
            SourceRange conditionRange;
            conditionRange.setBegin(conditionStart);
            conditionRange.setEnd(conditionEnd);
            // Insert to the hashmap of line numbers of conditions
            // Line number needs to be adjusted due to possible added lines of fake header statements
            conditionLineMap[numConditions] = correctSourceLine(locIfStatement, numAddedLines);
            // Insert to the hashmap of text of conditions
            conditionStringMap[numConditions] = myRewriter.getRewrittenText(conditionRange);

            Stmt* Then = IfStatement->getThen();
            if(isa<CompoundStmt>(Then)) {
                // Then is a compound statement
                // Add coverage recorder to the end of the compound
                myRewriter.InsertTextAfter(
                    Then->getBeginLoc().getLocWithOffset(1),
                    stmtRecordCoverage(2 * numConditions)
                    );
            } else {
                // Then is a single statement
                // decorate it with {} and add coverage recorder
                // Need to be aware of a statement with macros
                SourceLocation startLoc = myRewriter.getSourceMgr().getFileLoc(
                    Then->getBeginLoc());
                SourceLocation endLoc = myRewriter.getSourceMgr().getFileLoc(
                    Then->getEndLoc());
                SourceRange newRange;
                newRange.setBegin(startLoc);
                newRange.setEnd(endLoc);

                std::stringstream sourcestream;
                // If there's no else block/statement, it's better add else here
                // or it might be confused with end of function and end of if
                bool hasElse = false;
                if (IfStatement->getElse()) hasElse = true;
                sourcestream << "{"
                        << stmtRecordCoverage(2 * numConditions)
                        << originalRewriter.getRewrittenText(newRange) 
                        << ";\n}";
                
                if (!hasElse){
                    sourcestream << " else { "
                        << stmtRecordCoverage(2 * numConditions + 1)
                        << "}\n";
                }
                myRewriter.ReplaceText(
                    newRange.getBegin(),
                    originalRewriter.getRewrittenText(newRange).length() + 1,
                    sourcestream.str()
                );
                if(!hasElse){
                    numConditions++;
                    return true;
                }
            }
            
            Stmt* Else = IfStatement->getElse();
            if (Else) {
                // Deal with Else
                if (isa<CompoundStmt>(Else)) {
                    // Else is a compound statement
                    // Add coverage recorder to the end of the compound
                    myRewriter.InsertTextAfter(
                        Else->getBeginLoc().getLocWithOffset(1),
                        stmtRecordCoverage(2 * numConditions + 1)
                        );
                } else if (isa<IfStmt>(Else)) {
                    // Else is another condition (else if)
                    std::stringstream ss;
                    ss << "{\n"
                        << stmtRecordCoverage(2 * numConditions + 1)
                        << "\n";
                    myRewriter.InsertTextAfter(
                        Else->getBeginLoc(),
                        ss.str()
                    );
                    myRewriter.InsertTextAfter(
                        Else->getEndLoc().getLocWithOffset(2),
                        "}\n"
                    );
                } else {
                    // Else is a single statement
                    // decorate it with {} and add coverage recorder
                    SourceLocation startLoc = myRewriter.getSourceMgr().getFileLoc(
                        Else->getBeginLoc());
                    SourceLocation endLoc = myRewriter.getSourceMgr().getFileLoc(
                        Else->getEndLoc());
                    SourceRange newRange;
                    newRange.setBegin(startLoc);
                    newRange.setEnd(endLoc);
                
                    std::stringstream sourcestream;
                    sourcestream << "{"
                        << stmtRecordCoverage(2 * numConditions + 1)
                        << myRewriter.getRewrittenText(newRange) 
                        << ";\n}";
                    myRewriter.ReplaceText(
                        newRange.getBegin(),
                        myRewriter.getRewrittenText(newRange).length() + 1,
                        sourcestream.str()
                    );
                }
                
            } else {
                // Else does not exist
                // Add corresponding else and coverage recorder in it
                std::stringstream newElse;
                newElse << "else {\n" 
                    << stmtRecordCoverage(2 * numConditions + 1)
                    << "}\n";
                myRewriter.InsertTextBefore(
                    IfStatement->getSourceRange().getEnd().getLocWithOffset(2),
                    newElse.str()
                );
            }
            
            numConditions++;
        } else if (isa<CallExpr>(s)){
            CallExpr *functionCall = cast<CallExpr>(s);
            SourceLocation startLoc = myRewriter.getSourceMgr().getFileLoc(
                    functionCall->getCallee()->getBeginLoc());
            SourceLocation endLoc = myRewriter.getSourceMgr().getFileLoc(
                    functionCall->getCallee()->getEndLoc());
            SourceRange newRange;
            newRange.setBegin(startLoc);
            newRange.setEnd(endLoc);
            std::string functionName = myRewriter.getRewrittenText(newRange);
            functionName = originalRewriter.getRewrittenText(functionCall->getCallee()->getSourceRange());
            if (setFunctions.find(functionName) != setFunctions.end()){
                myRewriter.InsertTextAfter(
                    functionCall->getEndLoc().getLocWithOffset(0),
                    localRecorderArgument()
                );
            }
            if (functionName == "barrier") {
                //Since people usually use OpenCL pre-defined macros as the argument of barrier
                //It's better to use SourceMgr getFileLoc here to retrieve the argument
                std::string locBarrierCall = functionCall->getBeginLoc().printToString(myRewriter.getSourceMgr());
                barrierLineMap[numBarriers] = correctSourceLine(locBarrierCall, numAddedLines);

                Expr* barrierArg = functionCall->getArg(0);
                std::stringstream newBarrierCall;
                SourceLocation barrierArgStartLoc = myRewriter.getSourceMgr().getFileLoc(barrierArg->getBeginLoc());
                SourceLocation barrierArgEndLoc = myRewriter.getSourceMgr().getFileLoc(barrierArg->getEndLoc());
                SourceRange barrierArgRange;
                barrierArgRange.setBegin(barrierArgStartLoc);
                barrierArgRange.setEnd(barrierArgEndLoc);
                newBarrierCall << "OCL_NEW_BARRIER(" << numBarriers << "," << myRewriter.getRewrittenText(barrierArgRange) << ")";
                myRewriter.ReplaceText(functionCall->getSourceRange(), newBarrierCall.str());

                numBarriers++;
            }
        } else if (isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s)) {
            std::stringstream loop_counter_init;
            const Expr* loopCond;
            const Stmt* loopBody;
            int offsetForRewrittingCond = 1;
            // Before the loop, initialise loop counter and boundary recorder
            // Need to determine if the loop needs surrounding curly braces e.g.
            /*
                if (COND)
                    // location where initialisation statements will be added
                    for (;;)
            */
            // However if-visitor will add curly braces
            /*
            const auto& parents = context->getParents(*s);
            const Stmt* ST = parents[0].get<Stmt>();
            if (!isa<CompoundStmt>(ST) && !isa<IfStmt>(ST)) {
                loop_counter_init << "{\n";
            }
            */
            loop_counter_init << kernel_rewriter_constants::PRIVATE_LOOP_ITERATION_COUNTER << "[" << numLoops << "] = 0;\n"
                            << kernel_rewriter_constants::PRIVATE_LOOP_BOUNDARY_RECORDER << "[" << numLoops << "] = true;\n";
            myRewriter.InsertTextBefore(s->getBeginLoc(), loop_counter_init.str());
            if (isa<ForStmt>(s)) {
                ForStmt *forLoop = cast<ForStmt>(s);
                //const Stmt* forInit = forLoop->getInit();
                loopCond = forLoop->getCond();
                //const Expr* forInc = forLoop->getInc();
                loopBody = forLoop->getBody();
            }

            if (isa<WhileStmt>(s)) {
                WhileStmt *whileLoop = cast<WhileStmt>(s);
                loopCond = whileLoop->getCond();
                loopBody = whileLoop->getBody();
            }

            if (isa<DoStmt>(s)) {
                DoStmt* doLoop = cast<DoStmt>(s);
                loopCond = doLoop->getCond();
                loopBody = doLoop->getBody();
                offsetForRewrittingCond = 2;
            }

            SourceLocation loopBodyStartLoc = myRewriter.getSourceMgr().getFileLoc(loopBody->getBeginLoc());
            SourceLocation loopBodyEndLoc = myRewriter.getSourceMgr().getFileLoc(loopBody->getEndLoc());
            SourceRange loopBodyFileRange;
            loopBodyFileRange.setBegin(loopBodyStartLoc);
            loopBodyFileRange.setEnd(loopBodyEndLoc);

            // If the body of the foor loop is not compound, we need to add extra curly braces
            if (isa<CompoundStmt>(loopBody)){ // Compound body
                myRewriter.InsertTextAfter(
                    loopBodyFileRange.getBegin().getLocWithOffset(1),
                    stmtLoopIterInc(numLoops)
                );
            } else { // Single Statement
                std::stringstream sourceStrStream;
                sourceStrStream << "{\n"
                                << stmtLoopIterInc(numLoops)
                                << myRewriter.getRewrittenText(loopBodyFileRange) << "\n"
                                << "}\n";
                myRewriter.ReplaceText(loopBodyFileRange.getBegin(),
                                        originalRewriter.getRewrittenText(loopBodyFileRange).length()+1,
                                        sourceStrStream.str());
            }

            // update the loop array
            

            myRewriter.InsertTextAfter(s->getEndLoc().getLocWithOffset(offsetForRewrittingCond), stmtRecordLoopExecStatus(numLoops));
            
            /*
            if (!isa<CompoundStmt>(ST) && !isa<IfStmt>(ST)) {
                myRewriter.InsertTextAfter(s->getEndLoc(), "}\n");
            }
            */

            // rewrite the condition stmt of the loop
            SourceLocation loopCondStartLoc = myRewriter.getSourceMgr().getFileLoc(loopCond->getBeginLoc());
            SourceLocation loopCondEndLoc = myRewriter.getSourceMgr().getFileLoc(loopCond->getEndLoc());
            SourceRange loopCondFileRange;
            loopCondFileRange.setBegin(loopCondStartLoc);
            loopCondFileRange.setEnd(loopCondEndLoc);
            std::stringstream newCondExpr;
            newCondExpr << myRewriter.getRewrittenText(loopCondFileRange) << exprLoopBoundaryReached(numLoops);
            myRewriter.ReplaceText(loopCondFileRange, newCondExpr.str());
            //myRewriter.InsertTextAfter(loopCond->getEndLoc(), exprLoopBoundaryReached(numLoops));
            numLoops++;
        }
        
        return true;
    }

    bool VisitFunctionDecl(FunctionDecl *f){
        // Need to deal with 4 possible types of function declarations
        // 1. __kernel function - add both __global parameter and __local array definition
        // 2. __kernel function prototype - add _global parameter
        // 3. non-kernel function - add _local array parameter
        // 4. non-kernel function prototype - same of 3

        // If it is a kernel function, typeString = "__kernel"
        SourceLocation locStart, locEnd;
        SourceRange sr;
        locStart = f->getBeginLoc();
        locEnd = f->getBeginLoc().getLocWithOffset(8);
        sr.setBegin(locStart);
        sr.setEnd(locEnd);
        std::string typeString = myRewriter.getRewrittenText(sr);
        std::string functionName = f->getQualifiedNameAsString();
        bool needComma = f->getNumParams() == 0? false: true;
        if (typeString == "__kernel"){
            if (f->hasBody()){
                // add global recorder array as argument to function definition
                SourceRange funcSourceRange = f->getSourceRange();
                std::string funcSourceText = myRewriter.getRewrittenText(funcSourceRange);
                std::string funcFirstLine = funcSourceText.substr(0, funcSourceText.find_first_of('{'));
                unsigned offset = funcFirstLine.find_last_of(')');
                SourceLocation loc = f->getBeginLoc().getLocWithOffset(offset);
                myRewriter.InsertTextAfter(loc, declRecorder(needComma));

                // define recorder array as __local array
                loc = f->getBody()->getBeginLoc().getLocWithOffset(1);
                myRewriter.InsertTextAfter(loc, declLocalRecorder());
                
                // update local recorder to global recorder array
                if (countConditions || countLoops){
                    loc = f->getBody()->getEndLoc();
                    bool updateBranch = (countConditions != 0);
                    bool updateLoop = (countLoops != 0);
                    myRewriter.InsertTextAfter(loc, stmtUpdateGlobalRecorder(updateBranch, updateLoop));
                }

                // Host code generator part 2: Set argument
                int argumentLocation = f->param_size();

            }
            else {
                // add global recorder array as argument to function prototype
                SourceLocation loc = f->getEndLoc();
                myRewriter.InsertTextBefore(loc, declRecorder(needComma));
            }
        } else {
            // Not a kernel function
            if (f->hasBody()){
                // If it is a function definition
                SourceRange funcSourceRange = f->getSourceRange();
                std::string funcSourceText = myRewriter.getRewrittenText(funcSourceRange);
                std::string funcFirstLine = funcSourceText.substr(0, funcSourceText.find_first_of('{'));
                unsigned offset = funcFirstLine.find_last_of(')');
                SourceLocation loc = f->getBeginLoc().getLocWithOffset(offset);
                myRewriter.InsertTextAfter(loc, declLocalRecorderArgument(needComma));
            } else {
                // If it is a function declaration without definition
                SourceLocation loc = f->getEndLoc();
                myRewriter.InsertTextBefore(loc, declLocalRecorderArgument(needComma));
            }
        }
        return true;
    }

private:
    Rewriter &myRewriter;
    Rewriter &originalRewriter;
    ASTContext *context;

    std::string stmtRecordCoverage(const int& id){
        std::stringstream ss;
        // old implementation
        // ss << kernel_rewriter_constants::COVERAGE_RECORDER_NAME << "[" << id << "] = true;\n";
        // replaced by atomic_or operation to avoid data race
        ss << "\natomic_or(&" << kernel_rewriter_constants::LOCAL_COVERAGE_RECORDER_NAME << "[" << id << "], 1);\n";
        return ss.str();
    }

    std::string stmtLoopIterInc(const int& id) {
        std::stringstream ss;
        ss << "\n" << kernel_rewriter_constants::PRIVATE_LOOP_ITERATION_COUNTER << "[" << id << "]++;\n";
        return ss.str();
    }

    std::string stmtRecordLoopCoverage(const int& id, const int& flag) {
        std::stringstream ss;
        ss << "\natomic_or(&" << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME << "[" << id << "], " << flag << ");\n";
        return ss.str();
    }

    std::string exprLoopBoundaryReached(const int& id) {
        std::stringstream ss;
        ss << " || (" << kernel_rewriter_constants::PRIVATE_LOOP_BOUNDARY_RECORDER << "[" << id << "] = false)"; 
        return ss.str();
    }

    std::string stmtRecordLoopExecStatus(const int& id) {
        std::stringstream ss;
        ss  << "\n"
            << "if (" << kernel_rewriter_constants::PRIVATE_LOOP_ITERATION_COUNTER << "[" << id << "] == 0) {\n"
            << "    atomic_or(&" << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME << "[" << id << "], 1);\n}"
            << "if (" << kernel_rewriter_constants::PRIVATE_LOOP_ITERATION_COUNTER << "[" << id << "] == 1) {\n"
            << "    atomic_or(&" << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME << "[" << id << "], 2);\n}"
            << "if (" << kernel_rewriter_constants::PRIVATE_LOOP_ITERATION_COUNTER << "[" << id << "] > 1) {\n"
            << "    atomic_or(&" << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME << "[" << id << "], 4);\n}"
            << "if (!" << kernel_rewriter_constants::PRIVATE_LOOP_BOUNDARY_RECORDER << "[" << id << "]) {\n"
            << "    atomic_or(&" << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME << "[" << id << "], 8);\n}";
        return ss.str();
    }

    std::string declRecorder(bool needComma=true){
        std::stringstream ss;

        if (countConditions) {
            if (needComma) {
                ss << ", __global int* " << kernel_rewriter_constants::GLOBAL_COVERAGE_RECORDER_NAME;
            } else {
                ss << "__global int* " << kernel_rewriter_constants::GLOBAL_COVERAGE_RECORDER_NAME;
            }
            needComma = true;
        }

        if (countBarriers) {
            if (needComma) {
                ss << ", __global int* " << kernel_rewriter_constants::GLOBAL_BARRIER_DIVERFENCE_RECORDER_NAME;
            } else {
                ss << "__global int* " << kernel_rewriter_constants::GLOBAL_BARRIER_DIVERFENCE_RECORDER_NAME;
            }
            needComma = true;
        }

        if (countLoops) {
            if (needComma) {
                ss << ", __global int* " << kernel_rewriter_constants::GLOBAL_LOOP_RECORDER_NAME;
            } else {
                ss << "__global int* " << kernel_rewriter_constants::GLOBAL_LOOP_RECORDER_NAME;
            }
        }
        return ss.str();
    }

    std::string declLocalRecorder(){
        std::stringstream ss;
        if (countConditions){
            ss << "__local int " << kernel_rewriter_constants::LOCAL_COVERAGE_RECORDER_NAME << "[" << 2 * countConditions << "];\n"
               << "for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < " << 2 * countConditions << "; ++ocl_kernel_init_i) {\n"
               << "    " << kernel_rewriter_constants::LOCAL_COVERAGE_RECORDER_NAME << "[ocl_kernel_init_i] = 0;\n}\n";
        }
        if (countBarriers){
            ss << "__local int " << kernel_rewriter_constants::LOCAL_BARRIER_COUNTER_NAME << "[" << countBarriers << "];\n"
               << "for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < " << countBarriers << "; ++ocl_kernel_init_i) {\n"
               << "    " << kernel_rewriter_constants::LOCAL_BARRIER_COUNTER_NAME << "[ocl_kernel_init_i] = 0;\n}\n";
        }
        if (countLoops) {
            ss << "__local int " << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME << "[" << countLoops << "];\n"
               << "for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < " << countLoops << "; ++ocl_kernel_init_i) {\n"
               << "    " << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME << "[ocl_kernel_init_i] = 0;\n}\n";
            ss << "int " << kernel_rewriter_constants::PRIVATE_LOOP_ITERATION_COUNTER << "[" << countLoops << "];\n";
            ss << "bool " << kernel_rewriter_constants::PRIVATE_LOOP_BOUNDARY_RECORDER << "[" << countLoops << "];\n";
        }
        ss << "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n";
        return ss.str();
    }

    std::string declLocalRecorderArgument(bool needComma=true){
        std::stringstream ss;

        if (countConditions) {
            if (needComma){
                ss << ", __local int* " << kernel_rewriter_constants::LOCAL_COVERAGE_RECORDER_NAME;
            } else {
                ss << "__local int* " << kernel_rewriter_constants::LOCAL_COVERAGE_RECORDER_NAME;
            }
            needComma = true;
        }

        if (countBarriers) {
            if (needComma) {
                ss << ", __global int* " << kernel_rewriter_constants::GLOBAL_BARRIER_DIVERFENCE_RECORDER_NAME
                    << ", __local int* " << kernel_rewriter_constants::LOCAL_BARRIER_COUNTER_NAME;
            } else {
                ss << "__global int* " << kernel_rewriter_constants::GLOBAL_BARRIER_DIVERFENCE_RECORDER_NAME
                    << ", __local int* " << kernel_rewriter_constants::LOCAL_BARRIER_COUNTER_NAME;
            }
            needComma = true;
        }
        
        if (countLoops) {
            if (needComma) {
                ss << ", __local int* " << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME;
            } else {
                ss << "__local int*" << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME;
            }
        }

        return ss.str();
    }

    std::string localRecorderArgument(){
        std::stringstream ss;
        if (countConditions){
            ss << ", " << kernel_rewriter_constants::LOCAL_COVERAGE_RECORDER_NAME;
        }
        if (countBarriers){
            ss << ", " << kernel_rewriter_constants::GLOBAL_BARRIER_DIVERFENCE_RECORDER_NAME
                    << ", " << kernel_rewriter_constants::LOCAL_BARRIER_COUNTER_NAME;
        }
        if (countLoops) {
            ss << ", " << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME;
        }
        return ss.str();
    }

    std::string stmtUpdateGlobalRecorder(bool updateBranch=true, bool updateLoop=true){
        std::stringstream ss;
        if (updateBranch) {
            ss << "for (int update_recorder_i = 0; update_recorder_i < " << (countConditions*2) << "; update_recorder_i++) { \n";
            ss << "  atomic_or(&" << kernel_rewriter_constants::GLOBAL_COVERAGE_RECORDER_NAME << "[update_recorder_i], " << kernel_rewriter_constants::LOCAL_COVERAGE_RECORDER_NAME << "[update_recorder_i]); \n";
            ss << "}\n";
        }
        if (updateLoop) {
            ss << "for (int update_recorder_i = 0; update_recorder_i < " << countLoops << "; update_recorder_i++) { \n";
            ss << "  atomic_or(&" << kernel_rewriter_constants::GLOBAL_LOOP_RECORDER_NAME << "[update_recorder_i], " << kernel_rewriter_constants::LOCAL_LOOP_RECORDER_NAME << "[update_recorder_i]); \n";
            ss << "}\n";
        }
        return ss.str();
    }

    std::string correctSourceLine(std::string originalSourceLine, int offset){
        size_t p1, p2;
        p1 = originalSourceLine.substr(0, originalSourceLine.find_last_of(':')).find_last_of(':') + 1;
        p2 = originalSourceLine.find_last_of(':');
        int newLineNumber = std::stoi(originalSourceLine.substr(p1, p2-p1)) - offset;
        std::string neworiginalSourceLine = originalSourceLine.substr(0, p1);
        neworiginalSourceLine.append(std::to_string(newLineNumber));
        neworiginalSourceLine.append(originalSourceLine.substr(p2));
        return neworiginalSourceLine;
    }
};

class ASTConsumerForKernelRewriter : public ASTConsumer{
public:
    ASTConsumerForKernelRewriter(Rewriter &r, Rewriter &original_r, ASTContext* context) : visitor(r, original_r, context) {}

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

        if (countBarriers){
            source.append(kernel_rewriter_constants::NEW_BARRIER_MACRO);
            source.append("\n");
        }

        while(getline(bufferStream, line)){
            source.append(line);
            source.append("\n");
        }

        // Write modified kernel source code
        std::ofstream fileWriter;
        fileWriter.open(outputFileName);
        fileWriter << source;
        fileWriter.close();
        
        // Write data file
        std::string dataFileName = outputFileName + ".dat";
        std::stringstream outputBuffer;
        fileWriter.open(dataFileName);
        for (int i = 0; i < numConditions; i++){
            outputBuffer << "Condition ID: " << i << "\n";
            outputBuffer << "Source code line: " << conditionLineMap[i] << "\n";
            outputBuffer << "Condition: " << conditionStringMap[i] << "\n";
        }
        for (int i = 0; i < countBarriers; i++){
            outputBuffer << "Barrier ID: " << i << "\n";
            outputBuffer << "Source code line: " << barrierLineMap[i] << "\n";
        }
        outputBuffer << "\n";
        fileWriter << outputBuffer.str();
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
            outputFileName = outputFileName + "_cov.cl";
            myRewriter.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());
            originalRewriter.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());
            return llvm::make_unique<ASTConsumerForKernelRewriter>(myRewriter, originalRewriter, &ci.getASTContext());
    }

private:
    Rewriter myRewriter;
    Rewriter originalRewriter;
    // need original rewriter to retrieve correct text from original code
};

int rewriteOpenclKernel(ClangTool* tool, UserConfig* userConfig) {
    numConditions = 0;
    countConditions = 0;
    countBarriers = 0;
    numBarriers = 0;
    numLoops = 0;
    countLoops = 0;
    numAddedLines = userConfig->getNumAddedLines();

    // llvm::outs() << "Stage 1/2: code invastigation\n";
    tool->run(newFrontendActionFactory<ASTFrontendActionForKernelInvastigator>().get());    
    
    // llvm::outs() << "Stage 2/2: generate code\n";
    tool->run(newFrontendActionFactory<ASTFrontendActionForKernelRewriter>().get());


    return error_code::STATUS_OK;
    
}
