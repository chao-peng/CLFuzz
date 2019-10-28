#ifndef CLCOV_CFG_GENERATOR_HEADER_
#define CLCOV_CFG_GENERATOR_HEADER_

#include <sstream>
#include <fstream>
#include <vector>

// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Rewrite/Core/Rewriter.h"


using namespace clang::tooling;
using namespace llvm;
using namespace clang;
using namespace clang::ast_matchers;

enum CFGNodeType {
    CFGBlockNode,
    CFGConditionNode
};

enum ASTNodeType {
    IfNode,
    ForNode,
    DoWhileNode,
    WhileNode,
    CompoundNode,
    SimpleStatementNode
};

typedef std::vector<const Stmt*> Statements;
typedef std::vector<const Expr*> Expressions;

class CFGNode {
private:
    int id;
    int conditionID;
    CFGNodeType type;
    CFGNode* trueNode;
    CFGNode* falseNode;
    std::set<CFGNode*> parentNodes;
    const Expr* condition;
    Expressions expressions;
    Statements statements;
public:
    CFGNode(int _id, CFGNodeType _type) : id(_id), type(_type), trueNode(nullptr), falseNode(nullptr) {};
    int getID() const;
    int getConditionID() const;
    void setConditionID(const int& _conditionID);
    void destroy();
    void setParent(CFGNode* _parent);
    bool isCondition();
    bool conditionHasTrueBranch();
    bool conditionHasFalseBranch();
    void linkTo(CFGNode* _nodePtr, const bool& _edgeType = true);
    void appendStmt(const Stmt* stmt);
    void appendExpr(const Expr* expr);
    Statements getStatements();
    Expressions getExpressions();
    void setStatements(Statements _statements);
    void setExpressions(Expressions _expressions);
    const Expr* getCondition();
    void setCondition(const Expr* _condition);
    std::string prettyPrint(Rewriter* rewriter) const;
    void dump();
};
typedef CFGNode* CFGNodePtr;

ASTNodeType determineNodeType(const Stmt* _stmt);
bool nodeIsControlFlow(const Stmt* _stmt);
bool blockIsCondition(const int & _id);
bool conditionHasBranch(const int& _id, const bool& _branch);
void linkEdge(const int& _from, const int& _to, const bool& _edgeType = true);
void linkCFGNodes(const CFGNodePtr &_from, const CFGNodePtr &_to, const bool& _edgeType = true);
int finishPreviousBlock(const Stmt* statement);
int finishPreviousBlock(Statements statements);
int finishPreviousBlock(const Expr* expr);
int finishPreviousBlock(Expressions expressions);
int finishPreviousBlockAsCondition(const Expr* expr);
int handle_block(const CompoundStmt* _block);
int handleStmt(const Stmt* _stmt);
CFGNodePtr handleFunctionDefinition(const FunctionDecl* f, Rewriter* myRewriter);

#endif