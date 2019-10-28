#include "CFGGenerator.h"

int lastBlockNo;
int currentConditionNo;
int mostRecentIDInCodeBlocks;
std::map<int, CFGNodePtr> blocks;
std::map<int, int> trueEdges;
std::map<int, int> falseEdges;
std::map<int, bool> isWaitingBlock;
std::map<int, bool> isUnlinkedBlock;
std::list<int> conditions;
bool endBlockNeeded;
int branchDepth;
std::map<int, std::list<int>> depthInfo;

int CFGNode::getID() const {
    return id;
}

int CFGNode::getConditionID() const {
    return conditionID;
}

void CFGNode::setConditionID(const int& _conditionID) {
    conditionID = _conditionID;
}

void CFGNode::destroy() {
    if (trueNode) {
        trueNode->destroy();
        free(trueNode);
    }
    if (falseNode) {
        falseNode->destroy();
        free(falseNode);
    }
}

void CFGNode::setParent(CFGNode* _parent) {
    parentNodes.insert(_parent);
}

bool CFGNode::isCondition() {
    return type == CFGConditionNode;
}

bool CFGNode::conditionHasTrueBranch() {
    if (type != CFGConditionNode) return false;
    return trueNode != nullptr;
}

bool CFGNode::conditionHasFalseBranch() {
    if (type != CFGConditionNode) return false;
    return falseNode != nullptr;
}

void CFGNode::linkTo(CFGNode* _nodePtr, const bool& _edgeType) {
    if (_edgeType) {
        trueNode = _nodePtr;
        _nodePtr->setParent(this);
    } else {
        falseNode = _nodePtr;
        _nodePtr->setParent(this);
    }
}

void CFGNode::appendStmt(const Stmt* stmt) {
    statements.push_back(stmt);
}

void CFGNode::appendExpr(const Expr* expr) {
    expressions.push_back(expr);
}

Statements CFGNode::getStatements() {
    return statements;
}

Expressions CFGNode::getExpressions() {
    return expressions;
}

void CFGNode::setStatements(Statements _statements) {
    statements = _statements;
}

void CFGNode::setExpressions(Expressions _expressions) {
    expressions = _expressions;
}

const Expr* CFGNode::getCondition() {
    return condition;
}

void CFGNode::setCondition(const Expr* _condition) {
    condition = _condition;
}

std::string CFGNode::prettyPrint(Rewriter* rewriter) const {
    std::stringstream result;
    result << "CFG Node #" << id << "\n[Node Type] ";
    if (type == CFGConditionNode) {
        result << "Condition #" << conditionID << "\n";
        result << rewriter->getRewrittenText(condition->getSourceRange()) << "\n\n";
    } else {
        result << "Block\n";
        if (statements.size()) {
            for (auto it = statements.begin(); it != statements.end(); ++it) {
                result << rewriter->getRewrittenText((*it)->getSourceRange()) << "\n";
            }
            result << "\n";
        } else if (expressions.size()) {
            for (auto it = expressions.begin(); it != expressions.end(); ++it) {
                result << rewriter->getRewrittenText((*it)->getSourceRange()) << "\n";
            }
            result << "\n";
        } else {
            result << "[Empty block]";
        }
    }
    return result.str();
}

void CFGNode::dump() {
    llvm::outs() << "Block #" << id << "\n";
    if (trueNode) llvm::outs() << "True Node: #" << trueNode->getID() << "\n";
    if (falseNode) llvm::outs() << "False Node: #" << falseNode->getID() << "\n";
    if (parentNodes.size()) llvm::outs() << "Parent Nodes: ";
    for (auto it = parentNodes.begin(); it != parentNodes.end(); ++it) {
        llvm::outs() << "#" << (*it)->getID() << " ";
    }
    if (parentNodes.size()) llvm::outs() << "\n";
    if (trueNode) trueNode->dump();
    if (falseNode) falseNode->dump();
}

ASTNodeType determineNodeType(const Stmt* _stmt) {
    if (!_stmt) {
        llvm::outs() << "Nullptr in determineNodeType()\n";
        exit(-1);
    }
    if (isa<IfStmt>(_stmt)) {
        return IfNode;
    }
    if (isa<ForStmt>(_stmt)) {
        return ForNode;
    }
    if (isa<DoStmt>(_stmt)) {
        return DoWhileNode;
    }
    if (isa<WhileStmt>(_stmt)) {
        return WhileNode;
    }
    if (isa<CompoundStmt>(_stmt)) {
        return CompoundNode;
    }
    return SimpleStatementNode;
}

bool nodeIsControlFlow(const Stmt* _stmt) {
    ASTNodeType type = determineNodeType(_stmt);
    return type != CompoundNode && type != SimpleStatementNode;
}

bool blockIsCondition(const int & _id) {
    if (std::find(conditions.begin(), conditions.end(), _id) == conditions.end()) {
        return false;
    } else {
        return true;
    }
}

bool conditionHasBranch(const int& _id, const bool& _branch) {
    if (_branch) {
        if (trueEdges.find(_id) != trueEdges.end()) {
            return true;
        } else {
            return false;
        }
    } else {
        if (falseEdges.find(_id) != falseEdges.end()) {
            return true;
        } else {
            return false;
        }
    }
}

void linkEdge(const int& _from, const int& _to, const bool& _edgeType) {
    /*
    if (!_from) {
        llvm::outs() << "from is not yet initialised!\n";
        exit(-1);
    }
    if (!_to) {
        llvm::outs() << "to is not yet initialised!\n";
        exit(-1);
    }
    int fromID = _from->getID();
    int toID = _to->getID();
    */
    if (conditionHasBranch(_from, _edgeType)) {
        return;
    }
    if (_edgeType) {
        trueEdges[_from] = _to;
    } else {
        falseEdges[_from] = _to;
    }
}

void linkCFGNodes(const CFGNodePtr &_from, const CFGNodePtr &_to, const bool& _edgeType) {
    if (!_from) {
        llvm::outs() << "_from is not yet initialised!\n";
        exit(-1);
    }
    if (!_to) {
        llvm::outs() << "_to is not yet initialised!\n";
        exit(-1);
    }
    _from->linkTo(_to, _edgeType);
    _to->setParent(_from);
}

int finishPreviousBlock(const Stmt* statement) {
    int previousFinishedNO = lastBlockNo++;
    blocks[previousFinishedNO] = new CFGNode(previousFinishedNO, CFGBlockNode);
    std::vector<const Stmt*> statements{statement};
    blocks[previousFinishedNO]->setStatements(statements);
    return previousFinishedNO;
}

int finishPreviousBlock(Statements statements) {
    int previousFinishedNO = lastBlockNo++;
    blocks[previousFinishedNO] = new CFGNode(previousFinishedNO, CFGBlockNode);
    blocks[previousFinishedNO]->setStatements(statements);
    return previousFinishedNO;
}

int finishPreviousBlock(const Expr* expr) {
    int previousFinishedNO = lastBlockNo++;
    blocks[previousFinishedNO] = new CFGNode(previousFinishedNO, CFGBlockNode);
    Expressions expressions{expr};
    blocks[previousFinishedNO]->setExpressions(expressions);
    return previousFinishedNO;
}

int finishPreviousBlock(Expressions expressions) {
    int previousFinishedNO = lastBlockNo++;
    blocks[previousFinishedNO] = new CFGNode(previousFinishedNO, CFGBlockNode);
    blocks[previousFinishedNO]->setExpressions(expressions);
    return previousFinishedNO;
}

int finishPreviousBlockAsCondition(const Expr* expr) {
    int previousFinishedNO = lastBlockNo++;
    blocks[previousFinishedNO] = new CFGNode(previousFinishedNO, CFGConditionNode);
    blocks[previousFinishedNO]->setCondition(expr);
    return previousFinishedNO;
}

int handleStmt(const Stmt* _stmt) {
    ASTNodeType nodeType = determineNodeType(_stmt);
    int thisStmtID = lastBlockNo - 1;
    if (nodeType == IfNode) {
        const IfStmt* ifStmt = cast<IfStmt>(_stmt);
        const Expr* condition = ifStmt->getCond();
        int conditionNodeID = finishPreviousBlockAsCondition(condition);
        thisStmtID = conditionNodeID;
        conditions.push_back(thisStmtID);
        int thenNodeID;
        auto thenStmt = ifStmt->getThen();
        branchDepth++;
        if (isa<CompoundStmt>(thenStmt)) {
            auto thenBlock = cast<CompoundStmt>(thenStmt);
            thenNodeID = handle_block(thenBlock);
        } else {
            thenNodeID = handleStmt(thenStmt);
        }
        branchDepth--;
        linkEdge(conditionNodeID, thenNodeID);
        auto elseStmt = ifStmt->getElse();
        if (elseStmt != nullptr) {
            branchDepth++;
            int elseNodeID;
            if (isa<CompoundStmt>(elseStmt)) {
                auto elseBlock = cast<CompoundStmt>(elseStmt);
                elseNodeID = handle_block(elseBlock);
            } else {
                elseNodeID = handleStmt(elseStmt);
            }
            branchDepth--;
            linkEdge(conditionNodeID, elseNodeID, false);
        }
    } else if (nodeType == ForNode) {
        auto forStmt = cast<ForStmt>(_stmt);
        auto initStmt = forStmt->getInit();
        int initNodeID = finishPreviousBlock(initStmt);
        thisStmtID = initNodeID;
        auto conditionExpr = forStmt->getCond();
        int conditionNodeID = finishPreviousBlockAsCondition(conditionExpr);
        linkEdge(initNodeID, conditionNodeID);
        conditions.push_back(conditionNodeID);
        auto loopBody = forStmt->getBody();
        int bodyFirstID, bodyLastID;
        if (isa<CompoundStmt>(loopBody)) {
            auto loopBodyBlock = cast<CompoundStmt>(loopBody);
            bodyFirstID = handle_block(loopBodyBlock);
            bodyLastID = mostRecentIDInCodeBlocks;
        } else {
            bodyFirstID = handleStmt(loopBody);
            bodyLastID = bodyFirstID;
        }
        const Expr* incrementExpr = forStmt->getInc();
        int incrementNodeID = finishPreviousBlock(incrementExpr);
        linkEdge(bodyLastID, incrementNodeID);
        //linkEdge(incrementNodeID, conditionNodeID);
        linkEdge(conditionNodeID, bodyFirstID);
        //if (branchDepth != 1) {
            linkEdge(conditionNodeID, incrementNodeID + 1, false);
            linkEdge(incrementNodeID, incrementNodeID + 1);
        //}
    } else if (nodeType == DoWhileNode) {
        auto doStmt = cast<DoStmt>(_stmt);
        const Expr* doCondition = doStmt->getCond();
        auto loopBody = doStmt->getBody();
        int bodyFirstID, bodyLastID;
        if (isa<CompoundStmt>(doStmt)) {
            auto doBodyBlock = cast<CompoundStmt>(doStmt);
            bodyFirstID = handle_block(doBodyBlock);
            bodyLastID = mostRecentIDInCodeBlocks;
        } else {
            bodyFirstID = handleStmt(loopBody);
            bodyLastID = bodyFirstID;
        }
        thisStmtID = bodyFirstID;
        int conditionNodeID = finishPreviousBlockAsCondition(doCondition);
        conditions.push_back(conditionNodeID);
        linkEdge(bodyLastID, conditionNodeID);
        //linkEdge(conditionNodeID, bodyFirstID);
        if (branchDepth != 1) {
            linkEdge(conditionNodeID, conditionNodeID + 1, false);
        }
    } else if (nodeType == WhileNode) {
        auto whileStmt = cast<WhileStmt>(_stmt);
        const Expr* whileConditionExpr = whileStmt->getCond();
        int whileConditionNodeID = finishPreviousBlockAsCondition(whileConditionExpr);
        thisStmtID = whileConditionNodeID;
        conditions.push_back(whileConditionNodeID);
        auto loopBody = whileStmt->getBody();
        int bodyFirstID, bodyLastID;
        if (isa<CompoundStmt>(loopBody)) {
            auto loopBodyBlock = cast<CompoundStmt>(loopBody);
            bodyFirstID = handle_block(loopBodyBlock);
            bodyLastID = mostRecentIDInCodeBlocks;
        } else {
            bodyFirstID = handleStmt(loopBody);
            bodyLastID = bodyFirstID;
        }
        //linkEdge(bodyLastID, whileConditionNodeID);
        linkEdge(whileConditionNodeID, bodyFirstID);
        if (branchDepth != 1) {
            linkEdge(whileConditionNodeID, bodyLastID + 1, false);
        }
    } else {
        thisStmtID = finishPreviousBlock(_stmt);
    }
    return thisStmtID;
}

int handle_block(const CompoundStmt* _block) {
    int mostRecentID = lastBlockNo - 1;
    int thisBlockFirstID = -100;
    int previousFinishedID = -100;
    
    for (unsigned i = 0; i < _block->size(); ++i) {
        Statements currentBasicBlock;
        const Stmt* currentNode;
        while (i < _block->size()) {
            currentNode = _block->body_begin()[i];
            if (!nodeIsControlFlow(currentNode)) {
                currentBasicBlock.push_back(currentNode);
                ++i;
            } else {
                break;
            }
        }

        if (currentBasicBlock.size() || _block->size() == 0) {
            mostRecentID = finishPreviousBlock(currentBasicBlock);
            previousFinishedID = mostRecentID;
            if (thisBlockFirstID == -100) {
                thisBlockFirstID = mostRecentID;
            }
        }

        ASTNodeType currentNodeType = determineNodeType(currentNode);
        if (currentNodeType == IfNode) {
            mostRecentID = handleStmt(currentNode);
            if (previousFinishedID != -100 && std::find(conditions.begin(), conditions.end(), previousFinishedID) == conditions.end()) {
                linkEdge(previousFinishedID, mostRecentID);
                previousFinishedID = mostRecentID;
            }
            if (thisBlockFirstID == -100) {
                thisBlockFirstID = mostRecentID;
            }
        } else if (currentNodeType == ForNode) {
            mostRecentID = handleStmt(currentNode);
            if (previousFinishedID != -100) {
                linkEdge(previousFinishedID, mostRecentID);
                previousFinishedID = mostRecentID;
            }
            if (thisBlockFirstID == -100) {
                thisBlockFirstID = mostRecentID;
            }
        } else if (currentNodeType == DoWhileNode) {
            mostRecentID = handleStmt(currentNode);
            if (previousFinishedID != -100) {
                linkEdge(previousFinishedID, mostRecentID);
                previousFinishedID = mostRecentID;
            }
            if (thisBlockFirstID == -100) {
                thisBlockFirstID = mostRecentID;
            }
        } else if (currentNodeType == WhileNode) {
            mostRecentID = handleStmt(currentNode);
            if (previousFinishedID != -100 && std::find(conditions.begin(), conditions.end(), previousFinishedID) == conditions.end()) {
                linkEdge(previousFinishedID, mostRecentID);
                previousFinishedID = mostRecentID;
            }
            if (thisBlockFirstID == -100) {
                thisBlockFirstID = mostRecentID;
            }
        } 
    }
    mostRecentIDInCodeBlocks = mostRecentID;
    return thisBlockFirstID;
}

CFGNodePtr handleFunctionDefinition(const FunctionDecl* f, Rewriter* myRewriter) {
    auto functionBody = f->getBody();
    Expressions functionParameters;
    if (isa<CompoundStmt>(functionBody)) {
        auto functionBodyCompound = cast<CompoundStmt>(functionBody);
        handle_block(functionBodyCompound);
        auto lastNode = functionBodyCompound->body_back();
        if (nodeIsControlFlow(lastNode)) {
            int id = lastBlockNo++;
            blocks[id] = new CFGNode(id, CFGBlockNode);
        }
    }

    std::stringstream cfgstream;
    std::ofstream output("clfuzzer_tmp_cfg_000.gv");
    cfgstream << "digraph CFG{\n";

    int conditionID =  0;
    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        std::string shape = "shape=\"box\",style=\"rounded\"";
        if (std::find(conditions.begin(), conditions.end(), it->first) != conditions.end()) {
            shape = "shape=\"diamond\"";
            it->second->setConditionID(conditionID++);
        }
        isWaitingBlock[it->first] = true;
        isUnlinkedBlock[it->first] = true;
        cfgstream << "    nd_" << it->first << " [" << shape << ",labelloc=l, label=\"" << it->second->prettyPrint(myRewriter) << "\"];\n";
    }

    for (auto it = trueEdges.begin(); it != trueEdges.end(); ++it) {
        if (std::find(conditions.begin(), conditions.end(), it->first) != conditions.end()) {
            linkCFGNodes(blocks[it->first], blocks[it->second]);
            isUnlinkedBlock[it->second] = false;
            cfgstream << "    nd_" << it->first << " -> nd_" << it->second << "[label=\"True\"];\n";
        } else {
            linkCFGNodes(blocks[it->first], blocks[it->second]);
            isWaitingBlock[it->first] = false;
            isUnlinkedBlock[it->second] = false;
            cfgstream << "    nd_" << it->first << " -> nd_" << it->second << ";\n"; 
        }
    }

    for (auto it = falseEdges.begin(); it != falseEdges.end(); ++it) {
        linkCFGNodes(blocks[it->first], blocks[it->second], false);
        cfgstream << "    nd_" << it->first << " -> nd_" << it->second << "[label=\"False\"];\n";
        isWaitingBlock[it->first] = false;
        isUnlinkedBlock[it->second] = false;
    }

    isWaitingBlock[isWaitingBlock.size()-1] = false;
    
    for (auto it = isWaitingBlock.begin(); it != isWaitingBlock.end(); ++it) {
        if (it->second) {
            int id = it->first + 1;
            while (id < blocks.size() && !isUnlinkedBlock[id]) {
                id++;
            }
            if (id >= blocks.size()) id = blocks.size()-1;
            if (blockIsCondition(it->first)) {
                linkCFGNodes(blocks[it->first], blocks[id], false);
                cfgstream << "    nd_" << it->first << " -> nd_" << id << "[label=\"False\"];\n";
            } else {
                linkCFGNodes(blocks[it->first], blocks[id]);
                cfgstream << "    nd_" << it->first << " -> nd_" << id << ";\n";
            }
        }
    }

    cfgstream << "}\n";
    output << cfgstream.str();
    output.close();
    system("dot clfuzzer_tmp_cfg_000.gv -Tpdf -o clfuzzer_cfg.pdf");

    return blocks[0];
}