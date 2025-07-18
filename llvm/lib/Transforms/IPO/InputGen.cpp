//===-- Instrumentor.cpp - Highly configurable instrumentation pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InputGen.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GenericDomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Transforms/IPO/Instrumentor.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeMoverUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cstdint>
#include <functional>
#include <optional>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "input-gen"

static cl::opt<IGIMode> ClInstrumentationMode(
    "input-gen-mode", cl::desc("input-gen instrumentation mode"), cl::Hidden,
    cl::init(IGIMode::Disabled),
    cl::values(clEnumValN(IGIMode::Disabled, "disable", ""),
               clEnumValN(IGIMode::Record, "record", ""),
               clEnumValN(IGIMode::Generate, "generate", ""),
               clEnumValN(IGIMode::ReplayGenerated, "replay_generated", ""),
               clEnumValN(IGIMode::ReplayRecorded, "replay_recorded", "")));

static cl::list<std::string>
    AllowedExternalFuncs("input-gen-allow-external-funcs",
                         cl::desc("Specify allowed external function(s)"),
                         cl::Hidden);

static cl::list<std::string>
    EntryFunctionNames("input-gen-entry-function",
                       cl::desc("Tag the provided functions as entries."),
                       cl::Hidden);

static cl::opt<bool>
    EntryAllFunctions("input-gen-entry-all-functions",
                      cl::desc("Tag all function definitions as entries."),
                      cl::init(false), cl::Hidden);

#ifndef NDEBUG
static cl::opt<std::string>
    ClGenerateStubs("input-gen-generate-stubs",
                    cl::desc("Filename to generate the stubs for the input-gen "
                             "runtime in. Leave blank to disable."),
                    cl::Hidden);
#else
static constexpr char ClGenerateStubs[] = "";
#endif

static constexpr char InputGenRecordPathEnvVar[] = "INPUTGEN_RECORD_DUMP_PATH";
static constexpr char InputGenRuntimePrefix[] = "__ig_";
static constexpr char InputGenIndirectCalleeCandidateGlobalName[] =
    "__ig_indirect_callee_candidates";

namespace {

struct InputGenMemoryImpl;

struct BranchConditionInfo {
  struct ParameterInfo {
    enum KindTy { INST, ARG, LOAD, MEMCMP, STRCMP } Kind;
    Value *const V;
    Value *const Ptr1 = nullptr;
    Value *const Ptr2 = nullptr;
    const uint32_t TypeId = 0;
    const uint32_t Size = 0;
    using ArgumentMapTy = DenseMap<Value *, uint32_t>;
    ParameterInfo(Argument &A)
        : Kind(ARG), V(&A), TypeId(A.getType()->getTypeID()) {}
    ParameterInfo(Instruction &I)
        : Kind(INST), V(&I), TypeId(I.getType()->getTypeID()) {}
    ParameterInfo(LoadInst &LI, const DataLayout &DL)
        : Kind(LOAD), V(&LI), Ptr1(LI.getPointerOperand()),
          TypeId(LI.getType()->getTypeID()),
          Size(DL.getTypeStoreSize(LI.getType())) {}
    ParameterInfo(CallInst &CI, const DataLayout &DL)
        : Kind(MEMCMP), V(CI.getArgOperand(2)), Ptr1(CI.getArgOperand(0)),
          Ptr2(CI.getArgOperand(1)) {}
    ParameterInfo(KindTy K, Value *V, Value *Ptr1, Value *Ptr2)
        : Kind(K), V(V), Ptr1(Ptr1), Ptr2(Ptr2) {}
  };
  uint32_t No = 0;
  SmallVector<ParameterInfo> ParameterInfos;
  Function *Fn;
};

struct InputGenInstrumentationConfig : public InstrumentationConfig {

  InputGenInstrumentationConfig(InputGenMemoryImpl &IGMI, Module &M,
                                ModuleAnalysisManager &MAM, IGIMode Mode);
  virtual ~InputGenInstrumentationConfig() {}

  void populate(InstrumentorIRBuilderTy &IRB) override;

  DenseMap<Value *, BranchConditionInfo *> BCIMap;
  BranchConditionInfo &createBCI(Value &V) {
    auto *BCI = new BranchConditionInfo;
    BCIMap[&V] = BCI;
    return *BCI;
  }
  BranchConditionInfo &getBCI(Value &V) { return *BCIMap[&V]; }

  InputGenMemoryImpl &IGMI;

  FunctionAnalysisManager &getFAM() { return FAM; };

  template <typename T> typename T::Result getAnalysis(Function &F) {
    return FAM.getResult<T>(F);
  }

  IGIMode Mode;
  FunctionAnalysisManager &FAM;
  DenseMap<BranchInst *, uint32_t> BranchMap;
};

struct InputGenInstrumentationConfig;

struct InputGenMemoryImpl {
  InputGenMemoryImpl(Module &M, ModuleAnalysisManager &MAM, IGIMode Mode)
      : M(M), MAM(MAM), Mode(Mode), IConf(*this, M, MAM, Mode) {}

  bool instrument();

  bool createPathTable();
  bool createPathTable(Function &Fn);

  bool shouldInstrumentCall(CallInst &CI);
  bool shouldInstrumentLoad(LoadInst &LI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentStore(StoreInst &SI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentAlloca(AllocaInst &AI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentGlobalVariable(GlobalVariable &GV,
                                      InstrumentorIRBuilderTy &IIRB);

private:
  bool handleDeclarations();
  bool handleDeclaration(Function &F);
  bool shouldPreserveDeclaration(Function &F);
  void stubDeclaration(Function &F);
  bool isKnownDeclaration(Function &F);
  bool rewriteKnownDeclaration(Function &F);
  bool handleIndirectCalleeCandidates();
  bool handleGlobals();

  LLVMContext &getCtx() { return M.getContext(); }

  bool isRTFunc(StringRef Name) { return Name.starts_with(IConf.getRTName()); }
  bool isRTFunc(Function &F) { return isRTFunc(F.getName()); }

  /// Generates a function declaration
  /// void gen_value(void *pointer,
  ///                int32_t value_size,
  ///                int64_t alignment,
  ///                int32_t value_type_id);
  FunctionCallee getGenValueFunc() {
    if (!StubFunction)
      StubFunction = M.getOrInsertFunction(
          IConf.getRTName("", "gen_value"), Type::getVoidTy(getCtx()),
          PointerType::get(getCtx(), 0), IntegerType::get(getCtx(), 32),
          IntegerType::get(getCtx(), 64), IntegerType::get(getCtx(), 32));
    return StubFunction;
  }
  FunctionCallee StubFunction;

  Value *genValue(IRBuilderBase &IRB, Type *Ty) {
    AllocaInst *Ptr = IRB.CreateAlloca(Ty);
    IRB.CreateCall(
        getGenValueFunc(),
        {Ptr,
         ConstantInt::get(IntegerType::get(getCtx(), 32),
                          DL.getTypeStoreSize(Ty)),
         ConstantInt::get(IntegerType::get(getCtx(), 64),
                          Ptr->getAlign().value()),
         ConstantInt::get(IntegerType::get(getCtx(), 32), Ty->getTypeID())});
    return IRB.CreateLoad(Ty, Ptr);
  }

  Module &M;
  ModuleAnalysisManager &MAM;
  const IGIMode Mode;
  InputGenInstrumentationConfig IConf;
  const DataLayout &DL = M.getDataLayout();

  /// Globals that require runtime handling.
  SmallSet<GlobalVariable *, 8> RuntimeGlobals;
};

struct InputGenEntriesImpl {
  InputGenEntriesImpl(Module &M, IGIMode Mode) : M(M), Mode(Mode) {}

  bool instrument();

private:
  void createReplayRecordedModule();
  bool createEntryPoint();
  bool createRecordingHooks();
  bool processFunctions();
  bool processOtherFunctions();
  void collectIndirectCalleeCandidates();

  Module &M;
  const IGIMode Mode;
  const DataLayout &DL = M.getDataLayout();

  // The below three vectors contain all Functions in the module.

  /// The entry point functions.
  SmallVector<Function *> EntryFunctions;
  /// Other function definitions in the module.
  SmallVector<Function *> OtherFunctions;
  /// The function declarations.
  SmallVector<Function *> DeclaredFunctions;
};

struct BranchConditionIO : public InstructionIO<Instruction::Br> {
  BranchConditionIO() : InstructionIO<Instruction::Br>(/*IsPRE*/ true) {}
  virtual ~BranchConditionIO() {};

  static bool analyzeBranch(BranchInst &BI,
                            InputGenInstrumentationConfig &IConf,
                            uint32_t BCINo);

  StringRef getName() const override { return "branch_condition_info"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    IRTArgs.push_back(IRTArg(
        PointerType::getUnqual(Ctx), "branch_condition_fn",
        "The function computing the branch condition.", IRTArg::NONE,
        [&](Value &V, Type &Ty, InstrumentationConfig &IConf,
            InstrumentorIRBuilderTy &IIRB) {
          auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
          return IGIConf.getBCI(V).Fn;
        }));
    IRTArgs.push_back(
        IRTArg(PointerType::getUnqual(Ctx), "free_values",
               "Description of the free values.", IRTArg::NONE,
               [&](Value &V, Type &Ty, InstrumentationConfig &IConf,
                   InstrumentorIRBuilderTy &IIRB) {
                 return getFreeValues(V, Ty, IConf, IIRB);
               }));
    IRTArgs.push_back(IRTArg(
        IntegerType::getInt32Ty(Ctx), "num_arguments",
        "Number of arguments of the branch condition function.", IRTArg::NONE,
        [&](Value &V, Type &Ty, InstrumentationConfig &IConf,
            InstrumentorIRBuilderTy &IIRB) {
          auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
          return ConstantInt::get(
              &Ty,
              count_if(IGIConf.getBCI(V).ParameterInfos, [](const auto &PI) {
                return PI.Kind == BranchConditionInfo::ParameterInfo::INST ||
                       PI.Kind == BranchConditionInfo::ParameterInfo::ARG;
              }));
        }));
    IRTArgs.push_back(
        IRTArg(PointerType::getUnqual(Ctx), "arguments",
               "Description of the arguments.", IRTArg::NONE,
               [&](Value &V, Type &Ty, InstrumentationConfig &IConf,
                   InstrumentorIRBuilderTy &IIRB) {
                 return getArguments(V, Ty, IConf, IIRB);
               }));
    IConf.addChoice(*this);
  }

  Value *getArguments(Value &V, Type &Ty, InstrumentationConfig &IConf,
                      InstrumentorIRBuilderTy &IIRB);
  Value *getFreeValues(Value &V, Type &Ty, InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB);
};

bool BranchConditionIO::analyzeBranch(BranchInst &BI,
                                      InputGenInstrumentationConfig &IConf,
                                      uint32_t BCINo) {
  assert(BI.isConditional() && "Expected a conditional branch!");
  if (!isa<Instruction>(BI.getCondition()))
    return false;
  auto &BCI = IConf.createBCI(BI);

  const auto &DL = BI.getDataLayout();

  SmallPtrSet<Value *, 32> Ptrs;
  DenseMap<Value *, uint32_t> UseCountMap;
  DenseMap<Value *, uint32_t> ArgumentMap;
  SmallVector<Value *> Worklist;
  auto AddValue = [&](Value *V, uint32_t IncUses) {
    uint32_t &Uses = UseCountMap[V];
    if (IncUses) {
      if (Uses++)
        return;
    } else if (--Uses)
      return;
    Worklist.push_back(V);
  };
  AddValue(cast<Instruction>(BI.getCondition()), /*IncUses=*/true);

  auto IsPotentiallyFreePtr = [&](Value *Ptr) {
    auto *Obj = getUnderlyingObjectAggressive(Ptr);
    // TODO: adjust this once we can look through store chains
    return !(isa<AllocaInst>(Obj) || isa<CallInst>(Obj) ||
             isa<GlobalValue>(Obj));
  };

  bool HasLoad = false;
  while (!Worklist.empty()) {
    auto *V = Worklist.pop_back_val();
    if (auto *A = dyn_cast<Argument>(V)) {
      if (!ArgumentMap.contains(A)) {
        ArgumentMap[A] = ArgumentMap.size();
        BCI.ParameterInfos.emplace_back(*A);
      }
      continue;
    }
    bool InstIsOK = false;
    if (auto *LI = dyn_cast<LoadInst>(V)) {
      // TODO: check for dominating accesses and base ptr origin.
      if (IsPotentiallyFreePtr(LI->getPointerOperand())) {
        BCI.ParameterInfos.emplace_back(*LI, DL);
        HasLoad = true;
        AddValue(LI->getPointerOperand(), /*IncUses=*/true);
        Ptrs.insert(LI->getPointerOperand());
        continue;
      }
    }
    if (auto *CI = dyn_cast<CallInst>(V)) {
      // TODO: use target library info here
      if (CI->getCalledFunction() &&
          CI->getCalledFunction()->getName() == "memcmp") {
        if (IsPotentiallyFreePtr(CI->getArgOperand(0)) ||
            IsPotentiallyFreePtr(CI->getArgOperand(1))) {
          BCI.ParameterInfos.emplace_back(*CI, DL);
          HasLoad = true;
          InstIsOK = true;
        }
      }
      if (CI->getCalledFunction() &&
          CI->getCalledFunction()->getName() == "strcmp") {
        if (IsPotentiallyFreePtr(CI->getArgOperand(0)) ||
            IsPotentiallyFreePtr(CI->getArgOperand(1))) {
          BCI.ParameterInfos.emplace_back(
              BranchConditionInfo::ParameterInfo::STRCMP, nullptr,
              CI->getArgOperand(0), CI->getArgOperand(1));
          HasLoad = true;
          InstIsOK = true;
        }
      }
    }
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (!InstIsOK && (I->mayHaveSideEffects() || isa<PHINode>(I) ||
                        I->mayReadFromMemory())) {
        if (!ArgumentMap.contains(I)) {
          ArgumentMap[I] = ArgumentMap.size();
          BCI.ParameterInfos.emplace_back(*I);
        }
        continue;
      }
      for (auto *Op : I->operand_values()) {
        if (auto *OpI = dyn_cast<Instruction>(Op))
          AddValue(OpI, /*IncUses=*/true);
        if (auto *OpA = dyn_cast<Argument>(Op))
          AddValue(OpA, /*IncUses=*/true);
      }
      continue;
    }
    assert(isa<Constant>(V));
  }
  if (!HasLoad)
    return false;

  SmallVector<Type *> ParameterTypes;
  for (auto &PI : BCI.ParameterInfos) {
    switch (PI.Kind) {
    case BranchConditionInfo::ParameterInfo::ARG:
    case BranchConditionInfo::ParameterInfo::INST:
      ParameterTypes.push_back(PI.V->getType());
      break;
    case BranchConditionInfo::ParameterInfo::MEMCMP:
    case BranchConditionInfo::ParameterInfo::STRCMP:
    case BranchConditionInfo::ParameterInfo::LOAD:
      break;
    }
  }

  auto &Ctx = BI.getContext();
  auto *RetTy = Type::getInt8Ty(Ctx);
  Function *BCIFn = Function::Create(
      FunctionType::get(RetTy, {PointerType::getUnqual(Ctx)}, false),
      GlobalValue::InternalLinkage, IConf.getRTName("", "branch_cond_fn"),
      BI.getModule());

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", BCIFn);
  auto *ComputeBB = BasicBlock::Create(Ctx, "compute", BCIFn);

  StructType *STy = StructType::get(Ctx, ParameterTypes, /*isPacked=*/true);
  ValueToValueMapTy VM;

  IRBuilder<> IRB(EntryBB);
  Type *PtrTy = IRB.getPtrTy();
  FunctionCallee DecodeFn = BCIFn->getParent()->getOrInsertFunction(
      IConf.getRTName("", "decode"), FunctionType::get(PtrTy, {PtrTy}, false));

  AddValue(cast<Instruction>(BI.getCondition()), /*IncUses=*/false);
  while (!Worklist.empty()) {
    auto *V = Worklist.pop_back_val();
    if (isa<Constant>(V))
      continue;

    auto AMIt = ArgumentMap.find(V);
    if (AMIt != ArgumentMap.end()) {
      auto *Ptr = IRB.CreateStructGEP(STy, BCIFn->getArg(0), AMIt->second);
      VM[V] = IRB.CreateLoad(V->getType(), Ptr);
      continue;
    }
    assert(!isa<PHINode>(V));
    assert(UseCountMap[V] == 0);

    auto *I = cast<Instruction>(V);
    auto *CloneI = I->clone();
    CloneI->insertInto(ComputeBB, ComputeBB->begin());
    CloneI->setMetadata(LLVMContext::MD_dbg, nullptr);
    if (auto *CI = dyn_cast<CallInst>(CloneI))
      if (auto *Callee = CI->getCalledFunction())
        if (Callee->getName().starts_with(IConf.getRTName()))
          CI->setCalledFunction(CI->getModule()->getOrInsertFunction(
              (Callee->getName() + "2").str(), Callee->getFunctionType()));
    // Callee->getName().drop_front(IConf.getRTName().size())));

    VM[V] = CloneI;
    for (auto *Op : I->operand_values()) {
      if (Ptrs.count(Op)) {
        auto *CI = CallInst::Create(DecodeFn, {Op}, "", ComputeBB->begin());
        VM[Op] = CI;
      }
      if (auto *OpI = dyn_cast<Instruction>(Op)) {
        AddValue(OpI, /*IncUses=*/false);
      }
      if (auto *OpA = dyn_cast<Argument>(Op))
        AddValue(OpA, /*IncUses=*/false);
    }
  }
  RemapFunction(*BCIFn, VM, RF_IgnoreMissingLocals);

  IRB.CreateBr(ComputeBB);
  ReturnInst::Create(Ctx,
                     new ZExtInst(VM[BI.getCondition()], RetTy, "", ComputeBB),
                     ComputeBB);
  BCI.No = BCINo;
  BCI.Fn = BCIFn;
  return true;
}

Value *BranchConditionIO::getArguments(Value &V, Type &Ty,
                                       InstrumentationConfig &IConf,
                                       InstrumentorIRBuilderTy &IIRB) {
  auto &BI = cast<BranchInst>(V);
  auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
  auto &BCI = IGIConf.getBCI(V);
  if (BCI.ParameterInfos.empty())
    return Constant::getNullValue(&Ty);

  SmallVector<Type *> ParameterTypes;
  SmallVector<Value *> ParameterValues;

  auto PushValue = [&](Value *V) {
    ParameterTypes.push_back(V->getType());
    ParameterValues.push_back(V);
  };

  auto &DT = IIRB.analysisGetter<DominatorTreeAnalysis>(*BI.getFunction());
  auto IP = IIRB.IRB.GetInsertPoint();
  for (auto &PI : BCI.ParameterInfos) {
    switch (PI.Kind) {
    case BranchConditionInfo::ParameterInfo::INST:
      PushValue(IIRB.IRB.getInt32(PI.TypeId));
      PushValue(IIRB.IRB.getInt32(IIRB.DL.getTypeAllocSize(PI.V->getType())));
      if (!DT.dominates(cast<Instruction>(PI.V), IP))
        IP = *cast<Instruction>(PI.V)->getInsertionPointAfterDef();
      PushValue(PI.V);
      break;
    case BranchConditionInfo::ParameterInfo::ARG:
      PushValue(IIRB.IRB.getInt32(PI.TypeId));
      PushValue(IIRB.IRB.getInt32(IIRB.DL.getTypeAllocSize(PI.V->getType())));
      PushValue(PI.V);
      break;
    case BranchConditionInfo::ParameterInfo::LOAD:
    case BranchConditionInfo::ParameterInfo::MEMCMP:
    case BranchConditionInfo::ParameterInfo::STRCMP:
      break;
    }
  }

  IIRB.IRB.SetInsertPoint(IP);

  StructType *STy =
      StructType::get(IIRB.Ctx, ParameterTypes, /*isPacked=*/true);
  auto *AI = IIRB.getAlloca(BI.getFunction(), STy);
  for (auto [Idx, V] : enumerate(ParameterValues)) {
    auto *Ptr = IIRB.IRB.CreateStructGEP(STy, AI, Idx);
    IIRB.IRB.CreateStore(V, Ptr);
  }
  return AI;
}

Value *BranchConditionIO::getFreeValues(Value &V, Type &Ty,
                                        InstrumentationConfig &IConf,
                                        InstrumentorIRBuilderTy &IIRB) {
  auto &BI = cast<BranchInst>(V);
  auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
  auto &BCI = IGIConf.getBCI(V);
  if (BCI.ParameterInfos.empty())
    return Constant::getNullValue(&Ty);

  auto GetTypeOrEquivInt = [&](Type *Ty) -> Type * {
    if (Ty->isPointerTy())
      return IIRB.IRB.getIntNTy(IIRB.DL.getPointerSizeInBits());
    return Ty;
  };

  SmallVector<Type *> ParameterTypes;
  SmallVector<Value *> ParameterValues;

  auto IP = IIRB.getBestHoistPoint(IIRB.IRB.GetInsertPoint(), HOIST_MAXIMALLY);
  auto PushValue = [&](Value *V) {
    ParameterTypes.push_back(GetTypeOrEquivInt(V->getType()));
    ParameterValues.push_back(V);
    if (auto *I = dyn_cast<Instruction>(V)) {
      auto &DT = IIRB.analysisGetter<DominatorTreeAnalysis>(*I->getFunction());
      IP = IIRB.hoistInstructionsAndAdjustIP(*I, IP, DT);
    }
  };

  uint32_t NumFVIs = 0;
  for (auto &PI : BCI.ParameterInfos) {
    switch (PI.Kind) {
    case BranchConditionInfo::ParameterInfo::INST:
    case BranchConditionInfo::ParameterInfo::ARG:
      break;
    case BranchConditionInfo::ParameterInfo::LOAD:
      NumFVIs++;
      PushValue(IIRB.IRB.getInt32(PI.Kind));
      PushValue(IIRB.IRB.getInt32(PI.TypeId));
      PushValue(IIRB.IRB.getInt32(PI.Size));
      PushValue(PI.Ptr1);
      break;
    case BranchConditionInfo::ParameterInfo::MEMCMP:
      NumFVIs++;
      PushValue(IIRB.IRB.getInt32(PI.Kind));
      PushValue(PI.V);
      PushValue(PI.Ptr1);
      PushValue(PI.Ptr2);
      break;
    case BranchConditionInfo::ParameterInfo::STRCMP:
      NumFVIs++;
      PushValue(IIRB.IRB.getInt32(PI.Kind));
      PushValue(PI.Ptr1);
      PushValue(PI.Ptr2);
      break;
    }
  }

  IIRB.IRB.SetInsertPoint(IP);

  StructType *STy =
      StructType::get(IIRB.Ctx, ParameterTypes, /*isPacked=*/true);
  auto *AI = IIRB.getAlloca(BI.getFunction(), STy);
  for (auto [Idx, V] : enumerate(ParameterValues)) {
    auto *Ptr = IIRB.IRB.CreateStructGEP(STy, AI, Idx);
    IIRB.IRB.CreateStore(V, Ptr);
  }

  auto *FnTy = FunctionType::get(
      IIRB.PtrTy, {IIRB.Int32Ty, IIRB.Int32Ty, IIRB.PtrTy}, false);
  auto Fn = BI.getModule()->getOrInsertFunction(
      IConf.getRTName("register_", getName()), FnTy);
  auto *CI = IIRB.IRB.CreateCall(Fn, {IIRB.IRB.getInt32(IGIConf.getBCI(V).No),
                                      IIRB.IRB.getInt32(NumFVIs), AI});
  return CI;
}

bool InputGenMemoryImpl::shouldInstrumentLoad(LoadInst &LI,
                                              InstrumentorIRBuilderTy &IIRB) {
  if (auto *AI = dyn_cast<AllocaInst>(LI.getPointerOperand()))
    return shouldInstrumentAlloca(*AI, IIRB);
  return true;
}

bool InputGenMemoryImpl::shouldInstrumentStore(StoreInst &SI,
                                               InstrumentorIRBuilderTy &IIRB) {
  if (auto *AI = dyn_cast<AllocaInst>(SI.getPointerOperand()))
    return shouldInstrumentAlloca(*AI, IIRB);
  return true;
}

bool InputGenMemoryImpl::shouldInstrumentAlloca(AllocaInst &AI,
                                                InstrumentorIRBuilderTy &IIRB) {
  // TODO: look trough transitive users.
  auto IsUseOK = [&](Use &U) -> bool {
    if (auto *SI = dyn_cast<StoreInst>(U.getUser())) {
      if (SI->getPointerOperandIndex() == U.getOperandNo() &&
          AI.getAllocationSize(DL) >=
              DL.getTypeStoreSize(SI->getValueOperand()->getType()))
        return false;
    }
    if (auto *LI = dyn_cast<LoadInst>(U.getUser())) {
      if (LI->getPointerOperandIndex() == U.getOperandNo() &&
          AI.getAllocationSize(DL) >= DL.getTypeStoreSize(LI->getType()))
        return false;
    }
    return true;
  };
  return all_of(AI.uses(), IsUseOK);
}

bool InputGenMemoryImpl::shouldInstrumentCall(CallInst &CI) {
  if (isRTFunc(*CI.getCaller()) &&
      !CI.getCaller()->hasFnAttribute("instrument"))
    return false;
  auto *Callee = CI.getCalledFunction();
  if (!Callee)
    return (CI.mayHaveSideEffects() || CI.mayReadFromMemory()) && CI.arg_size();
  if (!Callee->isDeclaration())
    return false;
  if (!CI.getType()->isPointerTy() && none_of(CI.args(), [](Value *Arg) {
        return Arg->getType()->isPointerTy();
      }))
    return false;
  if (auto *II = dyn_cast<IntrinsicInst>(&CI)) {
    if (II->isAssumeLikeIntrinsic())
      return false;
  }
  if (isRTFunc(*Callee))
    return false;
  return true;
}

bool InputGenMemoryImpl::shouldInstrumentGlobalVariable(
    GlobalVariable &GV, InstrumentorIRBuilderTy &IIRB) {
  return RuntimeGlobals.contains(&GV);
}

bool InputGenMemoryImpl::createPathTable(Function &Fn) {
  bool Changed = false;

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto &LI = FAM.getResult<LoopAnalysis>(Fn);
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(Fn);
  auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(Fn);
  assert(!mayContainIrreducibleControl(Fn, &LI) && "TODO");
  DomTreeUpdater DTU(&DT, &PDT, DomTreeUpdater::UpdateStrategy::Eager);

  DenseMap<BasicBlock *, SmallVector<std::pair<Value *, uint32_t>>> CaseMap;
  for (auto &BB : Fn) {
    auto *SI = dyn_cast<SwitchInst>(BB.getTerminator());
    if (!SI)
      continue;
    CaseMap.clear();
    for (auto &Case : SI->cases())
      CaseMap[Case.getCaseSuccessor()].push_back(
          {Case.getCaseValue(), Case.getCaseIndex()});
    auto *OldSwitchBB = &BB;
    for (auto &[CaseBB, Values] : CaseMap) {
      auto *NewSwitchBB = SplitBlock(OldSwitchBB, SI, &DTU);
      auto NewTIIP = OldSwitchBB->getTerminator()->getIterator();
      auto *EdgeBB = SplitEdge(SI->getParent(), CaseBB, &DT);
      auto *EdgePDTNode = PDT.addNewBlock(EdgeBB, CaseBB);
      EdgePDTNode->setIDom(PDT.getNode(CaseBB));
      Value *Cond = nullptr;
      for (auto [V, Idx] : Values) {
        auto *NewCond =
            new ICmpInst(NewTIIP, ICmpInst::ICMP_EQ, SI->getCondition(), V);
        if (Cond)
          Cond = BinaryOperator::Create(Instruction::Or, Cond, NewCond, "",
                                        NewTIIP);
        else
          Cond = NewCond;
        (SI->case_begin() + Idx)->setSuccessor(SI->getDefaultDest());
      }
      BranchInst::Create(EdgeBB, NewSwitchBB, Cond, NewTIIP);
      DT.getNode(EdgeBB)->setIDom(DT.getNode(OldSwitchBB));
      NewTIIP->eraseFromParent();
      DT.deleteEdge(NewSwitchBB, EdgeBB);
      PDT.getNode(OldSwitchBB)->setIDom(PDT.getNode(NewSwitchBB)->getIDom());
      OldSwitchBB = NewSwitchBB;
    }
    BranchInst::Create(SI->getDefaultDest(), SI->getIterator());
    PDT.getNode(OldSwitchBB)->setIDom(PDT.getNode(SI->getDefaultDest()));
    assert(OldSwitchBB == SI->getParent());
    SI->eraseFromParent();
  }
  assert(DT.verify());
  assert(PDT.verify());
  LI.releaseMemory();
  LI.analyze(DT);
  LI.verify(DT);

  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(Fn);
  SE.verify();

  auto &EntryBB = Fn.getEntryBlock();

  uint32_t NumBranches = 0;
  auto HandleBlock = [&](BasicBlock &BB) {
    if (&BB == &EntryBB)
      return;
    errs() << "Compute conditions for " << BB.getName() << "\n";
    auto ControlConditionsOrNone =
        ControlConditions::collectControlConditions(BB, EntryBB, DT, PDT, 0);
    if (!ControlConditionsOrNone) {
      errs() << " - <none>\n";
      return;
    }
    for (auto &CC : ControlConditionsOrNone->getControlConditions()) {
      auto *BI = cast<BranchInst>(CC.getPointer());
      if (IConf.BranchMap.count(BI)) {
        errs() << " - CC: " << CC << " [known]\n";
        continue;
      }
      if (BranchConditionIO::analyzeBranch(*BI, IConf, NumBranches)) {
        IConf.BranchMap.insert({BI, NumBranches++});
        errs() << " - CC: " << CC << " [valid]\n";
        continue;
      }
      errs() << " - CC: " << CC << " [ignored]\n";
    }
  };

  SmallVector<BasicBlock *> ReturnBlocks, UnreachableBlocks;
  SetVector<DomTreeNodeBase<BasicBlock> *> PDTLevel, PDTNextLevel;
  for (auto *RootBB : PDT.roots()) {
    auto *TI = RootBB->getTerminator();
    if (TI->getNumSuccessors()) {
      errs() << "endless loop: " << *TI << "\n";
      llvm_unreachable("endless loops");
    }
    if (isa<ReturnInst>(TI))
      ReturnBlocks.push_back(RootBB);
    else if (isa<UnreachableInst>(TI))
      UnreachableBlocks.push_back(RootBB);
    else {
      errs() << "Unhandled terminator: " << *TI << "\n";
      llvm_unreachable("unhandled terminator");
    }
    PDTLevel.insert(PDT.getNode(RootBB));
  }

  for (auto &BB : Fn) {
    auto *PDTNode = PDT.getNode(&BB);
    if (!PDTNode->isLeaf())
      continue;
    auto *DTNode = DT.getNode(&BB);
    if (!DTNode->isLeaf())
      continue;
    HandleBlock(BB);
  }

  return Changed;
}

bool InputGenMemoryImpl::createPathTable() {
  bool Changed = false;

  for (auto &Fn : M) {
    if (!Fn.isDeclaration() && Fn.getName().ends_with(".wrapper"))
      Changed |= createPathTable(Fn);
  }

  return Changed;
}

bool isPersonalityFunction(Function &F) {
  return !F.use_empty() && all_of(F.uses(), [&](Use &U) {
    if (auto *UserF = dyn_cast<Function>(U.getUser()))
      if (UserF->getPersonalityFn() == &F)
        return true;
    return false;
  });
}

bool InputGenMemoryImpl::shouldPreserveDeclaration(Function &F) {
  StringRef Name = F.getName();
  bool UserAllowedExternal = llvm::any_of(
      AllowedExternalFuncs, [&](std::string N) { return N == Name; });
  bool IsCxaThrow = Name == "__cxa_throw";
  return isPersonalityFunction(F) || F.isIntrinsic() || isRTFunc(F) ||
         UserAllowedExternal || IsCxaThrow;
}

void InputGenMemoryImpl::stubDeclaration(Function &F) {
  F.setLinkage(GlobalValue::PrivateLinkage);
  F.setVisibility(GlobalValue::DefaultVisibility);
  F.setMetadata(LLVMContext::MD_dbg, nullptr);

  auto *EntryBB = BasicBlock::Create(getCtx(), "entry", &F);

  IRBuilder<> IRB(EntryBB);
  auto *RTy = F.getReturnType();
  if (RTy->isVoidTy()) {
    IRB.CreateRetVoid();
    return;
  }

  IRB.CreateRet(genValue(IRB, RTy));

  // To generate branch hints we need to generate the value at the call site
  // scope. The above stub is still required in cases where the function's
  // address is taken so we leave it as is.
  // TODO We can make this work for invoke as well but it is slightly more
  // annoying.
  SmallVector<CallInst *> ToStub;
  for (auto *User : F.users())
    if (auto *CI = dyn_cast<CallInst>(User))
      if (CI->getCalledFunction() == &F)
        ToStub.push_back(CI);
  for (auto *CI : ToStub) {
    IRBuilder<> IRB(CI);
    Value *V = genValue(IRB, RTy);
    CI->replaceAllUsesWith(V);
    CI->eraseFromParent();
  }
}

bool InputGenMemoryImpl::isKnownDeclaration(Function &F) {
  return StringSwitch<bool>(F.getName())
      .Case("memcmp", true)
      .Case("strcmp", true)
      // .Case("__sprintf_chk", true)
      .Default(false);
}

bool InputGenMemoryImpl::rewriteKnownDeclaration(Function &F) {
  assert(F.isDeclaration());
  if (isKnownDeclaration(F)) {
    F.setName(IConf.getRTName("known_", F.getName()));
    F.setComdat(nullptr);
    return true;
  }
  return false;
}

bool InputGenMemoryImpl::handleDeclaration(Function &F) {
  if (shouldPreserveDeclaration(F))
    return false;
  if (Mode == IGIMode::Generate && rewriteKnownDeclaration(F))
    return true;
  stubDeclaration(F);
  return true;
}

bool InputGenMemoryImpl::handleDeclarations() {
  bool Changed = false;
  SmallVector<Function *> Decls;
  for (Function &F : M)
    if (F.isDeclaration())
      Decls.push_back(&F);
  for (Function *F : Decls)
    Changed |= handleDeclaration(*F);
  return Changed;
}

bool InputGenMemoryImpl::handleIndirectCalleeCandidates() {
  bool Changed = false;
  SmallSetVector<Function *, 16> IndirectCalleeCandidates;
  if (GlobalVariable *GV =
          M.getGlobalVariable(InputGenIndirectCalleeCandidateGlobalName)) {
    if (GV->hasInitializer())
      if (auto *CA = dyn_cast<ConstantArray>(GV->getInitializer()))
        for (Use &Op : CA->operands())
          IndirectCalleeCandidates.insert(cast<Function>(Op));
    assert(GV->use_empty());
    GV->eraseFromParent();
    Changed = true;
  }

  // For now we get rid of them if they are not used. We can use them once we
  // decide to implement faking indirect callees.
  for (Function *F : IndirectCalleeCandidates) {
    if (F->use_empty()) {
      F->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

// These are global variables that are never meant to be defined and are just
// used to identify types in the source language
static bool isLandingPadType(GlobalVariable &GV) {
  return !GV.use_empty() && any_of(GV.uses(), [](Use &U) {
    if (isa<LandingPadInst>(U.getUser()))
      return true;
    return false;
  });
}

bool InputGenMemoryImpl::handleGlobals() {
  bool Changed = false;

  auto Erase = [&](StringRef Name) {
    if (GlobalVariable *GV = M.getNamedGlobal(Name)) {
      GV->eraseFromParent();
      Changed = true;
    }
  };
  Erase("llvm.global_ctors");
  Erase("llvm.global_dtors");

  for (auto &GV : M.globals()) {
    if (GV.isConstant() && GV.hasInitializer()) {
      // TODO We can avoid instrumenting some of the constant globals if we can
      // prove we never access them out of bounds.
      RuntimeGlobals.insert(&GV);
      continue;
    }
    if (GV.getSection() == "llvm.metadata")
      continue;
    if (isLandingPadType(GV))
      continue;

    // We need to be able to write to constant globals without initializers as
    // the runtime will attempt to generate values for their initial state.
    GV.setConstant(false);

    // Make sure they don't clash with anything we may link in.
    GV.setLinkage(GlobalValue::PrivateLinkage);
    GV.setVisibility(GlobalValue::DefaultVisibility);

    GV.setExternallyInitialized(false);

    // Make them definitions (as opposed to declarations) by giving them an
    // initial value.
    if (GV.isDeclaration())
      GV.setInitializer(Constant::getNullValue(GV.getValueType()));

    RuntimeGlobals.insert(&GV);

    Changed = true;
  }

  // TODO need to register these GVs with the runtime

  return Changed;
}

bool InputGenMemoryImpl::instrument() {
  if (!(Mode == IGIMode::Generate || Mode == IGIMode::ReplayGenerated))
    return false;

  bool Changed = false;

  Changed |= handleIndirectCalleeCandidates();
  Changed |= handleDeclarations();
  Changed |= handleGlobals();

  if (Mode == IGIMode::Generate) {
    // TODO: HACK for qsort, we need to actually check the functions we rename
    // here and qsort explicitly.
    for (auto &Fn : M) {
      if (Fn.isDeclaration() || !Fn.hasLocalLinkage())
        continue;
      bool HasNonQSortUses = false;
      for (auto &U : Fn.uses()) {
        if (auto *CU = dyn_cast<Constant>(U.getUser()))
          if (CU->getNumUses() == 1)
            if (auto *GV = dyn_cast<GlobalVariable>(CU->user_back()))
              if (GV->getName() == "llvm.compiler.used" ||
                  GV->getName() == "llvm.used")
                continue;
        auto *CI = dyn_cast<CallInst>(U.getUser());
        if (!CI || &CI->getCalledOperandUse() == &U ||
            !CI->getCalledFunction() ||
            CI->getCalledFunction()->getName() != "qsort") {
          HasNonQSortUses = true;
          break;
        }
      }
      if (!HasNonQSortUses)
        Fn.setName(IConf.getRTName() + Fn.getName());
    }
  }

  if (Mode == IGIMode::Generate)
    Changed |= createPathTable();

  InstrumentorPass IP(&IConf);

  auto PA = IP.run(M, MAM);
  if (!PA.areAllPreserved())
    Changed = true;

  return Changed;
}

void InputGenEntriesImpl::createReplayRecordedModule() {
  assert(Mode == IGIMode::Record);

  char *DumpPathC = getenv(InputGenRecordPathEnvVar);
  if (!DumpPathC)
    return;

  if (auto EC = sys::fs::create_directories(DumpPathC)) {
    llvm::errs() << "Failed to create inputgen recording dump dir `"
                 << DumpPathC << "' " << EC.message() << "\n";
    return;
  }

  IRBuilder<> IRB(M.getContext());
  GlobalVariable *DumpPathGV = IRB.CreateGlobalString(
      DumpPathC, std::string(InputGenRuntimePrefix) + "input_dump_path", 0, &M);
  DumpPathGV->setLinkage(llvm::GlobalValue::WeakAnyLinkage);

  for (Function *EntryFunction : EntryFunctions) {
    std::unique_ptr<Module> ReplayModule = CloneModule(M);

    for (Function &F : *ReplayModule) {
      if (F.getName() != EntryFunction->getName())
        F.removeFnAttr(Attribute::InputGenEntry);
      else
        assert(F.hasFnAttribute(Attribute::InputGenEntry) &&
               !F.isDeclaration());
    }

    StringRef DumpPathSR(DumpPathC);
    SmallString<128> ThisEntryPath(DumpPathSR.begin(), DumpPathSR.end());

    InputGenEntriesImpl EReplayImpl(*ReplayModule, IGIMode::ReplayRecorded);
    EReplayImpl.instrument();

    std::error_code EC;
    sys::path::append(ThisEntryPath, EntryFunction->getName());
    if (auto EC = sys::fs::create_directories(ThisEntryPath)) {
      llvm::errs() << "Failed to create inputgen recording dump dir `"
                   << ThisEntryPath << "' " << EC.message() << "\n";
      return;
    }
    sys::path::append(ThisEntryPath, "replay_module.bc");
    llvm::raw_fd_ostream Out(ThisEntryPath, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << "Failed to open file: " << EC.message() << "\n";
      continue;
    }
    llvm::errs() << "Dumping replay module to `" << ThisEntryPath << "'\n";
    WriteBitcodeToFile(*ReplayModule, Out);
  }
}

bool InputGenEntriesImpl::instrument() {
  if (Mode != IGIMode::Disabled) {

    bool Changed = false;

    for (auto &Fn : M.functions()) {
      if (Fn.hasFnAttribute(Attribute::InputGenEntry) && !Fn.isDeclaration()) {
        EntryFunctions.push_back(&Fn);
      } else if (Fn.isDeclaration()) {
        DeclaredFunctions.push_back(&Fn);
      } else {
        OtherFunctions.push_back(&Fn);
      }
    }

    if (Mode == IGIMode::Record)
      createReplayRecordedModule();

    if (Mode == IGIMode::Generate || Mode == IGIMode::ReplayGenerated ||
        Mode == IGIMode::ReplayRecorded) {
      Changed |= createEntryPoint();
    } else if (Mode == IGIMode::Record) {
      Changed |= createRecordingHooks();
    } else {
      llvm_unreachable("??");
    }
    Changed |= processFunctions();

    return Changed;
  }
  return false;
}

void processFunctionDefinitionForGenerate(Function *F) {
  assert(!F->isDeclaration());
  // We want to aggressively inline to strengthen the InputGenMemory
  // instrumentation analysis.
  F->addFnAttr(Attribute::AlwaysInline);
  // TODO also look at the callsites for noinline
  F->removeFnAttr(Attribute::NoInline);
  // opt_none is incompatible with always_inline
  F->removeFnAttr(Attribute::OptimizeNone);

  // We do not want any definitions to clash with any other modules we may
  // link in.
  F->setLinkage(GlobalValue::PrivateLinkage);
  F->setVisibility(GlobalValue::DefaultVisibility);
}

static void processFunctionDefinitionForReplay(Function *F) {
  // We do not want any definitions to clash with any other modules we may
  // link in.
  F->setLinkage(GlobalValue::PrivateLinkage);
  F->setVisibility(GlobalValue::DefaultVisibility);
}

void InputGenEntriesImpl::collectIndirectCalleeCandidates() {
  // Since the OtherFunctions may be unused and the EntryFunctions may get
  // inlined and deleted, we need to make sure they do not get optimized away
  // before we have a chance to consider them for indirect call candidates
  // later.
  // TODO we can be smarter about it, for example only collect the external
  // functions and the internal ones that have their address taken.
  Type *ArrayEltTy = llvm::PointerType::getUnqual(M.getContext());
  SmallSetVector<Constant *, 16> Init;
  for (Function *F : llvm::concat<Function *>(EntryFunctions, OtherFunctions)) {
    assert(!F->isDeclaration());
    Init.insert(ConstantExpr::getPointerBitCastOrAddrSpaceCast(F, ArrayEltTy));
  }

  ArrayType *ATy = ArrayType::get(ArrayEltTy, Init.size());
  new llvm::GlobalVariable(M, ATy, false, GlobalValue::ExternalLinkage,
                           ConstantArray::get(ATy, Init.getArrayRef()),
                           InputGenIndirectCalleeCandidateGlobalName);
}

bool InputGenEntriesImpl::processFunctions() {
  for (Function *F : llvm::concat<Function *>(EntryFunctions, OtherFunctions)) {
    if (Mode == IGIMode::Generate)
      processFunctionDefinitionForGenerate(F);
    if (Mode == IGIMode::ReplayGenerated || Mode == IGIMode::ReplayRecorded)
      processFunctionDefinitionForReplay(F);
  }

  if (Mode == IGIMode::Generate || Mode == IGIMode::ReplayGenerated)
    collectIndirectCalleeCandidates();

  // Clean up the inputgen_entry attributes now that they are no longer needed
  for (Function &F : M)
    F.removeFnAttr(Attribute::InputGenEntry);

  return true;
}

bool InputGenEntriesImpl::createRecordingHooks() {
  auto &Ctx = M.getContext();

  FunctionCallee RecordPush = M.getOrInsertFunction(
      std::string(InputGenRuntimePrefix) + "record_push", Type::getVoidTy(Ctx),
      PointerType::get(Ctx, 0), PointerType::get(Ctx, 0));
  FunctionCallee RecordPop =
      M.getOrInsertFunction(std::string(InputGenRuntimePrefix) + "record_pop",
                            Type::getVoidTy(Ctx), PointerType::get(Ctx, 0));

  uint32_t NumEntryPoints = EntryFunctions.size();

  for (uint32_t I = 0; I < NumEntryPoints; ++I) {
    Function *EntryPoint = EntryFunctions[I];
    std::string OriginalName = EntryPoint->getName().str();
    std::string SubFuncName =
        std::string(InputGenRuntimePrefix) + "entry_" + OriginalName;

    Function *IGEntry =
        Function::Create(EntryPoint->getFunctionType(),
                         EntryPoint->getLinkage(), OriginalName, M);
    IGEntry->setVisibility(EntryPoint->getVisibility());
    EntryPoint->replaceAllUsesWith(IGEntry);
    EntryPoint->setName(SubFuncName);
    IGEntry->setName(OriginalName);

    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", IGEntry);
    IRBuilder<> IRB(Ctx);
    IRB.SetInsertPoint(EntryBB);

    unsigned ArgsMemSize = 0;
    for (auto &Arg : IGEntry->args())
      ArgsMemSize += DL.getTypeStoreSize(Arg.getType()) * 8;
    Value *ArgsMem =
        IRB.CreateAlloca(IRB.getInt8Ty(), IRB.getInt32(ArgsMemSize));
    Value *CurArgMem = ArgsMem;
    SmallVector<Value *> Args;
    auto *PtrTy = PointerType::getUnqual(Ctx);
    for (auto &Arg : IGEntry->args()) {
      Args.push_back(&Arg);
      IRB.CreateStore(&Arg, CurArgMem);
      // NOTE we mistakenly use ptr as the basis instead of i8. leave it as is
      // for now because we have a lot of inputs using this...
      CurArgMem = IRB.CreateConstGEP1_32(PtrTy, CurArgMem,
                                         DL.getTypeStoreSize(Arg.getType()));
    }

    Value *FuncNameVal = IRB.CreateGlobalString(
        OriginalName, std::string(InputGenRuntimePrefix) +
                          "ig_record_entry_name." + EntryPoint->getName());
    IRB.CreateCall(RecordPush, {FuncNameVal, ArgsMem});
    CallInst *Call = IRB.CreateCall(FunctionCallee(EntryPoint), Args);
    IRB.CreateCall(RecordPop, {FuncNameVal});

    if (Call->getType()->isVoidTy())
      IRB.CreateRetVoid();
    else
      IRB.CreateRet(Call);
  }
  return true;
}

bool InputGenEntriesImpl::createEntryPoint() {
  auto &Ctx = M.getContext();
  auto *I32Ty = IntegerType::getInt32Ty(Ctx);
  auto *PtrTy = PointerType::getUnqual(Ctx);

  uint32_t NumEntryPoints = EntryFunctions.size();
  new GlobalVariable(M, I32Ty, true, GlobalValue::ExternalLinkage,
                     ConstantInt::get(I32Ty, NumEntryPoints),
                     std::string(InputGenRuntimePrefix) + "num_entry_points");

  Function *IGEntry = Function::Create(
      FunctionType::get(Type::getVoidTy(Ctx), {I32Ty, PtrTy}, false),
      GlobalValue::ExternalLinkage,
      std::string(InputGenRuntimePrefix) + "entry", M);

  auto *EntryChoice = IGEntry->getArg(0);
  auto *EntryObj = IGEntry->getArg(1);

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", IGEntry);
  auto *ReturnBB = BasicBlock::Create(Ctx, "return", IGEntry);
  auto *SI = SwitchInst::Create(EntryChoice, ReturnBB, NumEntryPoints, EntryBB);
  ReturnInst::Create(Ctx, ReturnBB);

  SmallVector<Constant *> Names;
  IRBuilder<> IRB(SI);
  for (uint32_t I = 0; I < NumEntryPoints; ++I) {
    Function *EntryPoint = EntryFunctions[I];
    Names.push_back(IRB.CreateGlobalString(
        EntryPoint->getName(), std::string(InputGenRuntimePrefix) +
                                   "ig_entry_name." + EntryPoint->getName()));
    EntryPoint->setName(std::string(InputGenRuntimePrefix) + "ig_entry_func." +
                        EntryPoint->getName());

    Function *EntryPointWrapper = Function::Create(
        FunctionType::get(Type::getVoidTy(Ctx), {I32Ty, PtrTy}, false),
        GlobalValue::InternalLinkage, EntryPoint->getName() + ".wrapper", M);
    // Tell Instrumentor not to ignore these functions.
    EntryPoint->addFnAttr("instrument");
    EntryPointWrapper->addFnAttr("instrument");
    EntryPointWrapper->addFnAttr(Attribute::NoInline);

    auto *WrapperEntryBB = BasicBlock::Create(Ctx, "entry", EntryPointWrapper);

    SmallVector<Value *> Parameters;
    Value *WrapperObjPtr = EntryPointWrapper->getArg(1);
    for (auto &Arg : EntryPoint->args()) {
      auto *LI = new LoadInst(Arg.getType(), WrapperObjPtr, Arg.getName(),
                              WrapperEntryBB);
      Parameters.push_back(LI);
      // NOTE we mistakenly use ptr as the basis instead of i8. leave it as is
      // for now because we have a lot of inputs using this...
      WrapperObjPtr = GetElementPtrInst::Create(
          PtrTy, WrapperObjPtr,
          {ConstantInt::get(I32Ty, DL.getTypeStoreSize(Arg.getType()))}, "",
          WrapperEntryBB);
    }

    auto *CI = CallInst::Create(EntryPoint->getFunctionType(), EntryPoint,
                                Parameters, "", WrapperEntryBB);
    if (!CI->getType()->isVoidTy())
      new StoreInst(CI, WrapperObjPtr, WrapperEntryBB);
    ReturnInst::Create(Ctx, WrapperEntryBB);

    EntryPoint->addFnAttr(Attribute::AlwaysInline);
    EntryPoint->removeFnAttr(Attribute::NoInline);
    // opt_none is incompatible with always_inline
    EntryPoint->removeFnAttr(Attribute::OptimizeNone);

    auto *DispatchBB = BasicBlock::Create(Ctx, "dispatch", IGEntry);
    auto *WrapperCI = CallInst::Create(EntryPointWrapper->getFunctionType(),
                                       EntryPointWrapper,
                                       {EntryChoice, EntryObj}, "", DispatchBB);
    // Force must tail to avoid IPO, especially Argument promotion.
    WrapperCI->setTailCallKind(CallInst::TCK_MustTail);
    SI->addCase(ConstantInt::get(I32Ty, I), DispatchBB);

    ReturnInst::Create(Ctx, DispatchBB);
  }
  ArrayType *NameArrayTy = ArrayType::get(PtrTy, NumEntryPoints);
  Constant *NameArray = ConstantArray::get(NameArrayTy, Names);

  new GlobalVariable(M, NameArrayTy, true, GlobalValue::ExternalLinkage,
                     NameArray,
                     std::string(InputGenRuntimePrefix) + "entry_point_names");

  return true;
}

InputGenInstrumentationConfig::InputGenInstrumentationConfig(
    InputGenMemoryImpl &IGI, Module &M, ModuleAnalysisManager &MAM,
    IGIMode Mode)
    : InstrumentationConfig(), IGMI(IGI), Mode(Mode),
      FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()) {
  ReadConfig = false;
  RuntimePrefix->setString(InputGenRuntimePrefix);
  RuntimeStubsFile->setString(ClGenerateStubs);
}

void InputGenInstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  assert(Mode != IGIMode::Disabled);

  ModuleIO::populate(*this, IIRB.Ctx);

  GlobalIO::ConfigTy GICConfig(/*Enable=*/false);
  GICConfig.set(GlobalIO::PassAddress);
  GICConfig.set(GlobalIO::PassName);
  GICConfig.set(GlobalIO::PassInitialValue);
  GICConfig.set(GlobalIO::PassInitialValueSize);
  GICConfig.set(GlobalIO::PassIsConstant);
  if (Mode == IGIMode::Generate)
    GICConfig.set(GlobalIO::ReplaceAddress);
  auto *GIC = InstrumentationConfig::allocate<GlobalIO>();
  GIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentGlobalVariable(cast<GlobalVariable>(V), IIRB);
  };
  GIC->init(*this, IIRB.Ctx, &GICConfig);

  if (Mode != IGIMode::Generate)
    return;

  UnreachableIO::ConfigTy UIOConfig(/*Enable=*/false);
  UnreachableIO::populate(*this, IIRB.Ctx, &UIOConfig);

  BasePointerIO::ConfigTy BPIOConfig(/*Enable=*/false);
  BPIOConfig.set(BasePointerIO::PassPointer);
  BPIOConfig.set(BasePointerIO::PassPointerKind);
  BasePointerIO::populate(*this, IIRB.Ctx, &BPIOConfig);

  LoopValueRangeIO::ConfigTy LVRIOConfig(/*Enable=*/false);
  LoopValueRangeIO::populate(*this, IIRB, &LVRIOConfig);

  auto *BIC = InstrumentationConfig::allocate<BranchConditionIO>();
  BIC->HoistKind = HOIST_MAXIMALLY;
  BIC->CB = [&](Value &V) { return BranchMap.count(&cast<BranchInst>(V)); };
  BIC->init(*this, IIRB.Ctx);

  AllocaIO::ConfigTy AICConfig(/*Enable=*/false);
  AICConfig.set(AllocaIO::PassAddress);
  AICConfig.set(AllocaIO::ReplaceAddress);
  AICConfig.set(AllocaIO::PassSize);
  AICConfig.set(AllocaIO::PassAlignment);
  auto *AIC = InstrumentationConfig::allocate<AllocaIO>(/*IsPRE=*/false);
  AIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentAlloca(cast<AllocaInst>(V), IIRB);
  };
  AIC->init(*this, IIRB.Ctx, &AICConfig);

  LoadIO::ConfigTy LICConfig(/*Enable=*/false);
  LICConfig.set(LoadIO::PassPointer);
  LICConfig.set(LoadIO::ReplacePointer);
  LICConfig.set(LoadIO::PassBasePointerInfo);
  LICConfig.set(LoadIO::PassLoopValueRangeInfo);
  LICConfig.set(LoadIO::PassValueSize);
  LICConfig.set(LoadIO::PassAlignment);
  LICConfig.set(LoadIO::PassValueTypeId);
  auto *LIC = InstrumentationConfig::allocate<LoadIO>(/*IsPRE=*/true);
  LIC->HoistKind = DO_NOT_HOIST;
  LIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentLoad(cast<LoadInst>(V), IIRB);
  };
  LIC->init(*this, IIRB, &LICConfig);

  StoreIO::ConfigTy SICConfig(/*Enable=*/false);
  SICConfig.set(StoreIO::PassPointer);
  SICConfig.set(StoreIO::ReplacePointer);
  SICConfig.set(StoreIO::PassBasePointerInfo);
  SICConfig.set(StoreIO::PassLoopValueRangeInfo);
  SICConfig.set(StoreIO::PassStoredValueSize);
  SICConfig.set(StoreIO::PassAlignment);
  SICConfig.set(StoreIO::PassValueTypeId);
  auto *SIC = InstrumentationConfig::allocate<StoreIO>(/*IsPRE=*/true);
  SIC->HoistKind = DO_NOT_HOIST;
  SIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentStore(cast<StoreInst>(V), IIRB);
  };
  SIC->init(*this, IIRB, &SICConfig);

  CallIO::ConfigTy CICConfig;
  CICConfig.ArgFilter = [&](Use &Op) {
    auto *CI = cast<CallInst>(Op.getUser());
    auto &TLI = IIRB.analysisGetter<TargetLibraryAnalysis>(*CI->getFunction());
    auto ACI = getAllocationCallInfo(CI, &TLI);
    return Op->getType()->isPointerTy() || ACI;
  };
  for (bool IsPRE : {true, false}) {
    auto *CIC = InstrumentationConfig::allocate<CallIO>(IsPRE);
    CIC->CB = [&](Value &V) {
      return IGMI.shouldInstrumentCall(cast<CallInst>(V));
    };
    CIC->init(*this, IIRB.Ctx, &CICConfig);
  }
}

bool tagEntries(Module &M) {
  bool Changed = false;
  if (EntryAllFunctions) {
    for (auto &F : M) {
      if (!F.isDeclaration()) {
        F.addFnAttr(Attribute::InputGenEntry);
        Changed = true;
      }
    }
  } else {
    for (std::string &Name : EntryFunctionNames) {
      Function *F = M.getFunction(Name);
      if (!F->isDeclaration()) {
        F->addFnAttr(Attribute::InputGenEntry);
        Changed = true;
      }
    }
  }
  return Changed;
}

} // namespace

PreservedAnalyses
InputGenInstrumentEntriesPass::run(Module &M, AnalysisManager<Module> &MAM) {
  IGIMode Mode = ClInstrumentationMode;
  InputGenEntriesImpl Impl(M, Mode);

  bool Changed = false;

  Changed |= tagEntries(M);
  Changed |= Impl.instrument();

  if (!Changed)
    return PreservedAnalyses::all();

#ifndef NDEBUG
  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));
#endif

  return PreservedAnalyses::none();
}

PreservedAnalyses
InputGenInstrumentMemoryPass::run(Module &M, AnalysisManager<Module> &MAM) {
  IGIMode Mode = ClInstrumentationMode;
  InputGenMemoryImpl Impl(M, MAM, Mode);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

#ifndef NDEBUG
  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));
#endif

  return PreservedAnalyses::none();
}
