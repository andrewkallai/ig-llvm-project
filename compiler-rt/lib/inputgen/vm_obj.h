#ifndef VM_OBJ_H
#define VM_OBJ_H

#include <algorithm>
#include <bit>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <list>
#include <map>
#include <random>
#include <string_view>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include "defer.h"
#include "logging.h"
#include "vm_choices.h"
#include "vm_enc.h"
#include "vm_values.h"

namespace __ig {

using UserObjSmallScheme = BucketSchemeTy</*EncodingNo=*/1,
                                          /*OffsetBits=*/12, /*BucketBits=*/3,
                                          /*RealPtrBits=*/32>;
using RTObjScheme = TableSchemeTy<2, 30>;
using UserObjLargeScheme = BigObjSchemeTy</*EncodingNo=*/3, /*ObjectBits=*/10>;

struct ObjectManager {
  ~ObjectManager();

  ObjectManager() : UserObjLarge(*this), UserObjSmall(*this), RTObjs(*this) {

    FVM.BranchConditionMap.reserve(1024);

    auto Min = getIntEnv("INPUTGEN_INT_MIN");
    auto Max = getIntEnv("INPUTGEN_INT_MAX");
    if (Min && Max)
      setIntDistribution(*Min, *Max);
    else
      setIntDistribution(-100, 128);
    setSeed(0);
  }

  ChoiceTrace *CT = nullptr;
  UserObjLargeScheme UserObjLarge;
  UserObjSmallScheme UserObjSmall;
  RTObjScheme RTObjs;

  std::string ProgramName;

  uint32_t Seed;
  std::mt19937 Generator;
  using SeedIntTy = decltype(RTObjs.IntDistribution.Min);
  std::uniform_int_distribution<SeedIntTy> RTObjSeedIncrementDistrib;
  std::uniform_int_distribution<SeedIntTy> RTObjSeedBeginDistrib;

  void setIntDistribution(int32_t Min, int32_t Max) {
    if (Min > Max)
      std::swap(Min, Max);
    RTObjs.IntDistribution.Min = Min;
    RTObjs.IntDistribution.Max = Max;
    RTObjSeedBeginDistrib =
        decltype(RTObjSeedBeginDistrib){std::numeric_limits<SeedIntTy>::min(),
                                        std::numeric_limits<SeedIntTy>::max()};
    RTObjSeedIncrementDistrib =
        decltype(RTObjSeedIncrementDistrib){1, Max - Min - 1};
  }

  void init(ChoiceTrace *CT, std::string_view ProgramName,
            std::function<void(uint32_t)> StopFn) {
    this->CT = CT;
    this->ProgramName = ProgramName;
    ErrorFn = StopFn;
  }

  void setSeed(uint32_t Seed) {
    this->Seed = Seed;
    Generator.seed(Seed);
  }

  RTObjScheme::SeedTy getRTObjSeed() {
    return RTObjScheme::SeedTy(RTObjSeedBeginDistrib(Generator),
                               RTObjSeedIncrementDistrib(Generator));
  }

  void saveInput(uint32_t EntryNo, uint32_t InputIdx, uint32_t ExitCode);
  void reset();

  void *getEntryObj();

  char *encodeUserObj(char *Ptr, uint32_t Size) {
    if (Size < (1 << 10))
      return UserObjSmall.encode(Ptr, Size);
    return UserObjLarge.encode(Ptr, Size);
  }

  char *decode(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.decode(VPtr);
    case 2:
      return RTObjs.decode(VPtr);
    case 3:
      return UserObjLarge.decode(VPtr);
    default:
      // TODO should we error here?
      return VPtr;
    }
  }

  __attribute__((always_inline)) char *
  decodeForAccess(char *VPtr, uint32_t AccessSize, uint32_t TypeId,
                  AccessKind AK, char *BasePtrInfo, bool &AnyInitialized,
                  bool &AllInitialized) {
    AnyInitialized = false;
    AllInitialized = true;
    switch ((uintptr_t)BasePtrInfo & 3) {
    case 1:
      return UserObjSmall.access(VPtr, AccessSize, TypeId, AK == WRITE);
    case 2:
      return RTObjs.access(VPtr, AccessSize, TypeId, AK, AnyInitialized,
                           AllInitialized);
    case 3:
      return UserObjLarge.access(VPtr, AccessSize, TypeId, AK == WRITE);
    default:
      std::cerr << "unknown encoding 1 " << getEncoding(VPtr) << "\n";
      error(1003);
      std::terminate();
    }
  }

  int32_t getEncoding(char *VPtr) {
    switch (EncodingSchemeTy::getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.isMagicIntact(VPtr) ? 1 : ~0;
    case 2:
      return RTObjs.isMagicIntact(VPtr) ? 2 : ~0;
    case 3:
      return UserObjLarge.isMagicIntact(VPtr) ? 3 : ~0;
    default:
      return ~0;
    }
  }

  char *addGlobal(char *Addr, char *Name, int32_t Size) {
    return RTObjs.create(Size, Name);
  }

  char *add(int32_t Size, RTObjScheme::SeedTy Seed) {
    return RTObjs.create(Size);
  }

  std::pair<int32_t, int32_t> getPtrInfo(char *VPtr, bool AllowToFail) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.getPtrInfo(VPtr);
    case 2:
      return RTObjs.getPtrInfo(VPtr);
    case 3:
      return UserObjLarge.getPtrInfo(VPtr);
    default:
      if (AllowToFail)
        return {-2, -2};
      std::cerr << "unknown encoding 2 " << getEncoding(VPtr) << "\n";
      error(1004);
      std::terminate();
    }
  }
  char *getBasePtrInfo(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.getBasePtrInfo(VPtr);
    case 2:
      return RTObjs.getBasePtrInfo(VPtr);
    case 3:
      return UserObjLarge.getBasePtrInfo(VPtr);
    default:
      std::cerr << "unknown encoding 3 " << getEncoding(VPtr) << "\n";
      error(1005);
      std::terminate();
    }
  }
  char *getBase(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.getBase(VPtr);
    case 2:
      return RTObjs.getBase(VPtr);
    case 3:
      return UserObjLarge.getBase(VPtr);
    default:
      std::cerr << "unknown encoding 4 " << getEncoding(VPtr) << "\n";
      error(1005);
      std::terminate();
    }
  }
  char *getBaseVPtr(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.getBaseVPtr(VPtr);
    case 2:
      return RTObjs.getBaseVPtr(VPtr);
    case 3:
      return UserObjLarge.getBaseVPtr(VPtr);
    default:
      std::cerr << "unknown encoding 5 " << getEncoding(VPtr) << "\n";
      error(1005);
      std::terminate();
    }
  }

  bool comparePtrs(bool CmpResult, char *LHSPtr, int32_t LHSInfo,
                   uint32_t LHSOffset, char *RHSPtr, int32_t RHSInfo,
                   uint32_t RHSOffset) {
    if (LHSInfo == RHSInfo) {
      // TODO: Learn from the pointer offset about future runs.
      return CmpResult;
    }

    auto TryToMakeObjNull = [&](char *Obj, RTObjScheme::TableEntryTy &TE,
                                uint32_t Offset) {
      if (TE.AnyAccess)
        return CmpResult;
      if (TE.IsNull)
        return !CmpResult;
      //      if (CT->addBooleanChoice()) {
      //        TE.IsNull = true;
      //        return !CmpResult;
      //      }
      return CmpResult;
    };
    auto *LHSTE = LHSInfo >= 0 ? &RTObjs.Table[LHSInfo] : nullptr;
    auto *RHSTE = RHSInfo >= 0 ? &RTObjs.Table[RHSInfo] : nullptr;
    if (LHSPtr == 0 && RHSInfo > 0)
      return TryToMakeObjNull(RHSPtr, *RHSTE, RHSOffset);
    if (RHSPtr == 0 && LHSInfo > 0)
      return TryToMakeObjNull(LHSPtr, *LHSTE, LHSOffset);

    if (LHSInfo < 0 || RHSInfo < 0) {
      std::cerr
          << "comparison of user object and runtime object! C/C++ UB detected! "
             "("
          << LHSInfo << "[" << LHSOffset << "] " << RHSInfo << "[" << RHSOffset
          << "])\n";
      error(1006);
      std::terminate();
    }

    // Merge objects or
    return CmpResult;
  }

  uint64_t ptrToInt(char *VPtr, uint64_t Value) {
    auto [PtrInfo, PtrOffset] = getPtrInfo(VPtr, /*AllowToFail=*/true);
    if (PtrInfo >= 0) {
      auto &TE = RTObjs.Table[PtrInfo];
      if (TE.IsNull)
        return 0;
      if (TE.AnyAccess)
        return Value;
      //      if (CT->addBooleanChoice()) {
      //        TE.IsNull = true;
      //        return 0;
      //      }
    }
    return Value;
  }

  bool checkRange(char *VPtr, uint32_t Size) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.checkSize(VPtr, Size);
    case 2: {
      bool AnyInitialized = false, AllInitialized = true;
      RTObjs.access(VPtr, Size, 0, CHECK_INITIALIZED, AnyInitialized,
                    AllInitialized);
      return AllInitialized;
    }
    case 3:
      return UserObjLarge.checkSize(VPtr, Size);
    default:
      return true;
    }
  }

  char *decodeAndCheckInitialized(char *VPtr, uint32_t Size,
                                  bool &AllInitialized) {
    switch (getEncoding(VPtr)) {
    case 1:
      AllInitialized = true;
      return UserObjSmall.decode(VPtr);
    case 2: {
      AllInitialized = true;
      bool AnyInitialized = false;
      return RTObjs.access(VPtr, Size, 0, CHECK_INITIALIZED, AnyInitialized,
                           AllInitialized);
    }
    case 3:
      AllInitialized = true;
      return UserObjLarge.decode(VPtr);
    default:
      AllInitialized = true;
      return VPtr;
    }
  }

  bool getDesiredOutcome(uint32_t ChoiceNo) {
    return CT->addBooleanChoice(ChoiceNo);
  }

  FreeValueManager FVM;

  void checkBranchConditions(char *VP, char *VPBP, char *VCP = nullptr,
                             char *VCPBP = nullptr) {
    if (((uint64_t)VPBP & 3) == 2 && ((uint64_t)VCPBP & 3) == 2)
      FVM.checkBranchConditions(VP, VPBP, VCP, VCPBP);
    else if (((uint64_t)VPBP & 3) == 2)
      FVM.checkBranchConditions(VP, VPBP, nullptr, nullptr);
    else if (((uint64_t)VCPBP & 3) == 2)
      FVM.checkBranchConditions(VCP, VCPBP, nullptr, nullptr);
  }
  void addBranchCondition(char *VPtr, BranchConditionInfo *BCI) {
    FVM.BranchConditions[VPtr].insert(BCI);
  }
  BranchConditionInfo *getOrCreateBranchCondition(uint32_t No) {
    if (FVM.BranchConditionMap.size() <= No)
      FVM.BranchConditionMap.resize(std::bit_ceil(No + 1));
    auto *&BCI = FVM.BranchConditionMap[No];
    if (!BCI)
      BCI = new BranchConditionInfo;
    return BCI;
  }
};

} // namespace __ig
#endif
