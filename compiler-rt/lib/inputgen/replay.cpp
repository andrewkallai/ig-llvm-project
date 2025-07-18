#include "common.h"
#include "global_manager.h"
#include "logging.h"
#include "timer.h"
#include "vm_storage.h"

#include <cstdint>
#include <cstdio>
#include <exception>

namespace __ig {
bool GMInit = false;
DeferGlobalConstruction<GlobalManager, GMInit> GM;

void error(uint32_t ErrorCode) {
  std::cerr << "This should never happen\n";
  std::terminate();
}

} // namespace __ig

using namespace __ig;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <file.inp> [<entry_no>]\n";
    printNumAvailableFunctions();
    printAvailableFunctions();
    exit(static_cast<int>(ExitStatus::WrongUsage));
  }

  uint32_t EntryNo = 0;
  if (argc > 2)
    EntryNo = std::atoi(argv[2]);
  if (EntryNo >= __ig_num_entry_points) {
    fprintf(stderr, "Entry %u is out of bounds, %u available\n", EntryNo,
            __ig_num_entry_points);
    exit(static_cast<int>(ExitStatus::EntryNoOutOfBounds));
  }

  void *P;
  storage::StorageManager SM;
  {
    Timer T("init");
    std::ifstream IFS(argv[1], std::ios_base::in | std::ios_base::binary);
    const int BufferSize = 65536; // Example: 64KB
    char *Buffer = new char[BufferSize];
    IFS.rdbuf()->pubsetbuf(Buffer, BufferSize);
    IFS.tie(nullptr);

    GM->sort();
    INPUTGEN_DEBUG({
      assert(GM.isConstructed());
      std::cerr << "Globals in replay module\n";
      for (auto G : GM->Globals)
        std::cerr << G.Name << "\n";
    });
    SM.read(IFS, *GM);
    INPUTGEN_DEBUG({
      std::cerr << "Globals in input\n";
      for (auto G : SM.Globals)
        std::cerr << G.Name << "\n";
    });

    P = SM.getEntryPtr();
  }
  {
    Timer T("replay");
    __ig_entry(EntryNo, P);
  }
  exit(static_cast<int>(ExitStatus::Success));
}
