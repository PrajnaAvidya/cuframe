#pragma once

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define CUFRAME_NVTX_PUSH(name) nvtxRangePushA(name)
#define CUFRAME_NVTX_POP() nvtxRangePop()
#else
#define CUFRAME_NVTX_PUSH(name) ((void)0)
#define CUFRAME_NVTX_POP() ((void)0)
#endif
