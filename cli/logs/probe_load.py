"""Ground-truth probe: does this .aimodel load on this machine's toolchain?

Run as a subprocess — a 0.4.0 asset aborts the process (LLVM ERROR), it does not raise.
"""
import asyncio
import sys
from pathlib import Path

import coreai.runtime as rt


async def main(path: str) -> None:
    m = await rt.AIModel.load(Path(path), rt.SpecializationOptions.cpu_only())
    print("LOAD OK", m.function_names)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
