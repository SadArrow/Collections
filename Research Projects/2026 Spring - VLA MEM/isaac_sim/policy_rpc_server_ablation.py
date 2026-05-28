from __future__ import annotations

import importlib
import sys


def main() -> None:
    print("[rpc_ablation] import_policy_prompting_ablation:start", flush=True)
    sys.modules["policy_prompting"] = importlib.import_module("policy_prompting_ablation")
    print("[rpc_ablation] import_policy_prompting_ablation:done", flush=True)
    print("[rpc_ablation] import_base_policy_rpc_server:start", flush=True)
    base = importlib.import_module("policy_rpc_server")
    print("[rpc_ablation] import_base_policy_rpc_server:done", flush=True)
    base.main()


if __name__ == "__main__":
    main()
