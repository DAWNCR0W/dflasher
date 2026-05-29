from __future__ import annotations

import argparse
import json
import os
import sys

from transformers import AutoConfig

from dflasher.model_profile import target_layer_ids_for_policy


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch vLLM configured for Speculators hidden-state extraction.",
        usage=(
            "python -m dflasher.hidden_server MODEL [--hidden-states-path PATH] "
            "[--target-layer-ids IDS ...] [--trust-remote-code] -- *VLLM_ARGS"
        ),
    )
    parser.add_argument("model", type=str)
    parser.add_argument("--hidden-states-path", type=str, default="/tmp/hidden_states")
    parser.add_argument("--target-layer-ids", type=int, nargs="+")
    parser.add_argument("--include-last-layer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_known_args()


def build_vllm_command(args: argparse.Namespace, vllm_args: list[str]) -> list[str]:
    vllm_args = [arg for arg in vllm_args if arg != "--"]
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(config, "text_config"):
        config = config.text_config

    if getattr(config, "num_hidden_layers", None) is None:
        raise ValueError("Could not infer num_hidden_layers from the model config.")
    num_hidden_layers = int(config.num_hidden_layers)
    if args.target_layer_ids:
        target_layer_ids = list(args.target_layer_ids)
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
    else:
        target_layer_ids = list(
            target_layer_ids_for_policy(num_hidden_layers, "unknown", "speculators")
        )
        if num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
    invalid_layers = [
        layer_id for layer_id in target_layer_ids if layer_id < 0 or layer_id > num_hidden_layers
    ]
    if invalid_layers:
        raise ValueError(
            "target_layer_ids must be within the hidden-state range "
            f"[0, {num_hidden_layers}]: {invalid_layers}"
        )

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    kv_transfer_config = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": args.hidden_states_path},
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--speculative_config",
        json.dumps(speculative_config),
        "--kv_transfer_config",
        json.dumps(kv_transfer_config),
        *vllm_args,
    ]
    if args.trust_remote_code and "--trust-remote-code" not in command:
        command.append("--trust-remote-code")
    return command


def main() -> None:
    args, vllm_args = parse_args()
    command = build_vllm_command(args, vllm_args)
    print("Running command:")
    print(" ".join(command))
    if args.dry_run:
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
