"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"


def _get_client():
    from mlflow.tracking import MlflowClient
    from src.config import get_settings
    settings = get_settings()
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

def _get_target_version(client, name, config_id):
    filter_string = f"name = '{name}' AND tags.config_id = '{config_id}'"
    versions = client.search_model_versions(filter_string)
    if not versions:
        print(f"error: no version found with config_id={config_id}", file=sys.stderr)
        sys.exit(1)
    elif len(versions) > 1:
        versions.sort(key=lambda x: int(x.version), reverse=True)
        latest_version = versions[0]
        v_nums = sorted([int(v.version) for v in versions])
        print(f"warning: multiple versions match config_id={config_id} (MLflow versions {v_nums}); using latest ({latest_version.version})")
        return latest_version
    return versions[0]


def cmd_set(args: argparse.Namespace) -> None:
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import RestException
    import json
    from datetime import datetime, timezone
    
    client = _get_client()
    name = args.name
    alias = args.alias
    config_id = args.config_id

    target_version = _get_target_version(client, name, config_id)

    current_config_id = ""
    try:
        current_mv = client.get_model_version_by_alias(name, alias)
        current_config_id = current_mv.tags.get("config_id", "")
    except RestException:
        pass

    client.set_registered_model_alias(name, alias, target_version.version)

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": alias,
        "from": current_config_id,
        "to": config_id,
        "op": "set"
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        
    if current_config_id:
        print(f"{alias}: {current_config_id} → {config_id}")
    else:
        print(f"{alias}: (unset) → {config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    from mlflow.exceptions import RestException
    
    client = _get_client()
    name = args.name
    alias = args.alias
    
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except RestException:
        print(f"error: alias {alias} not found", file=sys.stderr)
        sys.exit(1)
        
    run = client.get_run(mv.run_id)
    metrics = run.data.metrics
    
    print(f"{name} @ {alias}")
    print(f"  config_id: {mv.tags.get('config_id', '')}")
    print(f"  model: {mv.tags.get('model', '')}")
    if "accuracy_overall" in metrics:
        print(f"  accuracy_overall: {metrics['accuracy_overall']}")
    if "verdict_rate_leaked" in metrics:
        print(f"  verdict_rate_leaked: {metrics['verdict_rate_leaked']}")
    if "total_cost_usd" in metrics:
        print(f"  total_cost_usd: ${metrics['total_cost_usd']}")


def cmd_list(args: argparse.Namespace) -> None:
    from mlflow.exceptions import RestException
    
    client = _get_client()
    name = args.name
    try:
        rm = client.get_registered_model(name)
    except RestException:
        print("no aliases set")
        return
        
    if not rm.aliases:
        print("no aliases set")
        return
        
    for alias, version in rm.aliases.items():
        mv = client.get_model_version(name, version)
        config_id = mv.tags.get('config_id', '')
        print(f"{alias} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    from mlflow.exceptions import RestException
    import json
    from datetime import datetime, timezone
    
    client = _get_client()
    name = args.name
    alias = args.alias
    
    try:
        current_mv = client.get_model_version_by_alias(name, alias)
        current_config_id = current_mv.tags.get("config_id", "")
    except RestException:
        print("nothing to roll back")
        sys.exit(0)
        
    if not LOG_FILE.exists():
        print(f"no promotion history for alias {alias}")
        sys.exit(1)
        
    last_event = None
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev["alias"] == alias:
                last_event = ev
                
    if not last_event:
        print(f"no promotion history for alias {alias}")
        sys.exit(1)
        
    if last_event["op"] == "rollback":
        print(f"error: {alias} was just rolled back; no further history to walk back to", file=sys.stderr)
        sys.exit(1)
        
    if last_event["op"] == "set" and not last_event["from"]:
        print(f"{alias} has no previous target (first promotion ever)", file=sys.stderr)
        sys.exit(1)
        
    target_config_id = last_event["from"]
    target_version = _get_target_version(client, name, target_config_id)
        
    client.set_registered_model_alias(name, alias, target_version.version)
    
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": alias,
        "from": current_config_id,
        "to": target_config_id,
        "op": "rollback"
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        
    print(f"{alias}: {current_config_id} → {target_config_id} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
