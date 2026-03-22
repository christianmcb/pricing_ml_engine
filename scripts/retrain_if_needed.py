import argparse
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import pandas as pd

from src.live_ops_utils import load_json, load_json_or_default, save_json, utc_now_iso


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_state(path: Path) -> dict:
    return load_json_or_default(path, {"history": []})


def save_state(path: Path, state: dict) -> None:
    save_json(path, state)



def parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def bool_series_tail_all(df: pd.DataFrame, column: str, n: int) -> bool:
    if column not in df.columns:
        return False
    if len(df) < n:
        return False
    tail = df[column].fillna(False).astype(bool).tail(n)
    return bool(tail.all())


def evaluate_rules(
    summary: dict,
    batch_metrics: pd.DataFrame,
    *,
    psi_threshold: float,
    min_consecutive_drift: int,
    min_consecutive_pred_shift: int,
    roc_auc_min: float,
    log_loss_max: float,
    brier_max: float,
) -> list[dict]:
    """Evaluates monitoring data against retrain trigger rules and returns a list of triggered rules."""
    latest = summary.get("latest_batch_summary", {}) or {}
    triggers = []
    
    max_psi = latest.get("max_feature_psi")
    if max_psi is not None and float(max_psi) >= psi_threshold:
        triggers.append(
            {
                "rule": "latest_max_psi",
                "triggered": True,
                "value": float(max_psi),
                "threshold": float(psi_threshold),
            }
        )
    
    if bool_series_tail_all(batch_metrics, "drift_alert", min_consecutive_drift):
        triggers.append(
            {
                "rule": "consecutive_drift_alerts",
                "triggered": True,
                "window": int(min_consecutive_drift),
            }
        )
    
    if bool_series_tail_all(
        batch_metrics, "prediction_shift_alert", min_consecutive_pred_shift
    ):
        triggers.append(
            {
                "rule": "consecutive_prediction_shift_alerts",
                "triggered": True,
                "window": int(min_consecutive_pred_shift),
            }
        )

    # Supervised checks only if metric is present and non-null
    roc_auc = latest.get("roc_auc")
    if roc_auc is not None and not pd.isna(roc_auc) and float(roc_auc) < roc_auc_min:
        triggers.append(
            {
                "rule": "roc_auc_below_min",
                "triggered": True,
                "value": float(roc_auc),
                "threshold": float(roc_auc_min),
            }
        )
    
    log_loss_value = latest.get("log_loss")
    if (
        log_loss_value is not None
        and not pd.isna(log_loss_value)
        and float(log_loss_value) > log_loss_max
    ):
        triggers.append(
            {
                "rule": "log_loss_above_max",
                "triggered": True,
                "value": float(log_loss_value),
                "threshold": float(log_loss_max),
            }
        )

    brier = latest.get("brier_score")
    if brier is not None and not pd.isna(brier) and float(brier) > brier_max:
        triggers.append(
            {
                "rule": "brier_above_max",
                "triggered": True,
                "value": float(brier),
                "threshold": float(brier_max),
            }
        )

    return triggers


def in_cooldown(last_retrain_at: str | None, cooldown_hours: float) -> tuple[bool, float | None]:
    """Returns whether the retrain cooldown is still active and hours elapsed since last retrain."""
    last_dt = parse_iso(last_retrain_at)
    if last_dt is None:
        return False, None

    elapsed_hours = (utc_now() - last_dt).total_seconds() / 3600.0
    blocked = elapsed_hours < cooldown_hours
    return blocked, elapsed_hours


def run_training(train_command: str | None) -> subprocess.CompletedProcess:
    if train_command:
        cmd = shlex.split(train_command)
    else:
        cmd = [sys.executable, "scripts/train.py"]

    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decide whether to retrain based on monitoring outputs."
    )
    parser.add_argument("--summary", default="outputs/monitoring/summary.json")
    parser.add_argument("--batch-metrics", default="outputs/monitoring/batch_metrics.csv")
    parser.add_argument("--decision-output", default="outputs/monitoring/retrain_decision.json")
    parser.add_argument("--state-file", default="outputs/retrain_state.json")

    parser.add_argument("--psi-threshold", type=float, default=None)
    parser.add_argument("--min-consecutive-drift", type=int, default=3)
    parser.add_argument("--min-consecutive-pred-shift", type=int, default=3)

    parser.add_argument("--roc-auc-min", type=float, default=0.75)
    parser.add_argument("--log-loss-max", type=float, default=0.65)
    parser.add_argument("--brier-max", type=float, default=0.25)

    parser.add_argument("--cooldown-hours", type=float, default=24.0)
    parser.add_argument("--execute", action="store_true", help="Actually run training if triggered.")
    parser.add_argument(
        "--train-command",
        default=None,
        help="Optional custom train command, e.g. 'python scripts/train.py'",
    )

    return parser.parse_args()



def main() -> None:
    """Evaluates retrain triggers from monitoring outputs and optionally executes training."""
    args = parse_args()

    summary_path = Path(args.summary)
    batch_metrics_path = Path(args.batch_metrics)
    decision_output_path = Path(args.decision_output)
    state_path = Path(args.state_file)

    summary = load_json(summary_path)
    batch_metrics = pd.read_csv(batch_metrics_path)
    state = load_state(state_path)

    psi_threshold = (
        float(args.psi_threshold)
        if args.psi_threshold is not None
        else float(summary.get("psi_threshold", 0.2))
    )

    triggers = evaluate_rules(
        summary=summary,
        batch_metrics=batch_metrics,
        psi_threshold=psi_threshold,
        min_consecutive_drift=args.min_consecutive_drift,
        min_consecutive_pred_shift=args.min_consecutive_pred_shift,
        roc_auc_min=args.roc_auc_min,
        log_loss_max=args.log_loss_max,
        brier_max=args.brier_max,
    )

    last_retrain_at = state.get("last_retrain_at_utc")
    cooldown_blocked, elapsed_hours = in_cooldown(last_retrain_at, args.cooldown_hours)

    should_retrain = len(triggers) > 0 and not cooldown_blocked
    execute_retrain = bool(args.execute and should_retrain)

    decision = {
        "timestamp_utc": utc_now_iso(),
        "summary_path": str(summary_path),
        "batch_metrics_path": str(batch_metrics_path),
        "psi_threshold_used": float(psi_threshold),
        "cooldown_hours": float(args.cooldown_hours),
        "cooldown_blocked": bool(cooldown_blocked),
        "hours_since_last_retrain": elapsed_hours,
        "triggers": triggers,
        "should_retrain": bool(should_retrain),
        "execute_requested": bool(args.execute),
        "executed": False,
        "train_command": args.train_command or f"{sys.executable} scripts/train.py",
        "train_return_code": None,
        "train_stdout_tail": None,
        "train_stderr_tail": None,
    }

    if execute_retrain:
        proc = run_training(args.train_command)
        decision["executed"] = True
        decision["train_return_code"] = proc.returncode
        decision["train_stdout_tail"] = (proc.stdout or "")[-4000:]
        decision["train_stderr_tail"] = (proc.stderr or "")[-4000:]

        if proc.returncode == 0:
            state["last_retrain_at_utc"] = utc_now_iso()
            state["last_retrain_status"] = "success"
        else:
            state["last_retrain_status"] = "failed"

    # Keep a compact history
    history = state.get("history", [])
    history.append(
        {
            "timestamp_utc": decision["timestamp_utc"],
            "should_retrain": decision["should_retrain"],
            "executed": decision["executed"],
            "train_return_code": decision["train_return_code"],
            "n_triggers": len(triggers),
            "cooldown_blocked": decision["cooldown_blocked"],
        }
    )
    state["history"] = history[-200:]

    save_json(decision_output_path, decision)
    save_state(state_path, state)

    print(f"Decision written to: {decision_output_path}")
    print(f"Should retrain: {decision['should_retrain']}")
    print(f"Cooldown blocked: {decision['cooldown_blocked']}")
    print(f"Triggers: {len(triggers)}")
    if triggers:
        print("Triggered rules:")
        for t in triggers:
            print(f" - {t['rule']}")
    if args.execute:
        print(f"Executed training: {decision['executed']}")
        print(f"Train return code: {decision['train_return_code']}")


if __name__ == "__main__":
    main()