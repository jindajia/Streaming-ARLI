"""Milestone reporter for the single-actor comparison (called by the training jobs; safe to call by hand).

Works in a dedicated clone (REPORT_REPO) so it never touches anyone's working tree; writers are serialised with an
atomic lock directory on scratch (stale after 20 min).  Regenerates results/single_actor_comparison/ (raw JSONL, CSV,
PNG+SVG figure, STATUS.md), commits only that directory and pushes to origin/main (rebase + retry on rejection).

    python scripts/report_single.py --reason "manual"
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_REPO = os.environ.get("ARLI_REPORT_REPO", "/N/scratch/jindjia/arli/report_repo")
LOCK = "/N/scratch/jindjia/arli/report_single.lock"
REMOTE = "git@github.com:jindajia/Streaming-ARLI.git"
OUT = "results/single_actor_comparison"
SA = "/N/scratch/jindjia/arli/runs_single"
ENV = dict(os.environ, GIT_SSH_COMMAND="ssh -i /N/u/jindjia/BigRed200/.ssh/jindajia-GitHub -o IdentitiesOnly=yes "
           "-o StrictHostKeyChecking=accept-new")
RUN_NAMES = [f"sa_{m}_seed{s}" for s in (0, 1) for m in ("shared_every_step", "final_step_only")]


def sh(*cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}: {r.stderr[-500:]}")
    return r


def lock():
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            with open(f"{LOCK}/owner", "w") as f:
                f.write(f"{os.uname().nodename} {os.getpid()} {os.environ.get('SLURM_JOB_ID')} {time.time()}\n")
            return
        except FileExistsError:
            if time.time() - os.path.getmtime(LOCK) > 1200:
                shutil.rmtree(LOCK, ignore_errors=True)
                continue
            if time.time() - t0 > 600:
                raise TimeoutError("report lock busy")
            time.sleep(5)


def job_table():
    r = subprocess.run(["squeue", "-u", os.environ.get("USER", "jindjia"), "-h", "-o", "%i|%j|%T|%M|%E"],
                       capture_output=True, text=True)
    rows = [l.split("|") for l in r.stdout.strip().splitlines() if l.split("|")[1].startswith("sa_")]
    return rows, r.returncode == 0


def status_md(reason):
    lines = [f"# Single-actor comparison — status", "",
             f"_Auto-generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `scripts/report_single.py` ({reason})._", "",
             "Methods train from RL step 0 with the frozen π0 backbone; ARLI curves are the R2 runs with the same "
             "protocol and prefill std 0.3. Real evaluations only.", "",
             "| run | latest eval step | successes / 50 | prefill env steps | grad steps | online actor calls | wall h |",
             "|---|---|---|---|---|---|---|"]
    for name in RUN_NAMES + ["r2_arli_seed0", "r2_arli_seed1_std030"]:
        base = SA if name.startswith("sa_") else "/N/scratch/jindjia/arli/runs"
        p = f"{base}/{name}/evals.jsonl"
        if not os.path.exists(p):
            lines.append(f"| `{name}` | — | — | — | — | — | — |")
            continue
        ev = sorted({e["milestone"]: e for e in map(json.loads, open(p))}.values(), key=lambda e: e["milestone"])
        e = ev[-1]
        wall = e.get("wall_total_s")
        lines.append(f"| `{name}` | {e['online_env_steps']:,} | {e['n_success']} | {e['prefill_env_steps']:,} | "
                     f"{e['grad_steps']:,} | {e.get('actor_calls_online', '—')} | "
                     f"{'—' if wall is None else f'{wall / 3600:.1f}'} |")
    rows, ok = job_table()
    lines += ["", "## Slurm jobs (at report time)", ""]
    if not ok:
        lines.append("_squeue unavailable from this host._")
    elif not rows:
        lines.append("_No sa_* jobs queued or running._")
    else:
        lines += ["| job | name | state | elapsed | dependency |", "|---|---|---|---|---|"]
        lines += [f"| {i} | {n} | {s} | {m} | {d} |" for i, n, s, m, d in rows]
    lines += ["", "Figure: `success_vs_env_steps.png` (vector: `.svg`); data: `evals.csv`, `raw/*.jsonl`.",
              "Design, objective and limitations: `DECISIONS.md` (D19–D24)."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="manual")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    lock()
    try:
        if not os.path.isdir(f"{REPORT_REPO}/.git"):
            sh("git", "clone", "-q", REMOTE, REPORT_REPO)
        for attempt in range(4):
            sh("git", "fetch", "-q", "origin", cwd=REPORT_REPO)
            sh("git", "checkout", "-q", "-B", "main", "origin/main", cwd=REPORT_REPO)
            sh("git", "reset", "-q", "--hard", "origin/main", cwd=REPORT_REPO)
            out = f"{REPORT_REPO}/{OUT}"
            os.makedirs(out, exist_ok=True)
            sh(sys.executable, f"{HERE}/scripts/plot_single.py", "--out", out,
               "--frozen", f"{HERE}/results/r1/frozen_async_d20.jsonl")
            with open(f"{out}/STATUS.md", "w") as f:
                f.write(status_md(args.reason))
            sh("git", "add", OUT, cwd=REPORT_REPO)
            if not sh("git", "diff", "--cached", "--quiet", cwd=REPORT_REPO, check=False).returncode:
                print("nothing to commit")
                return
            sh("git", "-c", "user.name=Jinda Jia", "-c", "user.email=jindjia@iu.edu", "commit", "-q", "-m",
               f"single-actor comparison report: {args.reason}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
               cwd=REPORT_REPO)
            if args.no_push:
                print("committed locally (no push)")
                return
            if sh("git", "push", "-q", "origin", "HEAD:main", cwd=REPORT_REPO, check=False).returncode == 0:
                print("pushed", sh("git", "rev-parse", "--short", "HEAD", cwd=REPORT_REPO).stdout.strip())
                return
            time.sleep(5 * (attempt + 1))  # rejected (someone else pushed): refetch and regenerate
        raise RuntimeError("push failed after retries")
    finally:
        shutil.rmtree(LOCK, ignore_errors=True)


if __name__ == "__main__":
    main()
