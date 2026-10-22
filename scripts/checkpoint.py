"""Keep recoverable source snapshots on Drive before moving to another revision."""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def sync(repo: Path, data: Path, revision: str, check=False):
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", revision) or revision.startswith("-"):
        raise ValueError("Geçersiz sürüm.")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("Yerel değişiklikler var. Önce kaydedin; otomatik güncelleme bunları ezmez.")
    current = git(repo, "rev-parse", "HEAD")
    git(repo, "fetch", "origin", revision)
    target = git(repo, "rev-parse", "FETCH_HEAD")
    print("Mevcut:", current, "\nHedef:", target)
    if check:
        return
    checkpoints = data / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    bundle = checkpoints / f"{current}.bundle"
    if not bundle.exists():
        git(repo, "bundle", "create", str(bundle), "HEAD")
        git(repo, "bundle", "verify", str(bundle))
    git(repo, "checkout", "--detach", target)
    (checkpoints / f"{int(time.time())}.json").write_text(json.dumps({"previous": current, "current": target, "bundle": str(bundle)}, indent=2))
    print("Kaynak güncellendi. Bağımlılık ve başlatma hücrelerini tekrar çalıştırın.")


def rollback(repo: Path, data: Path, revision: str):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Geri dönüş için tam 40 karakterli commit SHA gerekir.")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("Yerel değişiklikler var; geri dönüş durduruldu.")
    bundle = data / "checkpoints" / f"{revision}.bundle"
    git(repo, "bundle", "verify", str(bundle))
    git(repo, "fetch", str(bundle), "HEAD")
    if git(repo, "rev-parse", "FETCH_HEAD") != revision:
        raise RuntimeError("Checkpoint sürümü eşleşmiyor.")
    git(repo, "checkout", "--detach", revision)
    print("Geri dönüldü:", revision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "update", "rollback"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.action == "rollback":
        rollback(args.repo, args.data, args.revision)
    else:
        sync(args.repo, args.data, args.revision, args.action == "check")
