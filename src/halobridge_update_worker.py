"""Standalone, stdlib-only pipx update transaction run by a separate user unit.

This file is copied outside the pipx environment before starting. Never import
Halobridge here: pipx may replace the environment while this process is alive.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from urllib.parse import urlsplit
import ipaddress

REPO_URL = "https://github.com/Siyarbekir47/Halobridge.git"
TERMINAL_PHASES = {"succeeded", "failed", "rolled_back", "recovery_required"}
VERSION_RE = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}\Z")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# Probe only the bound router, bypassing HTTP proxies and redirects so a
# dashboard auth token cannot escape to a proxy or unrelated service.
urlopen = build_opener(ProxyHandler({}), _NoRedirect()).open


class WorkerError(RuntimeError):
    pass


def valid_ref(value: Any) -> bool:
    return (isinstance(value, str) and bool(REF_RE.fullmatch(value))
            and ".." not in value and "//" not in value
            and not value.endswith(("/", ".", ".lock")))


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, 0o600)
        os.replace(name, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def read_job(path: Path) -> dict | None:
    try:
        if path.stat().st_size > 64 * 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise WorkerError("The update lock must not be a symlink.")
    handle = path.open("a+b")
    try:
        if os.name == "posix":
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        handle.close()
        raise WorkerError("Another app update is already running.") from error
    return handle


def _validate(plan: dict) -> tuple[Path, Path, Path]:
    for field in ("version", "previous_version"):
        if not isinstance(plan.get(field), str) or not VERSION_RE.fullmatch(plan[field]):
            raise WorkerError("Update requires verified stable package versions.")
    if tuple(map(int, plan["version"].split("."))) <= tuple(map(int, plan["previous_version"].split("."))):
        raise WorkerError("Automatic downgrade is disabled.")
    if (not valid_ref(plan.get("source_ref")) or not re.fullmatch(r"[0-9a-f]{40}", str(plan.get("commit", "")))
            or not re.fullmatch(r"halobridge[a-zA-Z0-9_.@-]*\.service", str(plan.get("service", "")))
            or not re.fullmatch(r"[a-zA-Z0-9-]{1,80}", str(plan.get("id", "")))
            or plan.get("drained") is not True or not isinstance(plan.get("previous_pid"), int) or plan["previous_pid"] <= 0):
        raise WorkerError("Invalid update handoff.")
    for command in ("pipx", "systemctl"):
        if not isinstance(plan.get(command), str) or not Path(plan[command]).is_absolute():
            raise WorkerError("Update tool paths must be absolute.")
    prefix, journal, lock = (Path(plan[field]) for field in ("prefix", "journal", "lock"))
    if (not all(path.is_absolute() for path in (prefix, journal, lock)) or prefix.is_symlink()
            or journal.is_symlink() or lock.is_symlink() or journal.parent != lock.parent
            or prefix.name != "halobridge" or not (prefix / "pipx_metadata.json").is_file()
            or journal.parent == prefix or prefix in journal.parents):
        raise WorkerError("Invalid pipx environment or update state directory.")
    data = json.loads((prefix / "pipx_metadata.json").read_text(encoding="utf-8"))
    if data.get("main_package", {}).get("package") != "halobridge":
        raise WorkerError("The target is not the Halobridge pipx environment.")
    job = read_job(journal)
    if not job or job.get("id") != plan["id"] or job.get("phase") in TERMINAL_PHASES or job.get("cancelled"):
        raise WorkerError("The update handoff is no longer current.")
    return prefix, journal, lock


def run_command(args, *, timeout, capture_output=True, text=True, check=False, env=None):
    """Terminate the whole install process group before restoring on timeout."""
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=text, env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.communicate()
        raise
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def execute(plan: dict, *, runner=run_command) -> int:
    """Install, verify and restart; restore the entire old venv before any restart on failure."""
    prefix, journal, lock = _validate(plan)
    lock_handle = acquire_lock(lock)
    try:
        _validate(plan)
    except Exception:
        lock_handle.close()
        raise
    job = read_job(journal) or {}
    backup_dir = journal.parent / "app-update-backups" / plan["id"]
    backup = backup_dir / "venv"
    changed = False
    restarted = False
    app_links = []

    def save(phase: str, message: str, **extra):
        nonlocal job
        current = read_job(journal)
        if not current or current.get("id") != plan["id"]:
            raise WorkerError("The durable update handoff changed.")
        if current.get("cancelled") and phase not in {"failed", "rolling_back", "rolled_back", "recovery_required"}:
            raise WorkerError("The app update handoff was cancelled.")
        job = {**job, **current, "phase": phase, "message": message, "updated_at": time.time(), **extra}
        atomic_json(journal, job)

    def run(args: list[str], timeout=120) -> str:
        env = os.environ.copy()
        if plan.get("pipx_home"):
            env["PIPX_HOME"] = plan["pipx_home"]
        if plan.get("pipx_bin"):
            env["PIPX_BIN_DIR"] = plan["pipx_bin"]
        result = runner(args, capture_output=True, text=True, check=False, timeout=timeout, env=env)
        if result.returncode:
            # Package output can contain configured credentials. Keep it out of
            # the dashboard journal; the systemd unit's journal is available.
            raise WorkerError(f"{Path(args[0]).name} failed (exit {result.returncode}).")
        return result.stdout

    def installed() -> dict:
        script = ("import json,importlib.metadata as m;d=m.distribution('halobridge');"
                  "s=json.loads(d.read_text('direct_url.json') or '{}');"
                  "print(json.dumps({'version':d.version,'commit':s.get('vcs_info',{}).get('commit_id'),'url':s.get('url')}))")
        return json.loads(run([str(prefix / "bin" / "python"), "-I", "-c", script]))

    try:
        save("backing_up", "Backing up the current Halobridge environment.")
        backup_dir.mkdir(parents=True, mode=0o700)
        shutil.copytree(prefix, backup, symlinks=True)
        for raw in plan.get("app_links", []):
            path = Path(raw)
            if not path.is_absolute() or path.name != "halobridge" or path.is_dir():
                raise WorkerError("The exposed pipx command could not be backed up.")
            if path.is_symlink():
                app_links.append((path, "symlink", os.readlink(path)))
            elif path.is_file():
                saved = backup_dir / "halobridge-entry"
                shutil.copy2(path, saved)
                app_links.append((path, "file", saved))
            else:
                raise WorkerError("The exposed pipx command is missing.")
        save("installing", "Installing the verified Halobridge version.", backup_dir=str(backup_dir))
        changed = True
        run([plan["pipx"], "install", "--force", f"git+{REPO_URL}@{plan['source_ref']}"], timeout=1800)
        save("verifying", "Verifying the installed package version and source.")
        info = installed()
        if info.get("version") != plan["version"] or info.get("commit") != plan["commit"] or info.get("url") not in {REPO_URL, REPO_URL.removesuffix(".git")}:
            raise WorkerError("Installed version or source differs from the checked update; the branch may have moved.")
        # Re-read the durable authorization immediately before the disruptive
        # action. The app held the request admission gate until this handoff.
        current = read_job(journal)
        if not current or current.get("id") != plan["id"] or current.get("phase") != "verifying" or current.get("cancelled"):
            raise WorkerError("Update authorization changed before restart.")
        before = dict(line.split("=", 1) for line in run([
            plan["systemctl"], "--user", "show", plan["service"], "--property=ActiveState", "--property=MainPID",
        ]).splitlines() if "=" in line)
        if before.get("ActiveState") != "active" or before.get("MainPID") != str(plan["previous_pid"]):
            raise WorkerError("The Halobridge service changed during installation; restart was cancelled.")
        save("restarting", "Restarting the Halobridge user service.")
        restarted = True
        run([plan["systemctl"], "--user", "restart", plan["service"]], timeout=120)
        properties = dict(line.split("=", 1) for line in run([
            plan["systemctl"], "--user", "show", plan["service"], "--property=ActiveState", "--property=MainPID",
        ]).splitlines() if "=" in line)
        pid = properties.get("MainPID", "0")
        if properties.get("ActiveState") != "active" or not pid.isdigit() or int(pid) <= 0 or int(pid) == plan["previous_pid"]:
            raise WorkerError("The updated Halobridge service could not be verified.")
        if plan.get("health_url"):
            url = urlsplit(plan["health_url"])
            if url.scheme != "http" or url.username or url.password or url.path != "/dashboard/api/app-updates" or url.query or url.fragment:
                raise WorkerError("Invalid local service verification URL.")
            ipaddress.ip_address(url.hostname)
            headers = {"Accept": "application/json"}
            if plan.get("health_token"):
                headers["Authorization"] = "Bearer " + plan["health_token"]
            for attempt in range(15):
                try:
                    with urlopen(Request(plan["health_url"], headers=headers), timeout=2) as response:
                        content = response.read(65537)
                    if len(content) <= 65536 and json.loads(content).get("current_version") == plan["version"]:
                        break
                except (OSError, ValueError):
                    pass
                if attempt == 14:
                    raise WorkerError("The restarted service did not report the target Halobridge version.")
                time.sleep(2)
        save("succeeded", "Halobridge was updated and the user service restarted.")
        return 0
    except Exception as error:
        reason = str(error) if isinstance(error, WorkerError) else "The app update failed. Check the worker journal."
        if restarted:
            save("recovery_required", reason + " Restore the saved environment before restarting again.", backup_dir=str(backup_dir))
        elif changed:
            try:
                save("rolling_back", "Restoring the previous Halobridge environment.", error=reason)
                # Both locations are absolute, validated above and confined to
                # this transaction. No user-supplied filesystem commands run.
                if prefix.exists():
                    shutil.rmtree(prefix)
                shutil.copytree(backup, prefix, symlinks=True)
                for path, kind, saved in app_links:
                    if path.is_dir() and not path.is_symlink():
                        raise WorkerError("The exposed command was replaced with a directory.")
                    path.unlink(missing_ok=True)
                    if kind == "symlink":
                        path.symlink_to(saved)
                    else:
                        shutil.copy2(saved, path)
                if installed().get("version") != plan["previous_version"]:
                    raise WorkerError("Restored package version could not be verified.")
                save("rolled_back", reason + " The previous environment was restored; the service was not restarted.")
            except Exception:
                save("recovery_required", reason + " Automatic restoration failed. Restore the saved environment manually.", backup_dir=str(backup_dir))
        else:
            save("failed", reason)
        return 1
    finally:
        lock_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.plan.is_symlink() or args.plan.stat().st_size > 64 * 1024:
            raise WorkerError("Invalid update plan.")
        return execute(json.loads(args.plan.read_text(encoding="utf-8")))
    except (OSError, ValueError, WorkerError) as error:
        parser.exit(1, f"App update worker: {error}\n")
    finally:
        # The handoff can contain the optional dashboard auth token. Retain
        # only the safe journal and recovery backup after this worker exits.
        args.plan.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
