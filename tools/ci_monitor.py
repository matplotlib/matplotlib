import os
import signal
import sys
import time

import psutil

LOG = os.environ.get("CI_MONITOR_LOG", "/tmp/ci_monitor.tsv")
TOP_N = 20
INTERVAL = 2.0

_stop = False


def _on_term(signum, frame):
    global _stop
    _stop = True


def _clean(text):
    return text.replace("\t", " ").replace("\n", " ")


def _cmdline(proc):
    try:
        return " ".join(proc.cmdline())[:80]
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def snapshot():
    rows = []
    for proc in psutil.process_iter():
        try:
            cpu = proc.cpu_percent(None)
            name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        rows.append((cpu, proc.pid, name, _cmdline(proc)))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows[:TOP_N]


def mark(label):
    with open(LOG, "a") as f:
        f.write(f"M\t{time.time():.3f}\t{_clean(label)}\n")


def run():
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    psutil.cpu_percent(None)
    snapshot()
    while not _stop:
        time.sleep(INTERVAL)
        now = time.time()
        rows = snapshot()
        total = psutil.cpu_percent(None)
        with open(LOG, "a") as f:
            for cpu, pid, name, cmd in rows:
                fields = ["S", f"{now:.3f}", str(pid), f"{cpu:.1f}"]
                fields += [_clean(name), _clean(cmd)]
                f.write("\t".join(fields) + "\n")
            f.write(f"T\t{now:.3f}\t{total:.1f}\n")


def main(argv):
    if len(argv) >= 2 and argv[1] == "run":
        run()
    elif len(argv) >= 3 and argv[1] == "mark":
        mark(argv[2])
    else:
        sys.exit("usage: ci_monitor.py run | mark LABEL")


if __name__ == "__main__":
    main(sys.argv)
