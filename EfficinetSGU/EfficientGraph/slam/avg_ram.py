# avg_ram.py
import argparse, shlex, subprocess, time, psutil, os, sys

p = argparse.ArgumentParser()
p.add_argument("--cmd", required=True, help='예: "python test_bit.py"')
p.add_argument("--interval", type=float, default=0.5, help="샘플링 간격(초)")
p.add_argument("--sum-children", action="store_true", help="자식 프로세스 RSS 합산")
args = p.parse_args()

def rss_gb(proc, sum_children):
    try:
        r = proc.memory_info().rss
        if sum_children:
            for c in proc.children(recursive=True):
                try: r += c.memory_info().rss
                except: pass
        return r / (1024**3)
    except: 
        return 0.0

start = time.time()
proc = subprocess.Popen(shlex.split(args.cmd), stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setsid)
ps = psutil.Process(proc.pid)

peak = 0.0
area = 0.0
t_prev = time.time()

while True:
    ret = proc.poll()
    now = time.time()
    ram = rss_gb(ps, args.sum_children)
    dt = now - t_prev
    area += ram * max(0.0, dt)   # 시간가중 적분
    peak = max(peak, ram)
    t_prev = now
    if ret is not None:
        break
    time.sleep(args.interval)

elapsed = time.time() - start
avg = area / max(elapsed, 1e-9)
print(f"[RESULT] elapsed={elapsed:.2f}s ({elapsed/60:.2f} min) | avg_ram={avg:.3f} GB | peak_ram={peak:.3f} GB")
