"""
Validating RL graders before training: a per-task validation pipeline plus the statistics behind it.

Seven toy tasks, each with a deliberately planted defect (except T1):
  T1 good            strong hidden tests, deterministic
  T2 vacuous         F2P tests already pass on the unfixed code (grader carries no signal)
  T3 broken gold     test expectation is wrong, so the reference solution fails (task unsolvable)
  T4 flaky           one test fails ~8% of the time for reasons unrelated to the solution
  T5 order-dependent a test relies on state left behind by an earlier test
  T6 env-dependent   a test's expected value depends on the machine's time zone
  T7 weak tests      a plausible wrong solution (mutant) passes every test

Checks per task:
  gold      : reference solution passes all tests, K times in a row
  original  : every F2P test fails on the unfixed code; every P2P test passes on it
  trivial   : trivially wrong patches (return constant) fail
  order     : gold passes with tests reversed and with each test in its own process
  env       : gold passes under different time zones and hash seeds
  mutants   : each plausible wrong solution fails at least one test (mutation score)
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

K = 30                      # gold reruns for flakiness

RUNNER = """
import importlib, json, sys, time, os
if hasattr(time, "tzset"): time.tzset()
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
mod = importlib.import_module("test_hidden")
res = {}
for name in sys.argv[3].split(","):
    try:
        getattr(mod, name)(); res[name] = True
    except Exception:
        res[name] = False
print(json.dumps(res))
"""

# ------------------------------------------------------------------------ task definitions
BUGGY_MEDIAN = """
def mean(xs): return sum(xs) / len(xs)
def median(xs):
    s = sorted(xs); return s[len(s) // 2]
"""
GOLD_MEDIAN = """
def mean(xs): return sum(xs) / len(xs)
def median(xs):
    s = sorted(xs); n, m = len(s), len(s) // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2
"""
MEDIAN_MUTANTS = {
    "always averages": """
def mean(xs): return sum(xs) / len(xs)
def median(xs):
    s = sorted(xs); m = len(s) // 2
    return (s[m - 1] + s[m]) / 2 if len(s) > 1 else s[0]
""",
    "upper middle minus 0.5": """
def mean(xs): return sum(xs) / len(xs)
def median(xs):
    s = sorted(xs); n, m = len(s), len(s) // 2
    return s[m] if n % 2 else s[m] - 0.5
""",
    "forgets to sort": """
def mean(xs): return sum(xs) / len(xs)
def median(xs):
    s = list(xs); n, m = len(s), len(s) // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2
""",
}
TRIVIAL_MEDIAN = {"return 0": "def mean(xs): return sum(xs) / len(xs)\ndef median(xs): return 0\n"}

STRONG_TESTS = """
import random, statistics
from mathlib import median, mean
def test_f2p_even(): assert median([4, 1, 3, 2]) == 2.5
def test_f2p_even_negative(): assert median([-5, 10, 0, -1]) == -0.5
def test_f2p_property():
    rng = random.Random(1234)
    for _ in range(200):
        xs = [rng.uniform(-100, 100) for _ in range(rng.randint(1, 30))]
        assert abs(median(xs) - statistics.median(xs)) < 1e-9
def test_p2p_odd(): assert median([9, 7, 8]) == 8
def test_p2p_mean(): assert mean([1, 2, 3, 4]) == 2.5
"""
STRONG = dict(f2p=["test_f2p_even", "test_f2p_even_negative", "test_f2p_property"],
              p2p=["test_p2p_odd", "test_p2p_mean"])

TASKS = {
    "T1 good": dict(module="mathlib", original=BUGGY_MEDIAN, gold=GOLD_MEDIAN, tests=STRONG_TESTS,
                    mutants=MEDIAN_MUTANTS, trivial=TRIVIAL_MEDIAN, **STRONG),

    "T2 vacuous": dict(module="mathlib", original=BUGGY_MEDIAN, gold=GOLD_MEDIAN, tests="""
from mathlib import median, mean
def test_f2p_odd_only(): assert median([3, 1, 2]) == 2
def test_p2p_mean(): assert mean([1, 2, 3, 4]) == 2.5
""", f2p=["test_f2p_odd_only"], p2p=["test_p2p_mean"], mutants=MEDIAN_MUTANTS, trivial=TRIVIAL_MEDIAN),

    "T3 broken gold": dict(module="mathlib", original=BUGGY_MEDIAN, gold=GOLD_MEDIAN, tests="""
from mathlib import median, mean
def test_f2p_even(): assert median([1, 2, 3, 4]) == 2      # wrong expectation
def test_p2p_odd(): assert median([9, 7, 8]) == 8
""", f2p=["test_f2p_even"], p2p=["test_p2p_odd"], mutants={}, trivial=TRIVIAL_MEDIAN),

    "T4 flaky": dict(module="mathlib", original=BUGGY_MEDIAN, gold=GOLD_MEDIAN,
                     tests=STRONG_TESTS + """
def test_f2p_service_roundtrip():
    import random as _r
    if _r.random() < 0.08:                                  # simulated timeout / race
        raise AssertionError("upstream timed out")
    assert median([2, 4]) == 3
""", f2p=STRONG["f2p"] + ["test_f2p_service_roundtrip"], p2p=STRONG["p2p"],
                     mutants=MEDIAN_MUTANTS, trivial=TRIVIAL_MEDIAN),

    "T5 order-dependent": dict(module="mathlib", original=BUGGY_MEDIAN, gold=GOLD_MEDIAN, tests="""
from mathlib import median
_CACHE = {}
def test_f2p_even():
    _CACHE["data"] = [4, 1, 3, 2]
    assert median(_CACHE["data"]) == 2.5
def test_p2p_cached_data_sorted():
    assert sorted(_CACHE["data"]) == [1, 2, 3, 4]          # relies on the previous test
def test_p2p_odd(): assert median([9, 7, 8]) == 8
""", f2p=["test_f2p_even"], p2p=["test_p2p_cached_data_sorted", "test_p2p_odd"],
                               mutants={}, trivial=TRIVIAL_MEDIAN),

    "T6 env-dependent": dict(module="dates", original="""
import time
def iso_day(ts): return time.strftime("%d-%m-%Y", time.gmtime(ts))       # wrong format
""", gold="""
import time
def iso_day(ts): return time.strftime("%Y-%m-%d", time.gmtime(ts))
""", tests="""
import time
from dates import iso_day
TS = 1704070800                                            # 2024-01-01 01:00 UTC
def test_f2p_iso_utc():
    assert iso_day(TS) == time.strftime("%Y-%m-%d", time.localtime(TS))   # bug: local time
def test_p2p_type(): assert isinstance(iso_day(TS), str)
""", f2p=["test_f2p_iso_utc"], p2p=["test_p2p_type"], mutants={},
                             trivial={"return constant": "def iso_day(ts): return '1970-01-01'\n"}),

    "T7 weak tests": dict(module="mathlib", original=BUGGY_MEDIAN, gold=GOLD_MEDIAN, tests="""
from mathlib import median, mean
def test_f2p_even(): assert median([1, 2, 3, 4]) == 2.5
def test_p2p_odd(): assert median([9, 7, 8]) == 8
""", f2p=["test_f2p_even"], p2p=["test_p2p_odd"], mutants=MEDIAN_MUTANTS, trivial=TRIVIAL_MEDIAN),
}

# ------------------------------------------------------------------------ execution
WORK = Path(tempfile.mkdtemp(prefix="gradeval_"))
(WORK / "runner.py").write_text(RUNNER)


def run(task, source, order, isolate=False, env=None):
    d = Path(tempfile.mkdtemp(dir=WORK))
    (d / "repo").mkdir(); (d / "tests").mkdir()
    (d / "repo" / f"{task['module']}.py").write_text(textwrap.dedent(source))
    (d / "tests" / "test_hidden.py").write_text(textwrap.dedent(task["tests"]))
    e = {"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1", "TZ": "UTC"}
    e.update(env or {})
    groups = [[n] for n in order] if isolate else [order]
    res = {}
    for g in groups:
        out = subprocess.run([sys.executable, str(WORK / "runner.py"), str(d / "repo"), str(d / "tests"),
                              ",".join(g)], capture_output=True, text=True, env=e, timeout=30)
        try:
            res.update(json.loads(out.stdout.strip().splitlines()[-1]))
        except Exception:
            res.update({n: False for n in g})
    return res


def all_pass(res, names):
    return all(res.get(n, False) for n in names)


def validate(name, t):
    tests = t["f2p"] + t["p2p"]
    problems, info = [], {}

    gold = [all_pass(run(t, t["gold"], tests), tests) for _ in range(K)]
    info["gold"] = f"{sum(gold)}/{K}"
    if not all(gold):
        problems.append("gold fails" if sum(gold) == 0 else f"flaky (gold passed {sum(gold)}/{K})")

    orig = run(t, t["original"], tests)
    bad_f2p = [n for n in t["f2p"] if orig.get(n)]
    bad_p2p = [n for n in t["p2p"] if not orig.get(n)]
    info["original"] = "ok" if not (bad_f2p or bad_p2p) else "bad"
    if bad_f2p:
        problems.append(f"F2P already passes on original: {bad_f2p}")
    if bad_p2p:
        problems.append(f"P2P fails on original: {bad_p2p}")

    triv_pass = [k for k, src in t["trivial"].items() if all_pass(run(t, src, tests), tests)]
    info["trivial"] = "ok" if not triv_pass else "passes!"
    if triv_pass:
        problems.append(f"trivial patch passes: {triv_pass}")

    if all(gold):
        rev = all_pass(run(t, t["gold"], tests[::-1]), tests)
        iso = all_pass(run(t, t["gold"], tests, isolate=True), tests)
        info["order"] = "ok" if rev and iso else "dependent"
        if not (rev and iso):
            problems.append("order-dependent (" + ", ".join(x for x, ok in [("reversed", rev), ("isolated", iso)] if not ok) + ")")
        envs = [{"TZ": "America/Los_Angeles"}, {"TZ": "Pacific/Kiritimati"}, {"PYTHONHASHSEED": "12345"}]
        env_fail = [e for e in envs if not all_pass(run(t, t["gold"], tests, env=e), tests)]
        info["env"] = "ok" if not env_fail else "dependent"
        if env_fail:
            problems.append(f"environment-dependent: {env_fail}")
    else:
        info["order"] = info["env"] = "—"

    if t["mutants"]:
        killed = {m: not all_pass(run(t, src, tests), tests) for m, src in t["mutants"].items()}
        info["mutants"] = f"{sum(killed.values())}/{len(killed)}"
        survivors = [m for m, k in killed.items() if not k]
        if survivors:
            problems.append(f"weak tests: mutants survive {survivors}")
    else:
        info["mutants"] = "—"
    return info, problems


print(f"Per-task validation (gold reruns K = {K})\n")
print(f"   {'task':<20} {'gold':>6} {'original':>9} {'trivial':>8} {'order':>10} {'env':>10} {'mutants':>8}   verdict")
verdicts = {}
for name, t in TASKS.items():
    info, problems = validate(name, t)
    verdicts[name] = problems
    v = "ACCEPT" if not problems else "REJECT: " + "; ".join(problems)
    print(f"   {name:<20} {info['gold']:>6} {info['original']:>9} {info['trivial']:>8} "
          f"{info['order']:>10} {info['env']:>10} {info['mutants']:>8}   {v}")

# ------------------------------------------------------------------------ statistics
print("\nFlake detection: P(at least one failure in k reruns) = 1 − (1 − p)^k")
ks = [10, 30, 100, 300]
print("   flake rate p " + "".join(f"{'k=' + str(k):>9}" for k in ks) + "   k for 95% detection")
for p in [0.01, 0.05, 0.10]:
    need = math.ceil(math.log(0.05) / math.log(1 - p))
    print(f"   {p:>12.2f} " + "".join(f"{1 - (1 - p) ** k:9.2f}" for k in ks) + f"   {need:>19}")
print(f"   All {K} gold runs passing bounds the flake rate at ≈ 3/K = {3 / K:.2f} (95%, rule of three)")


def wilson(x, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = x / n
    c = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (ph + z * z / (2 * n)) / (1 + z * z / n)
    return max(0.0, m - c), min(1.0, m + c)


print("\nGrader confusion matrix over all labelled solutions (should-pass: gold; should-fail: original, trivial, mutants)")
for label, subset in [("all tasks", TASKS), ("accepted tasks only", {k: v for k, v in TASKS.items() if not verdicts[k]})]:
    fn = fp = n_pos = n_neg = 0
    for name, t in subset.items():
        tests = t["f2p"] + t["p2p"]
        n_pos += 1
        fn += not all_pass(run(t, t["gold"], tests), tests)
        for src in [t["original"], *t["trivial"].values(), *t["mutants"].values()]:
            n_neg += 1
            fp += all_pass(run(t, src, tests), tests)
    lo, hi = wilson(fp, n_neg)
    lo2, hi2 = wilson(fn, n_pos)
    print(f"   {label:<20} false positives {fp}/{n_neg} (95% CI {lo:.2f}–{hi:.2f})   "
          f"false negatives {fn}/{n_pos} (95% CI {lo2:.2f}–{hi2:.2f})")
