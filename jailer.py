"""JAILER CLI.

  python jailer.py --make-demo
  python jailer.py demo/kitchen_attack.png --task "Put the cup on the plate"
  python jailer.py demo/kitchen_attack.png --simulate-failure
"""
import argparse
import json
import sys

import cv2

import jailer_core as jc


def main():
    ap = argparse.ArgumentParser(description="JAILER — a vision firewall for robots")
    ap.add_argument("image", nargs="?")
    ap.add_argument("--task", default="Put the cup on the plate")
    ap.add_argument("--make-demo", action="store_true", help="generate synthetic demo scenes in demo/")
    ap.add_argument("--simulate-failure", action="store_true", help="force the fail-closed path")
    ap.add_argument("--no-naive", action="store_true", help="skip the unprotected planner")
    a = ap.parse_args()

    if a.make_demo:
        for p in jc.make_demo():
            print("wrote", p)
        if not a.image:
            return
    if not a.image:
        ap.error("image required (or --make-demo)")
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    res = jc.process(img, a.task, simulate_failure=a.simulate_failure, run_naive=not a.no_naive)
    r = res["report"]
    print("\n== UNPROTECTED ROBOT ==\n" + r.get("naive_plan", "(skipped)"))
    print("\n== ROBOT BEHIND JAILER ==\n" + r["guarded_plan"])
    print(f"\njailer: {r['jailer']['summary']}  threat={r['jailer']['threat_level']}  "
          f"fail_closed={r['fail_closed']}  counts={r['counts']}  latency={r['latency_s']}s")
    for it in r["jailer"]["items"]:
        print(f"  [{it['verdict']:9}] {it['text']!r} — {it['reason']}")
    print("saved", r["saved"])
    with open(r["saved"].replace(".png", ".json"), "w") as f:
        json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
