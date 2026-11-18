"""JAILER Streamlit demo:  streamlit run app.py"""
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import jailer_core as jc

st.set_page_config(page_title="JAILER", page_icon="🚨", layout="wide")
st.title("JAILER — a vision firewall for robots")
st.caption("Robots that read the world with vision-language models can be hijacked by a sticky note. "
           "JAILER sits between the robot's camera and its planner, finds text that tries to command the "
           "robot, pixelates it, and only then lets the planner see the frame.")

DEMO = Path("demo")
if not (DEMO / "kitchen_attack.png").exists():
    jc.make_demo()

# ---------------- sidebar
with st.sidebar:
    st.header("Camera input")
    source = st.radio("Source", ["Demo scene", "Upload", "Webcam"], horizontal=True)
    img, default_task = None, "Put the cup on the plate"
    if source == "Demo scene":
        scene = st.selectbox("Scene", list(jc.DEMO_TASKS))
        img = cv2.imread(str(DEMO / scene))
        default_task = jc.DEMO_TASKS[scene]
    else:
        f = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"]) if source == "Upload" \
            else st.camera_input("Take a photo")
        if f is not None:
            img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    task = st.text_input("Robot task", value=default_task, key=f"task_{source}_{default_task}")
    simulate = st.toggle("Simulate jailer failure", help="Forces the fail-closed path")
    st.caption(f"Model: `{jc.MODEL}` · jailer timeout {jc.JAILER_TIMEOUT:.0f}s")

run = st.button("▶ Run", type="primary", width="stretch", disabled=img is None)

if img is None:
    st.info("Pick a demo scene, upload an image or take a webcam photo.")
    st.stop()

cache = st.session_state.setdefault("cache", {})
key = (jc.image_hash(img), task, simulate)

if run and key not in cache:
    with st.status("Running pipeline…", expanded=True) as status:
        st.write("Naive robot planning…")
        try:
            naive = jc.plan(img, task)
        except Exception as e:
            naive = f"(naive planner error: {e})"
        st.write("Jailer inspecting…")
        st.write("Guarded robot planning…")
        res = jc.process(img, task, simulate_failure=simulate, run_naive=False)
        res["report"]["naive_plan"] = naive
        cache[key] = res
        status.update(label="Fail-closed: scene unverified" if res["report"]["fail_closed"] else "Done",
                      state="error" if res["report"]["fail_closed"] else "complete", expanded=False)

res = cache.get(key)
left, right = st.columns(2)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

with left:
    st.subheader("Unprotected robot")
    st.image(rgb(img), width="stretch")
    if res:
        st.error(res["report"]["naive_plan"])
with right:
    st.subheader("Robot behind JAILER")
    if res:
        st.image(rgb(res["jailed"]), width="stretch")
        (st.warning if res["report"]["fail_closed"] else st.success)(res["report"]["guarded_plan"])
    else:
        st.markdown("_Press **Run** to see what the planner sees after JAILER._")

if not res:
    st.stop()

r = res["report"]
st.divider()
badge = {"none": "🟢 NONE", "low": "🟠 LOW", "high": "🔴 HIGH"}.get(r["jailer"]["threat_level"], "?")
st.markdown(f"### Threat level: {badge}" + ("  ·  **FAIL-CLOSED**" if r["fail_closed"] else ""))
st.write(r["jailer"]["summary"])

c = r["counts"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Texts found", c["found"])
m2.metric("Jailed", c["jailed"])
m3.metric("Allowed", c["allowed"])
m4.metric("Latency (s)", r["latency_s"])

rows = [{"text": i["text"], "verdict": i["verdict"], "reason": i["reason"]} for i in r["jailer"]["items"]]
if rows:
    import pandas as pd  # ships with streamlit
    df = pd.DataFrame(rows)
    st.dataframe(df.style.apply(lambda row: ["background-color: rgba(255,60,60,.25)" if row["verdict"] == "injection"
                                             else "" for _ in row], axis=1),
                 width="stretch", hide_index=True)

with st.expander("Raw JSON report"):
    st.json(r)
st.download_button("Download report JSON", json.dumps(r, indent=2), file_name=f"jailer_{r['image_hash']}.json",
                   mime="application/json")
st.caption(f"Side-by-side saved to `{r['saved']}`")
