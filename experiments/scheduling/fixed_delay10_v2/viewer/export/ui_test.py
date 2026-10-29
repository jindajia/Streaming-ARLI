"""Headless-browser test of the built site (Playwright + Chromium; run on a login/CPU node, no GPU).

Serves SITE on 127.0.0.1:<free port>, loads the viewer and checks that playback, pause, seek, decision stepping,
checkpoint switching, lock mode, view toggles and the learning tab update the panels without console errors, and
that the numbers shown agree with the exported data.  Screenshots go to --shots.

    PLAYWRIGHT_BROWSERS_PATH=/N/scratch/jindjia/ms-playwright <venv>/bin/python viewer/export/ui_test.py --site DIR
"""
import argparse
import functools
import http.server
import json
import os
import socket
import sys
import threading

from playwright.sync_api import sync_playwright


def serve(site):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Quiet, directory=site))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    shots = args.shots or os.path.join(args.site, "..", "ui_test_shots")
    os.makedirs(shots, exist_ok=True)
    idx = json.load(open(f"{args.site}/data/index.json"))
    port = serve(args.site)
    errors, results = [], []

    def ok(cond, msg):
        results.append(("PASS" if cond else "FAIL") + " " + msg)
        print(results[-1], flush=True)

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport=dict(width=1500, height=1000), device_scale_factor=1)
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        ck_last = idx["checkpoints"][-1]["id"]
        seed = args.seed or idx["seeds"][0]["seed"]
        pg.goto(f"http://127.0.0.1:{port}/index.html#pol={ck_last}&seed={seed}&t=0")
        pg.wait_for_function("() => typeof curEp === 'function' && curEp() && document.querySelector('#epPlots canvas')", timeout=60000)
        pg.wait_for_timeout(800)
        pg.screenshot(path=f"{shots}/01_loaded.png", full_page=True)
        ok(pg.evaluate("S.pol") == ck_last and pg.evaluate("S.seed") == seed, "initial selection from URL hash")
        # play / pause
        pg.click("#bPlay")
        pg.wait_for_timeout(1200)
        pg.click("#bPlay")
        t1 = pg.evaluate("S.tick")
        ok(t1 > 20, f"playback advanced to tick {t1} in ~1.2 s at 1x (50 Hz)")
        pg.wait_for_timeout(300)
        ok(pg.evaluate("S.tick") == t1 and not pg.evaluate("S.playing"), "pause holds the tick")
        rd = pg.inner_text("#readouts")
        ok(f"{t1} /" in rd, "readouts follow the cursor tick")
        # step + decision stepping
        pg.keyboard.press("ArrowRight")
        ok(pg.evaluate("S.tick") == t1 + 1, "ArrowRight steps one control step")
        pg.click("#bNextDec")
        tdec = pg.evaluate("S.tick")
        ok(pg.evaluate("curEp().decAt[S.tick]") >= 0, f"next-decision lands on a decision tick ({tdec})")
        st = pg.inner_text("#statusBox")
        ok("Scheduler decision" in st, "status box shows a decision at a decision tick")
        # selected action matches the recorded action and the argmax
        chk = pg.evaluate("""() => { const E = curEp(), i = E.decAt[S.tick], v = currentView();
            let s = 0; for (let a = 0; a < 201; a++) s += v.pOn[a];
            return {sel: v.selAct, rec: E.meta.decisions.action[i], arg: v.stOn.argmax, sum: s, pk: v.stOn.pKeep, rep: v.stOn.rep}; }""")
        ok(chk["sel"] == chk["rec"] == chk["arg"], f"selected action == recorded == argmax ({chk['sel']})")
        ok(abs(chk["sum"] - 1) < 1e-6 and abs(chk["pk"] + chk["rep"] - 1) < 1e-6, f"P(KEEP) + sum P(REPLAN) = {chk['pk'] + chk['rep']:.9f}")
        # in-flight tick
        inflight = pg.evaluate("() => { const E = curEp(); for (let t = 0; t <= E.T; t++) if (E.reqAt[t] >= 0) return t; return -1; }")
        if inflight >= 0:
            pg.evaluate(f"setTick({inflight})")
            pg.wait_for_timeout(200)
            st = pg.inner_text("#statusBox")
            ok("request in flight" in st and pg.is_disabled("#bLock"), f"in-flight tick {inflight}: labelled, lock disabled")
            pg.screenshot(path=f"{shots}/02_inflight.png", full_page=False)
            pg.click("#bNearest")
            ok(pg.evaluate("curEp().decAt[S.tick]") >= 0, "jump to nearest decision")
        # click on the timeline seeks and pauses
        pg.locator("#timeline canvas").scroll_into_view_if_needed()
        pg.evaluate("window.scrollBy(0, -200)")
        box = pg.locator("#timeline canvas").bounding_box()
        pg.click("#bPlay")
        pg.locator("#timeline canvas").click(position=dict(x=64 + (box["width"] - 76) * 0.5, y=40))
        T = pg.evaluate("curEp().T")
        tt = pg.evaluate("S.tick")
        ok(abs(tt - T / 2) <= 3 and not pg.evaluate("S.playing"), f"timeline click paused and seeked to {tt} (~T/2={T / 2})")
        # chart click seeks
        pg.locator("#epPlots canvas").first.scroll_into_view_if_needed()
        cb = pg.locator("#epPlots canvas").first.bounding_box()
        pg.locator("#epPlots canvas").first.click(position=dict(x=64 + (cb["width"] - 76) * 0.25, y=60))
        ok(abs(pg.evaluate("S.tick") - T / 4) <= 3, "episode chart click seeks")
        # switch checkpoint keeps seed and tick
        pg.click("#bNextDec")
        tk = pg.evaluate("S.tick")
        ck2 = idx["checkpoints"][0]["id"]
        pg.evaluate(f"selectPolicy('{ck2}')")
        pg.wait_for_function(f"() => curEp() && curEp().id === '{ck2}_s{seed}'", timeout=30000)
        ok(pg.evaluate("S.seed") == seed and pg.evaluate("S.tick") == min(tk, pg.evaluate("curEp().T")), "checkpoint switch preserves seed and tick")
        ok("physical state" in pg.inner_text("#modeBanner"), "switch notice shown")
        # baseline
        pg.click("#btnBaseline")
        pg.wait_for_function(f"() => curEp() && curEp().id === 'baseline_s{seed}'", timeout=30000)
        ok("Fixed baseline" in pg.inner_text("#ckRead"), "fixed baseline selectable separately")
        # lock mode on a learned rollout
        pg.evaluate(f"selectPolicy('{ck_last}')")
        pg.wait_for_function(f"() => curEp() && curEp().id === '{ck_last}_s{seed}'", timeout=30000)
        pg.evaluate("setTick(curEp().meta.decisions.t[Math.floor(curEp().n / 2)])")
        pg.click("#bLock")
        pg.wait_for_function("() => S.lock && document.querySelector('#acrossPlots canvas')", timeout=60000)
        pg.wait_for_timeout(500)
        before = pg.evaluate("JSON.stringify({eid: S.lock.eid, i: S.lock.i, t: viewTick()})")
        inp0 = pg.evaluate("Array.from(App.eps.get(S.lock.eid).input(S.lock.i)).slice(0, 20)")
        pg.screenshot(path=f"{shots}/03_locked_last.png", full_page=True)
        res = []
        for c in idx["checkpoints"]:
            pg.evaluate(f"selectPolicy('{c['id']}')")
            pg.wait_for_timeout(60)
            v = pg.evaluate("""() => { const v = currentView(); let s = 0; for (let a = 0; a < 201; a++) s += v.pOn[a];
                return {probe: v.probeId, sum: s, pk: v.stOn.pKeep, pkoff: v.stOff.pKeep, eq: v.strength === 0, mask: Array.from(v.mask).join('')}; }""")
            res.append(v)
        after = pg.evaluate("JSON.stringify({eid: S.lock.eid, i: S.lock.i, t: viewTick()})")
        ok(before == after, "lock: source episode / decision / tick unchanged while the slider moves")
        ok(pg.evaluate("Array.from(App.eps.get(S.lock.eid).input(S.lock.i)).slice(0, 20)") == inp0, "lock: frozen input identical")
        ok(len({r["mask"] for r in res}) == 1, "lock: same feasibility mask for every probed checkpoint")
        ok(all(abs(r["sum"] - 1) < 1e-6 for r in res), "lock: every probed distribution sums to 1")
        ok(all(abs(r["pk"] - r["pkoff"]) < 1e-12 for r in res if r["eq"]), "lock: prior-on == prior-off where strength is 0")
        own = pg.evaluate(f"""() => {{ selectPolicy('{ck_last}'); const v = currentView(); const E = App.eps.get(S.lock.eid);
            return {{arg: v.stOn.argmax, rec: E.meta.decisions.action[S.lock.i]}}; }}""")
        ok(own["arg"] == own["rec"], "lock: probing the source checkpoint reproduces the recorded action")
        pg.evaluate(f"selectPolicy('{idx['checkpoints'][1]['id']}')")
        pg.wait_for_timeout(300)
        pg.screenshot(path=f"{shots}/04_locked_probe_early.png", full_page=True)
        # across-checkpoint chart click selects a checkpoint
        pg.locator("#acrossPlots canvas").first.scroll_into_view_if_needed()
        ab = pg.locator("#acrossPlots canvas").first.bounding_box()
        pg.locator("#acrossPlots canvas").first.click(position=dict(x=64 + (ab["width"] - 76) * ((300155 + 8000) / 316000), y=80))
        pg.wait_for_timeout(200)
        ok(pg.evaluate("S.lock.probe") == ck_last, "across-checkpoint chart click selects the nearest checkpoint")
        for v in ("qdis", "adv", "prob"):
            pg.click(f"#critView button[data-v={v}]")
        pg.click("#probView button[data-v=cond]")
        pg.click("#scaleView button[data-v=lin]")
        pg.wait_for_timeout(200)
        pg.screenshot(path=f"{shots}/05_locked_views.png", full_page=False)
        pg.click("#probView button[data-v=joint]")
        pg.click("#scaleView button[data-v=log]")
        pg.click("#critView button[data-v=qmin]")
        pg.click("#bLock")
        ok(pg.evaluate("S.lock") is None, "unlock returns to playback")
        # learning tab
        pg.click("button[data-tab=learning]")
        pg.wait_for_timeout(800)
        pg.screenshot(path=f"{shots}/06_learning.png", full_page=True)
        pg.locator("#learnPlots canvas").first.scroll_into_view_if_needed()
        lb = pg.locator("#learnPlots canvas").first.bounding_box()
        pg.locator("#learnPlots canvas").first.click(position=dict(x=64 + (lb["width"] - 76) * ((25146 + 5000) / 310000), y=80))
        pg.wait_for_timeout(500)
        want = min(idx["checkpoints"], key=lambda c: abs(c["train_steps"] - 25146))["id"]
        ok(pg.evaluate("S.pol") == want, f"learning chart click selects checkpoint ({pg.evaluate('S.pol')})")
        pg.click("button[data-tab=about]")
        pg.wait_for_timeout(300)
        pg.screenshot(path=f"{shots}/07_about.png", full_page=True)
        pg.click("button[data-tab=playback]")
        pg.wait_for_timeout(500)
        # narrow (phone) layout renders without horizontal scroll
        pg.set_viewport_size(dict(width=420, height=900))
        pg.wait_for_timeout(500)
        sw = pg.evaluate("document.documentElement.scrollWidth")
        ok(sw <= 430, f"narrow layout: no horizontal page scroll (scrollWidth {sw})")
        pg.screenshot(path=f"{shots}/08_narrow.png", full_page=False)
        b.close()
    ok(not errors, f"no console errors ({errors[:3]})")
    n_fail = sum(r.startswith("FAIL") for r in results)
    json.dump(dict(results=results, console_errors=errors), open(os.path.join(shots, "ui_test.json"), "w"), indent=1)
    print(f"UI TEST: {len(results) - n_fail} passed, {n_fail} failed; screenshots in {shots}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
