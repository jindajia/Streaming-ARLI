"""Online replay for SCH-01 (starts EMPTY; only this run's own training interactions are added).

Tables: frames [F, 64, 64, 3] uint8 (each distinct frame stored once), states [S] (frame refs + queue + robot +
runtime + feasibility mask + reference action), decisions [D] (state row, action, G, D, terminal, next-state row,
behaviour probability).  Checkpoints append only rows added since the previous save (part files).
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np
from flax.core import frozen_dict

from sch02 import core


class Replay:
    def __init__(self, cap_frames, cap_states, cap_dec, seed):
        A = core.NUM_ACTIONS
        self.frames = np.empty((cap_frames, 64, 64, 3), np.uint8)
        self.st = dict(f_now=np.empty(cap_states, np.int64), f_src=np.empty(cap_states, np.int64),
                       queue=np.empty((cap_states, core.K_CAP, 14), np.float32),
                       qmask=np.empty((cap_states, core.K_CAP), np.float32), robot=np.empty((cap_states, 28), np.float32),
                       runtime=np.empty((cap_states, 5), np.float32), mask=np.empty((cap_states, A), bool),
                       ref=np.empty(cap_states, np.int32), nfeas=np.empty(cap_states, np.float32))
        self.dec = dict(s=np.empty(cap_dec, np.int64), n=np.empty(cap_dec, np.int64), action=np.empty(cap_dec, np.int32),
                        G=np.empty(cap_dec, np.float32), D=np.empty(cap_dec, np.float32),
                        terminal=np.empty(cap_dec, np.float32), behav_p=np.empty(cap_dec, np.float32))
        self.nf = self.ns = self.nd = 0
        self.saved = (0, 0, 0)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.nd

    def _frame(self, img, cache):
        key = id(img)
        if key not in cache:
            self.frames[self.nf] = img
            cache[key] = self.nf
            self.nf += 1
        return cache[key]

    def _state(self, st, mask, cache):
        i = self.ns
        self.st["f_now"][i] = self._frame(st["img_now"], cache)
        self.st["f_src"][i] = self._frame(st["img_src"], cache)
        for k in ("queue", "qmask", "robot", "runtime"):
            self.st[k][i] = st[k]
        self.st["mask"][i] = mask
        self.st["ref"][i] = core.reference_index(st["q"], mask) if mask.any() else 0
        self.st["nfeas"][i] = mask.sum()
        self.ns += 1
        return i

    def add_episode(self, decisions):
        """decisions from Executor.run (training episodes only)."""
        cache, srow = {}, {}
        for d in decisions:
            key = id(d["state"])
            if key not in srow:
                srow[key] = self._state(d["state"], d["mask"], cache)
            nkey = id(d["next_state"])
            if nkey not in srow:
                srow[nkey] = self._state(d["next_state"], d["next_mask"], cache)
            j = self.nd
            self.dec["s"][j], self.dec["n"][j] = srow[key], srow[nkey]
            self.dec["action"][j], self.dec["G"][j], self.dec["D"][j] = d["action"], d["G"], d["D"]
            self.dec["terminal"][j] = float(d["terminal"])
            self.dec["behav_p"][j] = d["info"].get("p", np.nan)
            self.nd += 1
        return len(decisions)

    def sample(self, b):
        idx = self.rng.integers(self.nd, size=b)
        out = {}
        for pre, rows in (("s_", self.dec["s"][idx]), ("n_", self.dec["n"][idx])):
            out[pre + "img_now"] = self.frames[self.st["f_now"][rows]]
            out[pre + "img_src"] = self.frames[self.st["f_src"][rows]]
            for k in ("queue", "qmask", "robot", "runtime", "mask", "ref", "nfeas"):
                out[pre + k] = self.st[k][rows]
        for k in ("action", "G", "D", "terminal"):
            out[k] = self.dec[k][idx]
        return frozen_dict.freeze(out)

    # -- append-only checkpointing -----------------------------------------------------------------------
    def save_new(self, d):
        os.makedirs(d, exist_ok=True)
        (f0, s0, d0), (f1, s1, d1) = self.saved, (self.nf, self.ns, self.nd)
        if (f1, s1, d1) != (f0, s0, d0):
            part = dict(rng=None, f=(f0, f1, self.frames[f0:f1]), s=(s0, s1, {k: v[s0:s1] for k, v in self.st.items()}),
                        d=(d0, d1, {k: v[d0:d1] for k, v in self.dec.items()}))
            name = f"{d}/part_{d0:09d}_{d1:09d}.pkl"
            with open(name + ".tmp", "wb") as fh:
                pickle.dump(part, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(name + ".tmp", name)
        self.saved = (f1, s1, d1)
        return dict(nf=f1, ns=s1, nd=d1, rng=self.rng.bit_generator.state)

    def load(self, d, meta):
        for p in sorted(glob.glob(f"{d}/part_*.pkl")):
            d0 = int(os.path.basename(p).split("_")[1])
            if d0 >= meta["nd"]:
                os.remove(p)  # written after the checkpoint we resume from
                continue
            part = pickle.load(open(p, "rb"))
            (f0, f1, fr), (s0, s1, st), (q0, q1, dc) = part["f"], part["s"], part["d"]
            self.frames[f0:f1] = fr
            for k, v in st.items():
                self.st[k][s0:s1] = v
            for k, v in dc.items():
                self.dec[k][q0:q1] = v
        self.nf, self.ns, self.nd = meta["nf"], meta["ns"], meta["nd"]
        self.saved = (self.nf, self.ns, self.nd)
        self.rng.bit_generator.state = meta["rng"]
