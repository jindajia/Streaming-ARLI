"""Fetch gs://openpi-assets/checkpoints/pi0_aloha_sim over public HTTPS with per-object MD5 verification.

The ARLI paper cites s3://openpi-assets/checkpoints/pi0_aloha_sim; that S3 bucket no longer exists
(NoSuchBucket, 2026-10-27) and openpi now serves the same path from GCS.  Writes a manifest JSON.
    python scripts/download_checkpoint.py --dst /N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim
"""
import argparse
import base64
import hashlib
import json
import os
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--dst", default="/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim")
ap.add_argument("--manifest", default="results/manifest/pi0_aloha_sim_objects.json")
args = ap.parse_args()
prefix = "checkpoints/pi0_aloha_sim/"
items, token = [], None
while True:
    q = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size,md5Hash,generation,updated),nextPageToken"}
    if token:
        q["pageToken"] = token
    url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urllib.parse.urlencode(q)
    r = json.load(urllib.request.urlopen(url))
    items += r.get("items", [])
    token = r.get("nextPageToken")
    if not token:
        break
for it in items:
    rel = it["name"][len(prefix):]
    if it["name"].endswith("/"):
        continue
    out = os.path.join(args.dst, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    want = base64.b64decode(it["md5Hash"]).hex()
    if os.path.exists(out) and os.path.getsize(out) == int(it["size"]):
        h = hashlib.md5(open(out, "rb").read()).hexdigest()
        if h == want:
            continue
    src = "https://storage.googleapis.com/openpi-assets/" + urllib.parse.quote(it["name"])
    h = hashlib.md5()
    with urllib.request.urlopen(src) as resp, open(out + ".part", "wb") as f:
        while chunk := resp.read(1 << 22):
            f.write(chunk)
            h.update(chunk)
    assert h.hexdigest() == want, (it["name"], h.hexdigest(), want)
    os.replace(out + ".part", out)
    print("ok", rel, it["size"], flush=True)
os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
with open(args.manifest, "w") as f:
    json.dump(dict(source="gs://openpi-assets/" + prefix, cited_as="s3://openpi-assets/" + prefix, local=args.dst,
                   objects=[dict(name=i["name"], size=int(i["size"]), md5_b64=i["md5Hash"], generation=i.get("generation"),
                                 updated=i.get("updated")) for i in items if not i["name"].endswith("/")]), f, indent=1)
print("DOWNLOAD_COMPLETE", len(items), "objects")
