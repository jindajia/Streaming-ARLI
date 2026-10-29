#!/bin/bash
# Serve the SCH-01 fixed_delay10_v2 viewer (static files; no GPU) on 127.0.0.1:<port> of the node this runs on.
# From your laptop, one command starts the server on the login node you land on AND forwards the port to it:
#   ssh -L 8765:127.0.0.1:8765 jindjia@bigred200.uits.iu.edu \
#       bash /N/slate/jindjia/SD/VLA/streaming_arli_viewer/experiments/scheduling/fixed_delay10_v2/viewer/serve.sh 8765
# then open http://localhost:8765/ in a local browser.  Ctrl-C in that ssh session stops the server.
PORT=${1:-8765}
SITE=${2:-/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/viewer/site}
[ -f "$SITE/index.html" ] || { echo "no site at $SITE (run viewer/export/build_site.py)"; exit 1; }
echo "serving $SITE at http://127.0.0.1:$PORT/ on $(hostname) (Ctrl-C to stop)"
cd "$SITE" && exec /usr/bin/python3 -m http.server "$PORT" --bind 127.0.0.1
