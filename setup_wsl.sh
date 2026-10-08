#!/usr/bin/env bash
# Prenese projekt z Windows do Linuxu (WSL) a pripravi tam PyTorch.
# Duvod: Smart App Control ve Windows blokuje knihovny PyTorche.
# Linuxove knihovny uvnitr WSL tahle kontrola neresi a grafika jede
# pres ovladac z Windows, takze tu nic dalsiho instalovat nemusis.
#
#   bash /mnt/e/vlastni-ai/setup_wsl.sh
set -euo pipefail

WIN=/mnt/e/vlastni-ai
DST="$HOME/vlastni-ai"

echo "== systemove balicky (zepta se na heslo do Ubuntu) =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip rsync

echo "== kopiruji projekt do $DST =="
# data/ (1,3 GB surove Wikipedie) netreba, trenink bere uz pripravene out/*.bin.
# Na linuxovem disku to pobezi znatelne rychleji nez pres /mnt/c.
mkdir -p "$DST"
rsync -a --info=progress2 \
      --exclude .venv --exclude __pycache__ --exclude data \
      "$WIN/" "$DST/"
# soubory ulozene ve Windows mivaji konce radku CRLF, na kterych bash spadne
sed -i 's/\r$//' "$DST"/*.sh
chmod +x "$DST/trenuj_dal.sh"

echo "== virtualni prostredi a PyTorch s CUDA =="
cd "$DST"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install torch numpy \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple

echo "== kontrola grafiky =="
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA ve WSL nevidi grafiku"
print("OK:", torch.__version__, "|", torch.cuda.get_device_name(0))
EOF

echo
echo "Hotovo. Trenink: ~/vlastni-ai/trenuj_dal.sh"
