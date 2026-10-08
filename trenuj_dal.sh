#!/usr/bin/env bash
# Linuxova (WSL) obdoba trenuj_dal.cmd: pousti trenink, po padu ho do minuty
# nastartuje znovu od posledni zalohy, a kdyz pada opakovane bez jakehokoli
# postupu, vzda to - trvalou chybu nema smysl zkouset donekonecna.
#
#   ./trenuj_dal.sh        plny vykon
#   ./trenuj_dal.sh 0.5    polovicni vykon
cd "$(dirname "$0")"

VYKON="${1:-1.0}"
MAX_MARNYCH=5
ted() { date '+%d.%m.%Y %H:%M:%S'; }

# dva treninky najednou by se poprali o pamet grafiky a spadly by oba
if pgrep -f "python.*train.py" > /dev/null; then
    echo "[$(ted)] trenink uz bezi, tenhle se nespousti"
    exit 0
fi

marne=0
pokus=0
while true; do
    echo "[$(ted)] spoustim trenink (vykon $VYKON, restartu zatim $pokus)"
    pred=$(stat -c %Y out/model_last.pt 2>/dev/null || echo 0)

    .venv/bin/python train.py --resume --vykon "$VYKON" --batch-size 3 >> trenink.log 2>&1
    kod=$?

    if [ "$kod" -eq 0 ]; then
        echo "[$(ted)] TRENINK DOKONCEN" | tee -a trenink.log
        exit 0
    fi

    pokus=$((pokus + 1))
    # postup se pozna podle toho, ze se mezitim ulozila nova zaloha
    po=$(stat -c %Y out/model_last.pt 2>/dev/null || echo 0)
    if [ "$pred" = "$po" ]; then marne=$((marne + 1)); else marne=0; fi

    if [ "$marne" -ge "$MAX_MARNYCH" ]; then
        echo "[$(ted)] $marne padu po sobe bez postupu, posledni chyba $kod - vzdavam to" | tee -a trenink.log
        exit "$kod"
    fi

    echo "[$(ted)] trenink skoncil s chybou $kod, restart za 60 s (marnych pokusu $marne/$MAX_MARNYCH)" | tee -a trenink.log
    sleep 60
done
