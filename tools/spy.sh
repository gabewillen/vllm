#!/bin/bash
until grep -aq "spin start" $HOME/work/logs/$1.txt; do sleep 1; done; sleep 5
E=$(pgrep -f "VLLM::EngineCore" | head -1); W=$(pgrep -f "VLLM::Worker_TP0" | head -1); F=$(pgrep -f "python -u tools/run_llm.py" | head -1)
sudo $HOME/.local/bin/py-spy record --pid $E -d 15 -r 200 --idle -f raw -o $HOME/work/logs/spy_engine.txt >/dev/null 2>&1 &
sudo $HOME/.local/bin/py-spy record --pid $F -d 15 -r 200 --idle -f raw -o $HOME/work/logs/spy_front.txt >/dev/null 2>&1 &
sudo $HOME/.local/bin/py-spy record --pid $W -d 15 -r 200 --idle -f raw -o $HOME/work/logs/spy_worker.txt >/dev/null 2>&1
wait; sudo chown shadeform $HOME/work/logs/spy_*.txt; echo spied
