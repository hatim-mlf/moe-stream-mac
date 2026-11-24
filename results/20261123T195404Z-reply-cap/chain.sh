cd /Users/hatim/moe-stream-mac
.venv/bin/python tools/pi_agent_eval.py run --arm kat --max-tokens 32768 --cases 1575_D-python3,1580_A-cpp17,1580_A-python3,1604_D-python3,1622_C-cpp17,1622_C-python3,1598_E-cpp17,1598_E-python3,1600_E-python3,1613_D-cpp17,1613_D-python3 --out results/20261123T195404Z-reply-cap
echo "KAT-DONE rc=$?"
.venv/bin/python tools/pi_agent_eval.py run --arm pi --max-tokens 32768 --cases 1604_D-cpp17,1604_D-python3,1622_C-cpp17,1622_C-python3,1598_E-cpp17,1613_D-cpp17,1613_D-python3 --out results/20261123T195404Z-reply-cap
echo "PI-DONE rc=$?"
echo CHAIN-DONE
