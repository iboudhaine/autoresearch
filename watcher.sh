#!/bin/bash
SESSION="autoresearch"
PROMPT="continue the experiment loop per program.md"

while true; do
    json=$(./claude_usage.sh)
    util=$(echo "$json" | jq -r '.five_hour.utilization')
    resets=$(echo "$json" | jq -r '.five_hour.resets_at')

    if (( $(echo "$util >= 99" | bc -l) )); then
        reset_epoch=$(date -d "$resets" +%s)
        now_epoch=$(date +%s)
        wait_secs=$(( reset_epoch - now_epoch + 30 ))
        echo "$(date): at ${util}%, sleeping ${wait_secs}s until $resets"
        sleep "$wait_secs"
        echo "$(date): sending continue"
        tmux send-keys -t "$SESSION" "$PROMPT" Enter
        sleep 60
    else
        sleep 300
    fi
done
