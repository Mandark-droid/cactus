#!/bin/sh
echo "=== Memory Profile ==="
echo "--- Before model load ---"
cat /proc/meminfo | grep -E "MemFree|MemAvail"

/data/local/tmp/test_qwen3_moe /data/local/tmp/loggenix-moe Hello &
PID=$!
sleep 1
echo ""
echo "--- During init (PID=$PID) ---"
cat /proc/$PID/status 2>/dev/null | grep -E "VmPeak|VmRSS|VmSize|VmData|VmStk|RssAnon|RssFile|RssShmem"
sleep 3
echo ""
echo "--- During generation ---"
cat /proc/$PID/status 2>/dev/null | grep -E "VmPeak|VmRSS|VmSize|VmData|VmStk|RssAnon|RssFile|RssShmem"
wait $PID
echo ""
echo "--- After completion ---"
cat /proc/meminfo | grep -E "MemFree|MemAvail"
echo "=== Done ==="
