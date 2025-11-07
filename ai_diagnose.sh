#!/bin/bash

# AI-Driven System Troubleshooter

# Thresholds
CPU_LIMIT=90
MEM_LIMIT=10
DISK_LIMIT=90

echo "[INFO] Gathering system information..."

# Get CPU Usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}')
CPU_USAGE=${CPU_USAGE%.*}
echo "[INFO] CPU usage: $CPU_USAGE%"

# Get Free Memory %
MEM_TOTAL=$(free -m | awk '/Mem:/ {print $2}')
MEM_FREE=$(free -m | awk '/Mem:/ {print $4}')
MEM_PERCENT=$(( (MEM_FREE * 100) / MEM_TOTAL ))
echo "[INFO] Memory free: $MEM_PERCENT%"

# Get Disk Usage
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
echo "[INFO] Disk usage: $DISK_USAGE%"

echo

# Expert System Rules
# Rule 1: High CPU
if [ "$CPU_USAGE" -gt "$CPU_LIMIT" ]; then
  echo "[DIAGNOSIS] High CPU usage detected."
  echo "[SUGGESTION] Recommend checking background processes using 'top' or 'htop'."
  echo
fi

# Rule 2: Low Memory
if [ "$MEM_PERCENT" -lt "$MEM_LIMIT" ]; then
  echo "[DIAGNOSIS] Low memory detected."
  echo "[SUGGESTION] Recommend restarting heavy services or increasing swap space."
  echo
fi

# Rule 3: Low Disk Space
if [ "$DISK_USAGE" -gt "$DISK_LIMIT" ]; then
  echo "[DIAGNOSIS] Disk space is critically low."
  echo "[SUGGESTION] Clean up log files or unnecessary data in /var/log or /tmp."
  echo
fi

if [ "$CPU_USAGE" -le "$CPU_LIMIT" ] && [ "$MEM_PERCENT" -ge "$MEM_LIMIT" ] && [ "$DISK_USAGE" -le "$DISK_LIMIT" ]; then
  echo "[STATUS] System health appears to be normal."
fi
