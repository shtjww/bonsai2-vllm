#!/bin/bash
# network RX rate probe over 10s
DEV=$(ls /sys/class/net | grep -v lo | head -1)
R1=$(cat /sys/class/net/$DEV/statistics/rx_bytes)
sleep 10
R2=$(cat /sys/class/net/$DEV/statistics/rx_bytes)
echo "iface=$DEV rx_rate=$(( (R2-R1)/10/1024/1024 )) MB/s total_rx=$(( R2/1024/1024/1024 )) GB"
pip show vllm 2>/dev/null | head -2 || /root/miniconda3/bin/pip show vllm 2>/dev/null | head -2 || echo "vllm not installed yet"
