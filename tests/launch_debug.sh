#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Missing python file path to execute."
    exit
fi

while true
do
    echo "========== Debug server lisenting on port 5678... =========="
    python -m ptvsd --host 0.0.0.0 --port 5678 --wait --multiprocess $1
    echo "========== Debug session terminated. =========="
done
