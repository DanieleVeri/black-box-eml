#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Missing python file path to execute."
    exit
fi

while true
do
    python -m ptvsd --host 0.0.0.0 --port 5678 --wait --multiprocess $1
done
