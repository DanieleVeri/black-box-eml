#!/bin/bash

echo "Target:" $1
case $1 in

  "" | notebook)
    jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=experiments/notebooks
    ;;

  tests)
    python env/run_all_tests.py
    ;;

  debug)
    env/launch_debug.sh $2
    ;;

  *)
    echo -n "unknown command: $1"
    ;;
esac
