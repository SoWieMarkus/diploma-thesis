#!/bin/bash

case "$1" in
    "main")
        python main.py "${@:2}"
        ;;
    "visualizer")
        python visualizer.py "${@:2}"
        ;;
    "demo")
        python demo.py "${@:2}"
        ;;
    "evaluate")
        python evaluate.py "${@:2}"
        ;;
    *)
        echo "Usage: $0 {main|visualizer|demo|evaluate} [additional arguments]"
        echo "Available commands:"
        echo "  main       - Run main.py"
        echo "  visualizer - Run visualizer.py"
        echo "  demo       - Run demo.py"
        echo "  evaluate   - Run evaluate.py"
        exit 1
        ;;
esac