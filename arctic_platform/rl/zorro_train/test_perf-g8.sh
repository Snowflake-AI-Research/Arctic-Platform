#!/bin/bash
deepspeed --num_gpus 8 test_perf.py |& tee test_perf-g8.log
