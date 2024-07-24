#!/bin/bash
dataset=$1
port=$2

tensorboard --logdir  ../${dataset}_experiment/runs/ --port ${port}