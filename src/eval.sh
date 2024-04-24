#!/bin/bash

python robust_eval.py \
	--dataset cifar10 \
	--model_type resnet18 \
	--model_path ../cifar10_experiment/test/final_199.pt \
	--activation_name relu \