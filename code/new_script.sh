#!/bin/bash

python main_new_algorithm.py --run_name adam_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --test_beta --with_tracking
python main_new_algorithm.py --run_name adam_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --test_beta --with_tracking
python main_new_algorithm.py --run_name adam_50steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --test_beta --with_tracking
python main_new_algorithm.py --run_name adam_100steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --test_beta --with_tracking