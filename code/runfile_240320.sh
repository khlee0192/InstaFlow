#!/bin/bash

# These are with newly adjusting hyperparameters

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-100_tuninglr_0.05 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 100 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-200_tuninglr_0.05 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 200 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-300_tuninglr_0.05 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 300 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-400_tuninglr_0.05 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 400 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-500_tuninglr_0.05 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 500 --reg_coeff 0.0