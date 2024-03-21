#!/bin/bash

python main_reconstruction_exp.py --with_tracking --run_name 1000-1000-0 --start 0 --end 10 --decoder_inv_steps 1000 --forward_steps 1000 --tuning_steps 0 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 250-1000-0 --start 0 --end 10 --decoder_inv_steps 250 --forward_steps 1000 --tuning_steps 0 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 500-1000-0 --start 0 --end 10 --decoder_inv_steps 500 --forward_steps 1000 --tuning_steps 0 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 2000-1000-0 --start 0 --end 10 --decoder_inv_steps 2000 --forward_steps 1000 --tuning_steps 0 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 4000-1000-0 --start 0 --end 10 --decoder_inv_steps 4000 --forward_steps 1000 --tuning_steps 0 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-100 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 100 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-200 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 200 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-300 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 300 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-400 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 400 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 100-1000-500 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 1000 --tuning_steps 500 --reg_coeff 0.0