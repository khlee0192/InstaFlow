#!/bin/bash

# These are with newly adjusting hyperparameters

python main_reconstruction_exp.py --with_tracking --run_name 50-100-50_noreducelr --start 0 --end 10 --decoder_inv_steps 50 --forward_steps 100 --tuning_steps 50 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 100-100-100_noreducelr --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 100 --tuning_steps 100 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 200-100-200_noreducelr --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 200 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 400-100-400_noreducelr --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 400 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 800-100-800_noreducelr --start 0 --end 10 --decoder_inv_steps 800 --forward_steps 100 --tuning_steps 800 --reg_coeff 0.0