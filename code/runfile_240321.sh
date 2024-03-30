#!/bin/bash

# These are with newly adjusting hyperparameters

python main_reconstruction_exp.py --with_tracking --run_name 1600-100-0 --start 0 --end 10 --decoder_inv_steps 1600 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 800-100-0 --start 0 --end 10 --decoder_inv_steps 800 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 400-100-0 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 50-100-50 --start 0 --end 10 --decoder_inv_steps 50 --forward_steps 100 --tuning_steps 50 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 100-100-100 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 100 --tuning_steps 100 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 200-100-200 --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 200 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 400-100-400 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 400 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 800-100-800 --start 0 --end 10 --decoder_inv_steps 800 --forward_steps 100 --tuning_steps 800 --reg_coeff 0.0


# python main_reconstruction_exp.py --with_tracking --run_name 200-100-100 --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 100 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 200-100-200 --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 200 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 200-100-300 --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 300 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 200-100-400 --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 400 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 200-100-500 --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 500 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 300-100-100 --start 0 --end 10 --decoder_inv_steps 300 --forward_steps 100 --tuning_steps 100 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 300-100-200 --start 0 --end 10 --decoder_inv_steps 300 --forward_steps 100 --tuning_steps 200 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 300-100-300 --start 0 --end 10 --decoder_inv_steps 300 --forward_steps 100 --tuning_steps 300 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 300-100-400 --start 0 --end 10 --decoder_inv_steps 300 --forward_steps 100 --tuning_steps 400 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 300-100-500 --start 0 --end 10 --decoder_inv_steps 300 --forward_steps 100 --tuning_steps 500 --reg_coeff 0.0



# python main_reconstruction_exp.py --with_tracking --run_name 400-100-100 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 100 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 400-100-200 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 200 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 400-100-300 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 300 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 400-100-400 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 400 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 400-100-500 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 500 --reg_coeff 0.0


# python main_reconstruction_exp.py --with_tracking --run_name 500-100-100 --start 0 --end 10 --decoder_inv_steps 500 --forward_steps 100 --tuning_steps 100 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 500-100-200 --start 0 --end 10 --decoder_inv_steps 500 --forward_steps 100 --tuning_steps 200 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 500-100-300 --start 0 --end 10 --decoder_inv_steps 500 --forward_steps 100 --tuning_steps 300 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 500-100-400 --start 0 --end 10 --decoder_inv_steps 500 --forward_steps 100 --tuning_steps 400 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 500-100-500 --start 0 --end 10 --decoder_inv_steps 500 --forward_steps 100 --tuning_steps 500 --reg_coeff 0.0


# python main_reconstruction_exp.py --with_tracking --run_name 100-100-0 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 200-100-0 --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 300-100-0 --start 0 --end 10 --decoder_inv_steps 300 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 400-100-0 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 500-100-0 --start 0 --end 10 --decoder_inv_steps 500 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 600-100-0 --start 0 --end 10 --decoder_inv_steps 600 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 700-100-0 --start 0 --end 10 --decoder_inv_steps 700 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 800-100-0 --start 0 --end 10 --decoder_inv_steps 800 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 900-100-0 --start 0 --end 10 --decoder_inv_steps 900 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0

# python main_reconstruction_exp.py --with_tracking --run_name 1000-100-0 --start 0 --end 10 --decoder_inv_steps 00 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0