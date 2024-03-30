#!/bin/bash

# These are with newly adjusting hyperparameters

python main_reconstruction_exp.py --with_tracking --run_name latents-100-0_1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 100 --tuning_steps 0 --reg_coeff 1
python main_reconstruction_exp.py --with_tracking --run_name latents-200-0_1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 200 --tuning_steps 0 --reg_coeff 1
python main_reconstruction_exp.py --with_tracking --run_name latents-400-0_1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 300 --tuning_steps 0 --reg_coeff 1
python main_reconstruction_exp.py --with_tracking --run_name latents-800-0_1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 400 --tuning_steps 0 --reg_coeff 1
python main_reconstruction_exp.py --with_tracking --run_name latents-1600-0_1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 500 --tuning_steps 0 --reg_coeff 1

python main_reconstruction_exp.py --with_tracking --run_name latents-100-0_0.1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.1
python main_reconstruction_exp.py --with_tracking --run_name latents-200-0_0.1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 200 --tuning_steps 0 --reg_coeff 0.1
python main_reconstruction_exp.py --with_tracking --run_name latents-400-0_0.1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 300 --tuning_steps 0 --reg_coeff 0.1
python main_reconstruction_exp.py --with_tracking --run_name latents-800-0_0.1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 400 --tuning_steps 0 --reg_coeff 0.1
python main_reconstruction_exp.py --with_tracking --run_name latents-1600-0_0.1 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 500 --tuning_steps 0 --reg_coeff 0.1

python main_reconstruction_exp.py --with_tracking --run_name latents-100-0 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 100 --tuning_steps 0 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name latents-200-0 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 200 --tuning_steps 0 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name latents-400-0 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 300 --tuning_steps 0 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name latents-800-0 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 400 --tuning_steps 0 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name latents-1600-0 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 500 --tuning_steps 0 --reg_coeff 0.0