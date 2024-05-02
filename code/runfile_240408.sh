#!/bin/bash

# Testing new trial

python main_reconstruction_exp.py --with_tracking --run_name 25-100-0 --start 0 --end 10 --decoder_inv_steps 25 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 50-100-0 --start 0 --end 10 --decoder_inv_steps 50 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 100-100-0 --start 0 --end 10 --decoder_inv_steps 100 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 200-100-0 --start 0 --end 10 --decoder_inv_steps 200 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 400-100-0 --start 0 --end 10 --decoder_inv_steps 400 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 5-10-20 --start 0 --end 10 --decoder_inv_steps 5 --forward_steps 10 --tuning_steps 20 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 10-10-40 --start 0 --end 10 --decoder_inv_steps 10 --forward_steps 10 --tuning_steps 40 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 20-10-80 --start 0 --end 10 --decoder_inv_steps 20 --forward_steps 10 --tuning_steps 80 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 40-10-160 --start 0 --end 10 --decoder_inv_steps 40 --forward_steps 10 --tuning_steps 160 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 80-10-320 --start 0 --end 10 --decoder_inv_steps 80 --forward_steps 10 --tuning_steps 320 --tuning_lr 0.05 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 5-10-20 --start 0 --end 10 --decoder_inv_steps 5 --forward_steps 10 --tuning_steps 20 --tuning_lr 0.1 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 10-10-40 --start 0 --end 10 --decoder_inv_steps 10 --forward_steps 10 --tuning_steps 40 --tuning_lr 0.1 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 20-10-80 --start 0 --end 10 --decoder_inv_steps 20 --forward_steps 10 --tuning_steps 80 --tuning_lr 0.1 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 40-10-160 --start 0 --end 10 --decoder_inv_steps 40 --forward_steps 10 --tuning_steps 160 --tuning_lr 0.1 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 80-10-320 --start 0 --end 10 --decoder_inv_steps 80 --forward_steps 10 --tuning_steps 320 --tuning_lr 0.1 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 5-10-20 --start 0 --end 10 --decoder_inv_steps 5 --forward_steps 10 --tuning_steps 20 --tuning_lr 0.01 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 10-10-40 --start 0 --end 10 --decoder_inv_steps 10 --forward_steps 10 --tuning_steps 40 --tuning_lr 0.01 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 20-10-80 --start 0 --end 10 --decoder_inv_steps 20 --forward_steps 10 --tuning_steps 80 --tuning_lr 0.01 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 40-10-160 --start 0 --end 10 --decoder_inv_steps 40 --forward_steps 10 --tuning_steps 160 --tuning_lr 0.01 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 80-10-320 --start 0 --end 10 --decoder_inv_steps 80 --forward_steps 10 --tuning_steps 320 --tuning_lr 0.01 --reg_coeff 0.0


python main_reconstruction_exp.py --with_tracking --run_name 30-100-0 --start 0 --end 10 --decoder_inv_steps 30 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 60-100-0 --start 0 --end 10 --decoder_inv_steps 60 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 120-100-0 --start 0 --end 10 --decoder_inv_steps 120 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 240-100-0 --start 0 --end 10 --decoder_inv_steps 240 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 480-100-0 --start 0 --end 10 --decoder_inv_steps 480 --forward_steps 100 --tuning_steps 0 --tuning_lr 0.05 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 5-10-25 --start 0 --end 10 --decoder_inv_steps 5 --forward_steps 10 --tuning_steps 25 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 10-10-50 --start 0 --end 10 --decoder_inv_steps 10 --forward_steps 10 --tuning_steps 50 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 20-10-100 --start 0 --end 10 --decoder_inv_steps 20 --forward_steps 10 --tuning_steps 100 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 40-10-200 --start 0 --end 10 --decoder_inv_steps 40 --forward_steps 10 --tuning_steps 200 --tuning_lr 0.05 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 80-10-400 --start 0 --end 10 --decoder_inv_steps 80 --forward_steps 10 --tuning_steps 400 --tuning_lr 0.05 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 5-10-25 --start 0 --end 10 --decoder_inv_steps 5 --forward_steps 10 --tuning_steps 25 --tuning_lr 0.1 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 10-10-50 --start 0 --end 10 --decoder_inv_steps 10 --forward_steps 10 --tuning_steps 50 --tuning_lr 0.1 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 20-10-100 --start 0 --end 10 --decoder_inv_steps 20 --forward_steps 10 --tuning_steps 100 --tuning_lr 0.1 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 40-10-200 --start 0 --end 10 --decoder_inv_steps 40 --forward_steps 10 --tuning_steps 200 --tuning_lr 0.1 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 80-10-400 --start 0 --end 10 --decoder_inv_steps 80 --forward_steps 10 --tuning_steps 400 --tuning_lr 0.1 --reg_coeff 0.0

python main_reconstruction_exp.py --with_tracking --run_name 5-10-25 --start 0 --end 10 --decoder_inv_steps 5 --forward_steps 10 --tuning_steps 25 --tuning_lr 0.01 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 10-10-50 --start 0 --end 10 --decoder_inv_steps 10 --forward_steps 10 --tuning_steps 50 --tuning_lr 0.01 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 20-10-100 --start 0 --end 10 --decoder_inv_steps 20 --forward_steps 10 --tuning_steps 100 --tuning_lr 0.01 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 40-10-200 --start 0 --end 10 --decoder_inv_steps 40 --forward_steps 10 --tuning_steps 200 --tuning_lr 0.01 --reg_coeff 0.0
python main_reconstruction_exp.py --with_tracking --run_name 80-10-400 --start 0 --end 10 --decoder_inv_steps 80 --forward_steps 10 --tuning_steps 400 --tuning_lr 0.01 --reg_coeff 0.0