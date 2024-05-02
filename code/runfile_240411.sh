#!/bin/bash

# python main_new_algorithm.py --with_tracking --run_name 200-100-0_orig --start 0 --end 10 --decoder_inv_steps 200 --inversion_type dec_inv --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0
# python main_new_algorithm.py --with_tracking --run_name 100-100-0_orig --start 0 --end 10 --decoder_inv_steps 100 --inversion_type dec_inv --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0
# python main_new_algorithm.py --with_tracking --run_name 50-100-0_orig --start 0 --end 10 --decoder_inv_steps 50 --inversion_type dec_inv --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0
# python main_new_algorithm.py --with_tracking --run_name 25-100-0_orig --start 0 --end 10 --decoder_inv_steps 25 --inversion_type dec_inv --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0

python main_new_algorithm.py --with_tracking --run_name 200-100-0_new_16 --start 0 --end 10 --decoder_inv_steps 200 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0 --use_half
python main_new_algorithm.py --with_tracking --run_name 100-100-0_new_16 --start 0 --end 10 --decoder_inv_steps 100 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0 --use_half
python main_new_algorithm.py --with_tracking --run_name 50-100-0_new_16 --start 0 --end 10 --decoder_inv_steps 50 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0 --use_half
python main_new_algorithm.py --with_tracking --run_name 25-100-0_new_16 --start 0 --end 10 --decoder_inv_steps 25 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0 --use_half

python main_new_algorithm.py --with_tracking --run_name 200-100-0_new_16 --start 0 --end 10 --decoder_inv_steps 200 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0
python main_new_algorithm.py --with_tracking --run_name 100-100-0_new_16 --start 0 --end 10 --decoder_inv_steps 100 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0
python main_new_algorithm.py --with_tracking --run_name 50-100-0_new_16 --start 0 --end 10 --decoder_inv_steps 50 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0
python main_new_algorithm.py --with_tracking --run_name 25-100-0_new_16 --start 0 --end 10 --decoder_inv_steps 25 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0