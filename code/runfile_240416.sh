#!/bin/bash

python main_new_algorithm.py --with_tracking --run_name dec_inv_float32 --start 0 --end 10 --decoder_inv_steps 100 --inversion_type dec_inv --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0
python main_new_algorithm.py --with_tracking --run_name new_alg_float32 --start 0 --end 10 --decoder_inv_steps 100 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0
python main_new_algorithm.py --with_tracking --run_name new_alg_float16 --start 0 --end 10 --decoder_inv_steps 100 --inversion_type new_alg --forward_steps 100 --tuning_steps 0 --tuning_lr 0.0 --reg_coeff 0.0 --use_half