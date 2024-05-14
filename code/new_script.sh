#!/bin/bash

# test script
python main_new_algorithm.py --run_name testing_args --start 0 --end 1 --decoder_inv_steps 20 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --with_tracking

# ## with 0.01
# python main_new_algorithm.py --run_name adam_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --test_beta --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.01 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

# python main_new_algorithm.py --run_name adam_50steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --test_beta  --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.01 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

# python main_new_algorithm.py --run_name adam_100steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg  --test_beta --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.01 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

# ## with 0.02
# python main_new_algorithm.py --run_name adam_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.02 --decoder_adam True --inversion_type new_alg  --test_beta --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.02 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.02 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

# python main_new_algorithm.py --run_name adam_50steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.02 --decoder_adam True --inversion_type new_alg --test_beta  --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.02 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.02 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

# python main_new_algorithm.py --run_name adam_100steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.02 --decoder_adam True --inversion_type new_alg  --test_beta --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.02 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.02 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

# ## with 0.005
# python main_new_algorithm.py --run_name adam_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.005 --decoder_adam True --inversion_type new_alg  --test_beta --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.005 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 20 --decoder_lr 0.005 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

# python main_new_algorithm.py --run_name adam_50steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.005 --decoder_adam True --inversion_type new_alg --test_beta  --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.005 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 50 --decoder_lr 0.005 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

# python main_new_algorithm.py --run_name adam_100steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.005 --decoder_adam True --inversion_type new_alg  --test_beta --with_tracking
# python main_new_algorithm.py --run_name adam_grad_float_20steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.005 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_20steps --start 0 --end 100 --decoder_inv_steps 100 --decoder_lr 0.005 --decoder_adam True --inversion_type new_alg --with_tracking --use_float

##### 30 steps
# python main_new_algorithm.py --run_name adam_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.02 --decoder_adam True --inversion_type new_alg --test_beta --with_tracking
# python main_new_algorithm.py --run_name adam_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --test_beta --with_tracking
# python main_new_algorithm.py --run_name adam_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.005 --decoder_adam True --inversion_type new_alg --test_beta --with_tracking

# python main_new_algorithm.py --run_name adam_grad_float_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.02 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_grad_float_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.01 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_grad_float_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.005 --decoder_adam True --inversion_type dec_inv --with_tracking --use_float

# python main_new_algorithm.py --run_name adam_float_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.02 --decoder_adam True --inversion_type new_alg --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.01 --decoder_adam True --inversion_type new_alg --with_tracking --use_float
# python main_new_algorithm.py --run_name adam_float_30steps --start 0 --end 100 --decoder_inv_steps 30 --decoder_lr 0.005 --decoder_adam True --inversion_type new_alg --with_tracking --use_float