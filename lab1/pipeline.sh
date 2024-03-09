#!/bin/bash
python3 ./ml_pipeline/1_data_creation.py
python3 ./ml_pipeline/2_model_preprocessing.py
python3 ./ml_pipeline/3_model_preparation.py
python3 ./ml_pipeline/4_model_testing.py