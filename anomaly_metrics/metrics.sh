
path_to_the_generated_data="/data4/jiangtianjia/datasets/refdata/anogen_gen"
path_to_mvtec="/data4/jiangtianjia/datasets/mvtec"
result_localization_save="anogen100_results/loc_result.csv"
result_classification_save="anogen100_results/cla_result.csv"

#train and test the anomaly detection model
python train-localization.py --generated_data_path $path_to_the_generated_data  --mvtec_path=$path_to_mvtec
python test-localization.py --mvtec_path $path_to_mvtec --result_save $result_localization_save

#train and test anomaly classification model
python train-classification.py --mvtec_path=$path_to_mvtec --generated_data_path=$path_to_the_generated_data
python test-classification.py --mvtec_path=$path_to_mvtec --generated_data_path=$path_to_the_generated_data --result_save $result_classification_save