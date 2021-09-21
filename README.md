# Thesis-submit-CTG
Using CTU-CHB CTG dataset for prediction of Umbilical Artery pH value 

CTU-CHB dataset: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

Dataset convert to csv file: https://github.com/fabiom91/CTU-CHB_Physionet.org.git

LTC reference: https://github.com/raminmh/liquid_time_constant_networks/tree/master
	
## Files

File ctg_specific_pH.py is regression prediction

File ctg_classify_pH.py is classification

## Dataset prepare

Download the CTU-CHB files, convert those file to csv

After the script is finished there should be a file ```result/ctg_class/lstm_32.csv``` created, containing the following columns:
- ```window size```: The size of signal windows used for training, testing and validation
- ```best epoch```: Epoch number that achieved the best validation metric
- ```train loss```: Training loss achieved at the best epoch
- ```train accuracy```: Training metric achieved at the best epoch
- ```valid loss```: Validation loss achieved at the best epoch
- ```valid accuracy```: Best validation metric achieved during training
- ```test loss```: Loss on the test set
- ```test accuracy```: Metric on the test set
