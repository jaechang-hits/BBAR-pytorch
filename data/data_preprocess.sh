echo ">>> Run Directory: $1"
echo ">>> Number of CPUs: $2"

if [ -z "$2" ]
then
    echo "ERROR: Expected Two Parameters"
    echo "Example: ./data_preprocess.sh molport/ 4"
else
    echo ""
    echo ">>> get_library.py start (get minimum size fragments)"
    python preprocessing/get_library.py $1 --cpus $2
    echo ">>> get_library.py finish"

    echo ">>> get_datapoint.py (train data) start (get train datapoints)"
    python preprocessing/get_datapoint.py $1 --cpus $2 --mol train_smiles.csv --output train.csv
    python preprocessing/get_frag1_freq.py $1
    echo ">>> get_datapoint.py (train data) finish"

    echo ">>> get_datapoint.py (validation data) start (get validation datapoints)"
    python preprocessing/get_datapoint.py $1 --cpus $2 --mol val_smiles.csv --output val.csv
    echo ">>> get_datapoint.py (validation data) finish"

    #echo ">>> get_datapoint.py (test data) start (get test datapoints)"
    #python preprocessing/get_datapoint.py $1 --cpus $2 --mol val_smiles.csv --output val.csv
    #echo ">>> get_datapoint.py (test data) finish"

fi
