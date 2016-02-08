### EEG signal classification

Training model for classification of EEG samples into motor imagery classes to be used for a simple Brain-Computer Interface.

#### Data:

Data has been taken from the dataset created and contributed to PhysioNet by the developers of the BCI2000 instrumentation system. To download the dataset:

* `wget -r --no-parent --accept "*.edf"  "http://www.physionet.org/pn4/eegmmidb/"`
* `mv www.physionet.org/pn4/eegmmidb data/`


#### Dependencies: 

* eegtools
* scikit-learn
* numpy
* scipy