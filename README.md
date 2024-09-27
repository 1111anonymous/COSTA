# COSTA

This repository is the implementation of COSTA.

COSTA is short for '**Co**ntrastive **S**patial and **T**emporal Debi**a**sing framework for next POI recommendation'. It is a novel framework, equipped with four modules (i.e., Long-short Term Preference Encoder, User-side Spatial-temporal Signal Encoding, Location-side Spatial-temporal Signal Encoding and Contrast Debias Module), which is designed to alleviate the spatial and temporal
biases without compromising recommendation accuracy.


### File Descriptions

- `raw_data/`
  - `NYC.rar`: check-in information of New York;
  - `PHO.csv`: check-in information of Phoenix;
  - `SIN.rar`: : check-in information of Singapore;
- `main.py`: main file;
- `model.py`: COSTA model file;
- `settings.py`: parameter settings file;



### More Experimental Settings
- Environment
  - Our proposed COSTA is implemented using pytorch 1.10.1, with Python 3.7.11 from Anaconda 4.3.30. All the experiments are carried out on a machine with Windows 10, Intel CORE i7-8565U CPU, NIVIDA GeForce RTX 2080 and 16G RAM. The following packages are needed (along with their dependencies):
    - cuda==11.0
    - numpy==1.19.5
    - pandas==1.1.5
    - python==3.6.3
    - torch==1.10.1
    - Pillow==9.4.0
    - python-dateutil==2.8.2
- Data Preprocessing
  - Following state-of-the-arts, for each user, we chronologically divide his check-in records into different trajectories by day, and then take the earlier 80% of his trajectories as training set; the latest 10% of trajectories as the test set; and the rest 10% as the validation set. Besides, we filter out POIs with fewer than 10 interactions, inactive users with fewer than 5 trajectories, and trajectories with fewer than 3 check-in records.


### How To Run
```
$ python main.py
```
