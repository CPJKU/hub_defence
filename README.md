# Defending a Music Recommender Against Hubness-Based Adversarial Attacks

This repository accompanies the publication: Katharina Hoedt, Arthur Flexer, and Gerhard Widmer,
'Defending a Music Recommender Against Hubness-Based Adversarial Attacks', published at the 
SMC-22 Scientific Paper Track. To view our supplementary material, containing listening examples, 
look at our [Github page](https://cpjku.github.io/hub_defence/). 

This research was supported by the Austrian Science Fund (FWF, P 31988). GW’s work is supported by the European
Research Council (ERC) under the EU’s Horizon 2020 research and innovation programme, grant agreement No 101019375
(Whither Music?). 

If you encounter any issues or bugs within this repo, do not hesitate to contact us!

## Setup

All required libraries are listed in the file `environment.yml`. 
If you have [conda](https://docs.anaconda.com/anaconda/install/index.html) available, you can create an 
environment for this project and activate it as follows:

````
conda env create -f environment.yml
conda activate mp
````

### Data

The system we perform our experiments on is the [Fm4 Soundpark](https://fm4.orf.at/soundpark/) [3], 
where various songs can be found and downloaded. We cannot provide the exact data we used 
in this work here, nevertheless the code should provide a good starting base for experiments. Note
that we assume every song to have a duration of at least 2 minutes in our experiments. For more information, 
please contact us directly.

[3] Martin Gasser and Arthur Flexer (2009). Fm4 soundpark: Audio-based music recommendation in everyday use. 
In Proc. of the 6th Sound and Music Computing Conference, SMC, pages 23–25.

## Experiments

### Data Processing

To improve the runtime efficiency, we use cached audio files and pre-compute the Gaussians and KL-divergences. 
Before all of your subsequent experiments, therefore run e.g., 

````
python -m mp.utils.preprocess --data_path <your-data-path> --cache_file <your-path/cache_file.hdf5> --save_path <your-save-path>
````

and replace the `data_path`, `cache_file` and `save_path` accordingly.


### Attack on Undefended System 

The first line in Table 1 of the paper shows the result of attacking a yet undefended music recommender.
This can be done by calling e.g., 

````
python -m mp.attack.soundpark_attack --cache_file <your-path/cache_file.h5py> --save_path <your-save-path>
````

In case you want to change e.g., the attack parameter, check out the configuration file (`attack/config.vars`)
and the available arguments in `attack/soundpark_attack.py`. Make sure to set the `cache_file` and the
`save_path` correctly (as defined during data preprocessing).

During the attack various values are stored in log files, which can be used to compute everything shown in
Table 1 (i.e., SNR, k-occurrence,...) by calling e.g., 

````
python -m mp.utils.log_processing --log_files <log_file1>
````

Note that multiple log files can be given.


### Hub Analysis
To find how many hubs / anti-hubs are in your recommendation graph before and after applying Mutual Proximity,
you can call

````
python -m mp.utils.hub_analysis --save_path <your-save-path> --hub_size 5
````


### Retrieval Accuracy

To compute the retrieval accuracy as stated in the paper, call

````
python -m mp.utils.knn_classification --save_path <your-save-path> --label_file <path-to/labels.csv>
````

Note that you have to have a file containing annotations for the data (here: genres). The format
of the file is assumed to be (per line): `{index},` followed by zero, one or multiple annotations.  

### Attacks on Defended System

In Table 1, we compare three different settings of an attack on the defended system. The first one 
still tries to minimise the KL divergence, while the other two consider Mutual Proximity in the attack.
The last version of the attack disregards the norm of the perturbation in order to find
more successful adversaries (this can be done by setting `alpha=inf` in our code). To perform these
three attacks, you can run

````
python -m mp.mp_attack.soundpark_mp_attack --cache_file <your-path/cache_file.h5py> --save_path <your-save-path> --adv-vars mp/mp_attack/config.vars
python -m mp.mp_attack.soundpark_mp_attack --cache_file <your-path/cache_file.h5py> --save_path <your-save-path> --adv-vars mp/mp_attack/mp_config.vars
python -m mp.mp_attack.soundpark_mp_attack --cache_file <your-path/cache_file.h5py> --save_path <your-save-path> --adv-vars mp/mp_attack/mp_nonorm_config.vars
````

Again, you can summarise the log files by running

````
python -m mp.utils.log_processing --log_files <log_file1>
````

with the corresponding log-files as arguments.


### Post-hoc Defense

To perform the MP defence after an attack (on an *undefended* system) was already made, you can use the 
following commands:

````
python -m mp.post_hoc.mp_defence --adv_path <adversarial-folder> --save_path <your-save-path>
python -m mp.post_hoc.mp_log_processing --log_files <log-file>
````

This makes the defence and summarises the results of the according log files. Make sure to set the paths
and log-files correctly again!


## Parameters

To summarise, the hyper-parameters we used for Table 1 are as follows:

| Adaptation                     | Parameters                                                |
| :----------------------------- | :-------------------------------------------------------- |
| original                       | &epsilon; = 0.1, &eta; = 0.001, &alpha; = 25.0            |
| C&W<sub>KL</sub><sup>mod</sup> | &epsilon; = 1.0, &eta; = 0.001, &alpha; = 25.0            | 
| C&W<sub>MP</sub><sup>mod</sup> | &epsilon; = 1.0, &eta; = 0.0005, &alpha; = 100.0          |
| C&W<sub>MP</sub><sup>mod</sup> (w/o norm)  | &epsilon; = 1.0, &eta; = 0.0005, &alpha; = -- |

Here &epsilon; is the clipping factor, &eta; the learning-rate determining the
step size during optimisation, and &alpha; the factor that balances speed of convergence
vs. perceptibility.
