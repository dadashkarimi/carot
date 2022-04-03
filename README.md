

# Cross Atlas Remapping via Optimal Transport (CAROT)

This is the repository for CAROT, ross Atlas Remapping via Optimal Transport. All the data including `mappings`, `intrinsic evluation`, and `downstream analysis` are in `data/` folder.


![alt text](figs/ot.png)

CAROT uses optimal transport theory, or the mathematics of converting a probability distribution from one set to another, to find an optimal mapping
between two atlases that allows data processed from one atlas to be directly transformed into a connectome based on an
unavailable atlas without needing raw data. CAROT is designed for functional connectomes based on functional magnetic
imaging (fMRI) data. 

## Website: http://carotproject.com
We are happy to announce that we also launched carot website for demo: http://carotproject.com. In rare cases if your wifi can't recognize the name check out the exact ip address: http://34.238.128.124

## Requirements
The main packages we have used to run the carot pipelines includes:
1. `conda install -c conda-forge pot`
2. `conda install -c conda-forge matplotlib`
3. `conda install -c anaconda scikit-learn`
4. `conda install -c anaconda h5py`
5. `conda install -c anaconda scipy`
6. `conda install -c conda-forge argparse`
7. `conda install pandas`
8. `pip install numpy`

## Properties file
To specify the location of different files you need to change `config.properties`:
```console
[coord]
shen=/data_dustin/store4/Templates/shen_coords.csv
craddock=/data_dustin/store4/Templates/craddock_coords.csv
power=/data_dustin/store4/Templates/power_coords.csv
[path]
shen=/data_dustin/store4/Templates/HCP/shen/rest1.mat
craddock=/data_dustin/store4/Templates/HCP/craddock/rest1.mat
power=/data_dustin/store4/Templates/HCP/power/rest1.mat

```

## Creating mappings and validating connectomes
These are the arguments we use for running our scripts:

1. `-s` or `--source`: source atlas
2. `-t` or `--target`: target atlas
3. `-c` or `--c`: cost matrix (euclidean or functional)
4. `-task` or `--task`: task ("rest1","gambling","wm","motor","lang","social","relational","emotion")
5. `-id` or `--id`: id rate (True or False)
6. `-id_direction` or `--id_direction`: (ot-ot or orig-orig)
7. `-intrinsic` or `--intrinsic`: parameter sensitiviy (True or False)
8. `-simplex` or `--simplex`: (1: simplex ot, 2: average ot, 3: stacking ot, default is 2)
9. `-num_iters` or `--num_iters`: number of iterations in test
10. `-save_model` or `--save_model`: (True or False)


## 1. Building cost matrix
Here, we want to calculate cost matrix between different ROIs in two atlases. Then, we have to specify the names of two atlases with `-s` and `-t` and the task we want to learn mappings with `-task` .
```console
python build_cost_matrix.py -s craddock -t shen
```
The output will be stored in `cost_source_target.csv` with `n` rows and`m` columns indicating number of ROIs in source and target respectively. 

## 2. Finding mappings
Now, we can specify two atlases and the cost matrix derived from previous step to obtain optimal transport mapping between these two. 
```console
python build_mapping.py -s craddock -t shen -c cost_craddock_shen.csv
```
The output will be stored in `T_source_target.csv` with `n` rows and`m` columns indicating number of ROIs in source and target respectively. Each row is a probability distribution exhibiting optimum assignment of values from the appropriate node to target nodes.  

## 3. Carot: Transforming source(s) to a target atlas
Given cost matrix `cost_source_target.csv` and mapping `T_source_target.csv` now we can transfer source parcellation into other:
```console
python carot.py -s craddock -t shen -m T_source_target.csv
``````
## Replicating results in the paper
To run a simple script with source `brainnetome` and target `shen` using `rest1` with `euclidean` cost measure, and saving it: 
```console
python hcp_atlas_to_atlas.py -s brainnetome -t shen -task rest1 --save_model True -c euclidean
```
 
To run the main CAROT pipeline with `all` available atlases into `shen`:
```console
python hcp_atlas_to_atlas.py -s all -t shen -task rest1 -simplex 2 -sample_atlas 0
```

To run identification pipeline between estimated connectomes and databases `rest`` and `rest2` in HCP dataset:
```console
python hcp_atlas_to_atlas.py -t power -s all -id True  -id_direction orig-ot
```

To run parameter sensitivity to study different frame/train sizes:
```console
python hcp_atlas_to_atlas.py -s brainnetome -t power -task all --intrinsic true
```

## Sex classification on MDD dataset
To train a classification model on `PNC` dataset and test on `MDD`we need to use script `pnc_atlas_to_atlas.py `:


1. `-s` or `--source`: source atlas
2. `-t` or `--target`: target atlas
3. `-database` or `--database`: database (UCLA, PNC)
4. `-g` or `--g`: mapping trained on HCP (rest1 or mean)
5. `-sex_task` or `--sex_task`: which task we are training (rest1, nback, etc)
6. `-num_iters` or `--num_iters`: number of iterations to train
7. `-label` or `--label`: which label to train (sex, iq)
8. `-site` or `--site`: which site we are testing (1,2,3,..,24)


```console
python pnc_atlas_to_atlas.py -s craddock -t shen -database ucla -sex_task 2 -g mean -model reg -num_iters 100 -label sex -site 1
 ```

