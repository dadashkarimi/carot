

# Cross Atlas Remapping via Optimal Transport (CAROT)

This is the repository for CAROT, ross Atlas Remapping via Optimal Transport. All the data including `mappings`, `intrinsic evluation`, and `downstream analysis` are in `data/` folder.


![alt text](figs/ot.jpg)

CAROT uses optimal transport theory, or the mathematics of converting a probability distribution from one set to another, to find an optimal mapping
between two atlases that allows data processed from one atlas to be directly transformed into a connectome based on an
unavailable atlas without needing raw data. CAROT is designed for functional connectomes based on functional magnetic
imaging (fMRI) data. 

## Creating mappings and validating connectomes
The main code to create mappings, connectomes in a desired atlas, and testing is available in `hcp_atlas_to_atlas.py`.
The following arguments are needed to call the script:

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

