

# Cross Atlas Remapping via Optimal Transport (CAROT)

This is the repository for CAROT, ross Atlas Remapping via Optimal Transport. All the data including `mappings`, `intrinsic evluation`, and `downstream analysis` are in `data/` folder.


## Creating mappings 
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


```console
javid@ycs:~$ python hcp_atlas_to_atlas.py -s brainnetome -t shen -task rest1 --save_model True -c euclidean
```


## Connectome Correlation and Downstream Analysis 
