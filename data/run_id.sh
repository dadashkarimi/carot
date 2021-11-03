#!/bin/sh

python hcp_atlas_to_atlas.py -t brainnetome -s all -id True -test_size 90 -id_direction orig-ot 
python hcp_atlas_to_atlas.py -t brainnetome -s all -id True -test_size 90 -id_direction orig-orig  
python hcp_atlas_to_atlas.py -t brainnetome -s all -id True -test_size 90 -id_direction ot-orig 
#python hcp_atlas_to_atlas.py -t brainnetome -s all -id True -test_size 90 -id_direction ot-ot  &


