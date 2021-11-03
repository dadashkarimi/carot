#!/bin/sh

#python hcp_atlas_to_atlas.py -s shen -t craddock -task all 
python hcp_atlas_to_atlas.py -s shen -t craddock -task wm -cross_task gambling &
python hcp_atlas_to_atlas.py -s shen -t craddock -task gambling -cross_task wm &

python hcp_atlas_to_atlas.py -t shen -s craddock -task wm -cross_task gambling &
python hcp_atlas_to_atlas.py -t shen -s craddock -task gambling -cross_task wm &
#python hcp_atlas_to_atlas.py -t shen -s craddock -task wm -cross_task wm &
#python hcp_atlas_to_atlas.py -t shen -s craddock -task wm -cross_task wm &

#python hcp_atlas_to_atlas.py -s shen -t brainnetome -task wm  &
#python hcp_atlas_to_atlas.py -s shen -t brainnetome -task wm &
#python hcp_atlas_to_atlas.py -t shen -s brainnetome -task wm &
#python hcp_atlas_to_atlas.py -t shen -s brainnetome -task wm &
#python hcp_atlas_to_atlas.py -t schaefer -s brainnetome -task all &

