Check the documentation.pdf to see what this is about!

make sure you have packages:

numpy
opencv (3)!
matplotlib
scipy
mpl_toolkits
math
os


then just go and run: python get_positions.py
f.py contains all the helper functions doing the dirty work.

I work on an Ubuntu 15.10


The script:

Reads in the images
Findes the keypoints
matches the keypoints
estimates the Rotation and Translation
Visualises the result

creates an ./output/ folder with the results


OUTPUT:
./kps/          - the keypoints found in the images
./matches/      - the visualisation of the flow of the keypoints between consecutive images
./result/       - the 2D mapping and the 3D- coordinate plot
./logfile.txt   - the indivdicual rotation and translation of each step and information if the sanity check was negative.



