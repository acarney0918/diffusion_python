This repository encases python and shell scripts for lammpstrj file analysis. Analyses include:
      - Window averaged streaming MSD python script for lammpstrj dump position files (unwrapped)
      - Window averaged VACF for lammpstrj dump velocity files
      - Bash scripts (.sh) to execute MSD and VACF calculations from python directory that reads through working trial directories and performs calculations, moving the results to new directories in base directory
          - ex. msd folder made in input directory if /my/directory/base/input/python/ and 
          - v2 folder made in input directory if /my/directory/base/input/python/

These are working python scripts used by Annabelle Carney @ UVA in Department of Chemistry (PhD candidate, 3rd year graduate student as of 8/19/2025)
ChatGPT used in correcting some of the python code while debugging.

Last updated: 11:41am 8/19/2025
