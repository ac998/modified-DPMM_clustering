### Instructions to run
Run `python setup_utils.py build_ext --inplace` first build dependent cython modules (should generate a `.so` file in *nix or a 
`.pyd` file in Windows).  
Following is a brief description of driver files:
- `main-demo.py` - checking out clustering on vehicle trajectories and points on 2D cartesian plane.
- `main-object_clustering.py` - check it out on video frames.
- `SOTA_comparison.py` - Used other state of the art clustering algorithms on the same datasets to compare performance. Check results folder.

#### To do
Add  references for the SOTA alogorithms and code used to compare computation times. 
