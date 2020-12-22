Changed eTraGo files within this folder in relation to dev:

appl_Lin0/1.py:
- Settings in args for applying linkage method, to be used with cluster/snapshot_Lin[0/1].py

appl_Seg0/1.py:
- Includes the innovative changes implemented within the thesis.
	n_clusters must be set to 1,
	storage_constraints to '',
	'segmentation' to number of segments applied within clustering.
- Needs to be applied with the respective snapshot-file: cluster/snapshot_Seg[0/1].py

To use the respective snapshot-files, change its name to 'snapshot.py' so that eTraGo uses it. 