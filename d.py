import h5py

# Open an H5 file
with h5py.File('waste_classifier_best.h5', 'r') as f:
    print("Keys:", list(f.keys()))  # top-level groups
    model_weights = f['model_weights']
    print("Subgroups:", list(model_weights.keys()))
