#!/bin/bash
conda activate dfd-fcg
cd /data1/pipesk/pop_work/gnn_eadd

echo "Generating 20,000 node graph (Max capacity for dense PyTorch baseline)..."
python generate_data.py --n_products 15000 --n_users 6000 --n_sellers 1500

echo "Running Phase 2 on bigger graph..."
python -u run_phase2.py --skip_benchmark --skip_baselines > phase2_bigger.log 2>&1
echo "Done!"
