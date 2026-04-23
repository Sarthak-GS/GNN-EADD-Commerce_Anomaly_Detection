# GNN-EADD Makefile
# Premium orchestration for Phase 2 submission :)

.PHONY: help setup compile train benchmark clean plot

help:
	@echo "GNN-EADD Phase 2 Commands:"
	@echo "  make setup      Install dependencies into the gnn_anomaly environment"
	@echo "  make compile    Compile OpenMP and CUDA extensions"
	@echo "  make generate   Generate the synthetic graph data"
	@echo "  make train      Run full dual-stage training with CUDA parallel mode"
	@echo "  make benchmark  Run kernel performance comparison"
	@echo "  make plot       Visualize results"
	@echo "  make clean      Remove caches and build artifacts"

setup:
	conda run -n gnn_anomaly pip install -r requirements.txt

compile:
	@echo "Compiling OpenMP extension..."
	conda run -n gnn_anomaly python kernels/build_ext.py
	@echo "Compiling Raw CUDA extension..."
	-conda run -n gnn_anomaly python kernels/setup_cuda.py install --user

generate:
	conda run -n gnn_anomaly python generate_data.py --output data/graph.pt

train:
	conda run -n gnn_anomaly python run_all.py --parallel_mode cuda --decoder_type mlp

benchmark:
	conda run -n gnn_anomaly python benchmark.py --runs 10 --n_threads 4

plot:
	conda run -n gnn_anomaly python visualize.py --graph_path data/graph.pt --results_dir results

clean:
	rm -rf __pycache__ kernels/__pycache__ models/__pycache__ utils/__pycache__
	rm -f kernels/openmp_ext.so
	rm -rf build/
	@echo "Cleaned up build artifacts."
