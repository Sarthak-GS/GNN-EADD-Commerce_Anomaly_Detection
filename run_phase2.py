"""
run_phase2.py — Master script for GNN-EADD Phase 2 pipeline:
  1. Build CUDA kernels (if not already built)
  2. Run kernel correctness tests
  3. Run performance benchmarks (Sequential vs OpenMP vs CUDA)
  4. Train on graph data with CUDA-accelerated operations
  5. Compare against PyG baseline
  6. Generate all plots and reports
"""

import sys, os, subprocess, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from utils.utils import set_seed, compute_metrics, build_edge_indices


def check_cuda_kernels():
    """Check if custom CUDA kernels are compiled and importable."""
    try:
        import gnn_cuda_kernels
        print("[✓] CUDA kernels already installed")
        return True
    except ImportError:
        print("[!] CUDA kernels not found. Building...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', '.'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[✗] Build failed:\n{result.stderr[-500:]}")
            return False
        try:
            import gnn_cuda_kernels
            print("[✓] CUDA kernels built and installed")
            return True
        except ImportError:
            print("[✗] CUDA kernels still not importable after build")
            return False


def test_kernel_correctness():
    """Verify CUDA kernels produce correct results vs PyTorch reference."""
    import gnn_cuda_kernels
    device = torch.device('cuda')
    print("\n" + "="*60)
    print("  KERNEL CORRECTNESS TESTS")
    print("="*60)

    N, E, D = 500, 2000, 64
    torch.manual_seed(42)

    scores = torch.randn(N, device=device)
    row = torch.randint(0, N, (E,), device=device)
    col = torch.randint(0, N, (E,), device=device)

    # Test 1: Smoothness Loss
    cuda_loss = gnn_cuda_kernels.smoothness_loss(scores, row, col).item()
    ref_loss = ((scores[row] - scores[col])**2).mean().item()
    err1 = abs(cuda_loss - ref_loss)
    print(f"  [1] Smoothness Loss  | CUDA={cuda_loss:.6f} Ref={ref_loss:.6f} Err={err1:.2e} {'✓' if err1 < 1e-3 else '✗'}")

    # Test 2: GAT Attention
    Wh = torch.randn(N, D, device=device)
    attn_vec = torch.randn(2*D, device=device)
    cuda_attn = gnn_cuda_kernels.gat_attention(Wh, row, col, attn_vec, 0.2)
    # Reference
    ref_attn = torch.zeros(E, device=device)
    for e in range(E):
        val = (Wh[col[e]] * attn_vec[:D]).sum() + (Wh[row[e]] * attn_vec[D:]).sum()
        ref_attn[e] = val if val > 0 else val * 0.2
    err2 = (cuda_attn - ref_attn).abs().max().item()
    print(f"  [2] GAT Attention    | MaxErr={err2:.2e} {'✓' if err2 < 1e-4 else '✗'}")

    # Test 3: Neighbor Aggregation
    alpha = torch.rand(E, device=device)
    cuda_agg = gnn_cuda_kernels.neighbor_aggregation(Wh, alpha, row, col, N)
    ref_agg = torch.zeros(N, D, device=device)
    for e in range(E):
        ref_agg[col[e]] += alpha[e] * Wh[row[e]]
    err3 = (cuda_agg - ref_agg).abs().max().item()
    print(f"  [3] Neighbor Agg     | MaxErr={err3:.2e} {'✓' if err3 < 1e-3 else '✗'}")

    # Test 4: Tiled MatMul
    A = torch.randn(128, 64, device=device)
    B = torch.randn(64, 32, device=device)
    cuda_C = gnn_cuda_kernels.tiled_matmul(A, B)
    ref_C = A @ B
    err4 = (cuda_C - ref_C).abs().max().item()
    print(f"  [4] Tiled MatMul     | MaxErr={err4:.2e} {'✓' if err4 < 1e-3 else '✗'}")

    all_pass = all(e < 1e-3 for e in [err1, err2, err3, err4])
    print(f"\n  {'All tests PASSED ✓' if all_pass else 'Some tests FAILED ✗'}")
    return all_pass


def run_training_with_cuda(args):
    """Run the standard Phase 1 training pipeline (it auto-uses CUDA if available)."""
    from train import run_training
    return run_training(args)


def main():
    p = argparse.ArgumentParser(description="GNN-EADD Phase 2 Master Script")
    p.add_argument('--graph_path',     type=str, default='data/graph.pt')
    p.add_argument('--gae_epochs',     type=int, default=200)
    p.add_argument('--gat_epochs',     type=int, default=100)
    p.add_argument('--embed_dim',      type=int, default=128)
    p.add_argument('--hidden_dim',     type=int, default=64)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--lam',            type=float, default=0.5)
    p.add_argument('--seed',           type=int, default=42)
    p.add_argument('--results_dir',    type=str, default='results')
    p.add_argument('--skip_baselines', action='store_true')
    p.add_argument('--skip_benchmark', action='store_true')
    p.add_argument('--skip_training',  action='store_true')
    p.add_argument('--include_xlarge', action='store_true')
    args = p.parse_args()

    print("█" * 64)
    print("  GNN-EADD PHASE 2 — CUDA-Accelerated Pipeline")
    print("█" * 64)

    # Step 1: Build CUDA kernels
    has_cuda = check_cuda_kernels()

    # Step 2: Test correctness
    if has_cuda:
        test_kernel_correctness()

    # Step 3: Run benchmarks
    if not args.skip_benchmark:
        print("\n" + "="*60)
        print("  PERFORMANCE BENCHMARKS")
        print("="*60)
        from benchmark import run_benchmark
        bench_args = SimpleNamespace(
            results_dir=args.results_dir,
            include_xlarge=args.include_xlarge,
        )
        run_benchmark(bench_args)

    # Step 4: Train model
    if not args.skip_training:
        print("\n" + "="*60)
        print("  MODEL TRAINING")
        print("="*60)
        gae, gat, Z, A_per_type, graph, metrics = run_training_with_cuda(args)

        # Step 5: Visualization
        from visualize import make_all_plots
        ckpt = torch.load(Path(args.results_dir) / 'checkpoint.pt',
                         map_location='cpu', weights_only=False)
        N_P = graph.num_nodes_per_type['product']
        N_U = graph.num_nodes_per_type['user']
        N_S = graph.num_nodes_per_type['seller']
        node_types_np = np.array([0]*N_P + [1]*N_U + [2]*N_S)

        gat.eval()
        with torch.no_grad():
            scores_np = gat(Z, A_per_type).cpu().numpy()
        labels_np = graph.y.numpy()

        make_all_plots(
            ckpt['gae_losses'], ckpt['gat_history'],
            scores_np, labels_np, node_types_np, graph
        )

    # Step 6: PyG Baseline
    if not args.skip_baselines:
        from baseline_comparison import run_baseline_comparison
        run_baseline_comparison(args.graph_path)

    # Final summary
    print("\n" + "█"*64)
    print("  GNN-EADD PHASE 2 — COMPLETE")
    print("█"*64)
    
    if not args.skip_baselines and not args.skip_training:
        # Load the saved metrics to print a combined runtime comparison table
        pyg_path = Path(args.results_dir) / 'baseline_pyg_metrics.npy'
        ckpt_path = Path(args.results_dir) / 'checkpoint.pt'
        if pyg_path.exists() and ckpt_path.exists():
            metrics_pyg = np.load(pyg_path, allow_pickle=True).item()
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            our_times = ckpt.get('times', {'gae': 0.0, 'gat': 0.0})
            
            print(f"\n  ┌─────────────────────┬───────────┬───────────┐")
            print(f"  │ Runtime Comparison  │ Ours (s)  │ PyG (s)   │")
            print(f"  ├─────────────────────┼───────────┼───────────┤")
            print(f"  │ Stage 1 (GAE)       │ {our_times.get('gae', 0.0):>9.3f} │ {metrics_pyg.get('gae_time', 0.0):>9.3f} │")
            print(f"  │ Stage 2 (GAT)       │ {our_times.get('gat', 0.0):>9.3f} │ {metrics_pyg.get('gat_time', 0.0):>9.3f} │")
            print(f"  ├─────────────────────┼───────────┼───────────┤")
            print(f"  │ Total Time          │ {(our_times.get('gae', 0.0) + our_times.get('gat', 0.0)):>9.3f} │ {(metrics_pyg.get('gae_time', 0.0) + metrics_pyg.get('gat_time', 0.0)):>9.3f} │")
            print(f"  └─────────────────────┴───────────┴───────────┘\n")

    print(f"  Results directory: {args.results_dir}/")
    print(f"  Files generated:")
    results_dir = Path(args.results_dir)
    for f in sorted(results_dir.glob('*')):
        print(f"    • {f.name} ({f.stat().st_size:,} bytes)")
    print("█"*64 + "\n")


if __name__ == '__main__':
    main()
