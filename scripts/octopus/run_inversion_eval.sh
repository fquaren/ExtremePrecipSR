python ExtremePrecipSR/src/eval_inversion.py \
    --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03 \
    --arch Baseline \
    --batch_size 64 \
    --sample_fraction 0.1 \
    --output_name rapsd_NO_SMOOTHING_analysis_results.npz

python ExtremePrecipSR/src/eval_inversion.py \
    --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59 \
    --arch Isometric \
    --batch_size 64 \
    --sample_fraction 0.1 \
    --output_name rapsd_NO_SMOOTHING_analysis_results.npz

python ExtremePrecipSR/src/eval_inversion.py \
    --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48 \
    --arch Constrained \
    --batch_size 64 \
    --sample_fraction 0.1 \
    --output_name rapsd_NO_SMOOTHING_analysis_results.npz
