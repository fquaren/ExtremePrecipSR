python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
  --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03 \
  --arch Baseline \
  --target_idx 447523 \

python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
  --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59 \
  --arch Isometric \
  --target_idx 447523 \

python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
  --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48 \
  --arch Constrained \
  --target_idx 447523 \

python /home/fquareng/work/ExtremePrecipSR/src/plot_inversion_comparison.py \
  --files \
    /home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_447523_Baseline_0.npz \
    /home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_447523_Isometric_0.npz \
    /home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_447523_Constrained_0.npz \
  --output /home/fquareng/work/figures/comparison_NO_SMOOTHING_447523.pdf \
  --quantile_levels 0.01 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.99

# python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
#   --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03 \
#   --arch Baseline \
#   --target_idx 367064 \

# python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
#   --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59 \
#   --arch Isometric \
#   --target_idx 367064 \

# python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
#   --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48 \
#   --arch Constrained \
#   --target_idx 367064 \

# python /home/fquareng/work/ExtremePrecipSR/src/plot_inversion_comparison.py \
#   --files \
#     /home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_367064_Baseline_0.npz \
#     /home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_367064_Isometric_0.npz \
#     /home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_367064_Constrained_0.npz \
#   --output /home/fquareng/work/figures/comparison_NO_SMOOTHING_367064.pdf \
#   --quantile_levels 0.01 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.99

# python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
#   --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03 \
#   --arch Baseline \
#   --target_idx 196425 \

# python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
#   --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59 \
#   --arch Isometric \
#   --target_idx 196425 \

# python /home/fquareng/work/ExtremePrecipSR/src/inversion_test.py \
#   --run_dir /home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48 \
#   --arch Constrained \
#   --target_idx 196425 \

# python /home/fquareng/work/ExtremePrecipSR/src/plot_inversion_comparison.py \
#   --files \
#     /home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_196425_Baseline_0.npz \
#     /home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_196425_Isometric_0.npz \
#     /home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48/inversion_test/data/inversion_data_NO_SMOOTHING_real_data_196425_Constrained_0.npz \
#   --output /home/fquareng/work/figures/comparison_NO_SMOOTHING_196425.pdf \
#   --quantile_levels 0.01 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.99