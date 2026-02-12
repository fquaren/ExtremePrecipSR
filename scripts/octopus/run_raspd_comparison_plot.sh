# python /home/fquareng/work/ExtremePrecipSR/src/plot_rapsd_comparison.py \
#   --baseline "/home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03/inversion_test/rapsd_analysis_results.npz" \
#   --isometric "/home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59/inversion_test/rapsd_analysis_results.npz" \
#   --constrained "/home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48/inversion_test/rapsd_analysis_results.npz" \
#   --output "/home/fquareng/work/figures/comparison_raspd.pdf"

python /home/fquareng/work/ExtremePrecipSR/src/plot_rapsd_comparison.py \
  --baseline "/home/fquareng/work/experiment_runs/GammaEmulator_v6_Baseline_SingleRun_2026-01-14_08-50-03/inversion_test/rapsd_NO_SMOOTHING_analysis_results.npz" \
  --isometric "/home/fquareng/work/experiment_runs/GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59/inversion_test/rapsd_NO_SMOOTHING_analysis_results.npz" \
  --constrained "/home/fquareng/work/experiment_runs/GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48/inversion_test/rapsd_NO_SMOOTHING_analysis_results.npz" \
  --output "/home/fquareng/work/figures/comparison_NO_SMOOTHING_raspd.pdf"
