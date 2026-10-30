# DMP ball-catching results

These artifacts document the separate CSC2626-derived UR10 ball-catching exercise. They are not results from the Robomimic square-nut task.

## Robot execution

[Watch the simulated ball-catching rollout](ballcatch.mp4).

The [saved aggregate evaluation](ballcatch_metrics.csv) records **40% success** and **-17.76 average episode reward**. The evaluation entry point in `../main.py` calls `test_policy` with 20 episodes and seed 1. The CSV itself stores only aggregate reward and success; it does not include an episode count, per-episode outcomes, or a full run manifest. No multi-seed reliability claim is made.

## Trajectory fitting

The [tuning table](tuning.csv) evaluates 20 combinations of basis-function count and spring gain. The smallest recorded reconstruction RMSE is **0.06804 radians**, using **45 basis functions and gain 25**. This measures reconstruction on the demonstrated trajectories used in tuning, not held-out generalization. Reconstruction accuracy and catching success are different measures.

The six figures compare the first demonstrated trajectory with its DMP reconstruction, one joint per plot. This reconstruction exercise uses the default DMP settings, separately from the full-dataset tuning sweep.

| Joint | Reconstruction |
|---|---|
| 1 | [Plot](recons1.png) |
| 2 | [Plot](recons2.png) |
| 3 | [Plot](recons3.png) |
| 4 | [Plot](recons4.png) |
| 5 | [Plot](recons5.png) |
| 6 | [Plot](recons6.png) |

![Joint 1 demonstration and reconstruction](recons1.png)

The small `trained_dmp.pkl` file is retained as the existing fitted DMP artifact. Large neural-network checkpoints are excluded separately.
