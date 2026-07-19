# Calibration run — summary

- **run**: `20260719-think-10ep`
- **speculators ref**: `9b74129`  ·  **windtunnel ref**: `8934ea0`
- **data**: 2000 convs → regen `?` rows, train max_samples `6000`, epochs `10`
- **loss**: `{"ce":0.1,"tv":0.9}`  ·  GPUs `regen=2,3,6,7 serve=2,3 train=6,7`

## Timing (seconds)

| phase | seconds |
|---|--:|
| regen (shared) | — |
| prepare (shared) | — |
| train · eagle3 | 4829 |
| train · dflash | 13332 |
| train · dspark | 166 |

## Results

| lane | status | EAL (best) | val loss (best) | best ep | peak mem (MiB) | train s |
|---|---|--:|--:|--:|--:|--:|
| eagle3 | ok | 1.188 | 2.4426 | 7 | 6:27979, 7:27979 | 4829 |
| dflash | ok | 1.159 | 0.4707 | 5 | 6:39669, 7:39669 | 13332 |
| dspark | no_val_metrics | — | — | — | 6:38187, 7:38187 | 166 |

## Per-epoch EAL / loss

### eagle3

| epoch | EAL | loss |
|--:|--:|--:|
| 0 | 0.876 | 2.9311 |
| 1 | 0.982 | 2.7233 |
| 2 | 1.051 | 2.6209 |
| 3 | 1.094 | 2.5556 |
| 4 | 1.131 | 2.5049 |
| 5 | 1.156 | 2.4721 |
| 6 | 1.176 | 2.4542 |
| 7 | 1.188 | 2.4426 |

### dflash

| epoch | EAL | loss |
|--:|--:|--:|
| 0 | 0.781 | 0.5715 |
| 1 | 0.945 | 0.5217 |
| 2 | 1.038 | 0.4974 |
| 3 | 1.092 | 0.4831 |
| 4 | 1.137 | 0.4740 |
| 5 | 1.159 | 0.4707 |

### dspark

| epoch | EAL | loss |
|--:|--:|--:|

