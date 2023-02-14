# Hyperparameters used in litterature


#### Set Functions for Time Series, Horn et al., 2020

| Parameter            | P-Mortality | M-Mortality | P-Sepsis | Range         |
|----------------------|-------------|-------------|----------|---------------|
| optimizer            | Adam        | Adam        | Adam     | Adam          |
| learning rate        | 0.00567     | 0.00204     | 0.00027  | 0.0001 - 0.01 |
| batch size           | 256         | 256         | 128      | 128 - 256     |
| warmup steps         | 1000        | 1000        | 1000     | 1000          |
| n dims (mlp)         | 512         | 512         | 128      | 128 - 512     |
| n mlp layers         | 1           | 1           | 1        | 1             |
| n atn heads          | 2           | 8           | 2        | 2 - 8         |
| atn head size        | ?           | ?           | ?        | ?             |
| n transformer layers | 1           | 2           | 4        | 1 - 4         |
| dropout              | 0.3         | 0.4         | 0.1      | 0.1 - 0.4     |
| attn dropout         | 0.3         | 0.0         | 0.4      | 0.0 - 0.4     |
| aggregation fn       | max         | mean        | mean     | max, mean     |
| max timescale        | 1000.0      | 100.0       | 100.0    | 100.0 - 100.0 |

Missing parameters:
- atn_feed_forward_dim

Notes: 
- A warmup scheduler is used
- "dropout" is not applied to MLP, but somewhere inside the transformer block on top of the attention dropout