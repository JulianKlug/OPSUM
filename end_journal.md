# Early neurological deterioration forecasting

## Finding base model and metrics

| Start Date |End Date  |
|------------|----------|
| 2024-08-01 ||

### Metrics
- 25.09.24: AUROC might be be unsuited because of very rare events (XGB fits to predicting 0 all the time)
- but: when always predicting 0 -> AUROC = 0.5

TODO: try AUPRC, MCC

### Finetuning transformer Encoder

- 25.09.24: not too bad of a model (a lot of over prediction before the event)
  - at 6h: AUROC 0.73; AUPRC 0.019; MCC 0.028

### XGB

- 25.09.24: model seems useless, overfitting towards predicting 0

TODO: reframe training to be more like actual problem?


### Transformer Encoder-Decoder

- 25.09.24: model training done

TODO: test inference / evaluation 