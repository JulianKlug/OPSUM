import numpy
import torch as ch

# test data with X of shape (1, 6, 3) and Y of shape (1) and value 1
x = ch.randn(1, 6, 3)
y = ch.tensor([1])

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer

model = OPSUMTransformer(
            input_dim=3,
            num_layers=2,
            model_dim=8,
            dropout=0.1,
            ff_dim=16,
            num_heads=2,
            num_classes=1,
            max_dim=500,
            pos_encode_factor=1,
    feature_aggregration=True
        )

y_hat = model(x)
print(y_hat.shape)