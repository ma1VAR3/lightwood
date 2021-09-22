from copy import deepcopy
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import numpy as np
import pandas as pd

from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.general import evaluate_accuracy
from lightwood.analysis.nc.util import t_softmax


class GlobalFeatureImportance(BaseAnalysisBlock):
    def __init__(self):
        super().__init__(deps=None)

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)
        empty_input_accuracy = {}
        ignorable_input_cols = [x for x in ns.input_cols if (not ns.ts_cfg.is_timeseries or
                                                             (x not in ns.ts_cfg.order_by and
                                                              x not in ns.ts_cfg.historical_columns))]
        for col in ignorable_input_cols:
            partial_data = deepcopy(ns.encoded_val_data)
            partial_data.clear_cache()
            for ds in partial_data.encoded_ds_arr:
                ds.data_frame[col] = [None] * len(ds.data_frame[col])

            if not ns.is_classification:
                empty_input_preds = ns.predictor(partial_data)
            else:
                empty_input_preds = ns.predictor(partial_data, predict_proba=True)

            empty_input_accuracy[col] = np.mean(list(evaluate_accuracy(
                ns.data,
                empty_input_preds['prediction'],
                ns.target,
                ns.accuracy_functions
            ).values()))

        column_importances = {}
        acc_increases = []
        for col in ignorable_input_cols:
            accuracy_increase = (ns.normal_accuracy - empty_input_accuracy[col])
            acc_increases.append(accuracy_increase)

        # low 0.2 temperature to accentuate differences
        acc_increases = t_softmax(torch.Tensor([acc_increases]), t=0.2).tolist()[0]
        for col, inc in zip(ignorable_input_cols, acc_increases):
            column_importances[col] = 10 * inc  # scores go from 0 to 10 in GUI

        info['column_importances'] = column_importances
        self.is_prepared = True
        return info

    def explain(self, insights: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        # does nothing on inference
        return insights, {}
