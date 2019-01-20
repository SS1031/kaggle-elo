from features._101_aggregate import _101_Aggregate
from features._102_aggregate_authorized import _102_AggregateAuthorized
from features._103_aggregate_rejected import _103_AggregateRejected
from features._104_aggregate_per_month import _104_AggregatePerMonth
from features._105_aggregate_authorized_per_month import _105_AggregateAuthorizedPerMonth
from features._106_aggregate_rejected_per_month import _106_AggregateRejectedPerMonth
from features._201_aggregate import _201_Aggregate
from features._202_sub_aggregate import _202_SubAggregate
from features._301_train_test import _301_TrainTest

MAPPER = {
    "_101_aggregate": _101_Aggregate,
    "_102_aggregate_authorized": _102_AggregateAuthorized,
    "_103_aggregate_rejected": _103_AggregateRejected,
    "_104_aggregate_per_month": _104_AggregatePerMonth,
    "_105_aggregate_authorized_per_month": _105_AggregateAuthorizedPerMonth,
    "_106_aggregate_rejected_per_month": _106_AggregateRejectedPerMonth,
    "_201_aggregate": _201_Aggregate,
    "_202_sub_aggregate": _202_SubAggregate,
    "_301_train_test": _301_TrainTest,
}
