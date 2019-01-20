from features._101_aggregate import _101_Aggregate
from features._102_aggregate_authorized import _102_AggregateAuthorized
from features._103_aggregate_rejected import _103_AggregateRejected


MAPPER = {
    "_101_aggregate": _101_Aggregate,
    "_102_aggregate_authorized": _102_AggregateAuthorized,
    "_103_aggregate_rejected": _103_AggregateRejected,
}