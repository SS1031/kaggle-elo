from features._101_aggregate import _101_Aggregate
from features._102_aggregate_authorized import _102_AggregateAuthorized
from features._103_aggregate_rejected import _103_AggregateRejected
from features._104_aggregate_per_month import _104_AggregatePerMonth
from features._105_aggregate_authorized_per_month import _105_AggregateAuthorizedPerMonth
from features._106_aggregate_rejected_per_month import _106_AggregateRejectedPerMonth
from features._107_aggregate_per_hour import _107_AggregatePerHour
from features._108_purchase_date import _108_PurchaseDate
from features._109_special_date import _109_SpecialDate
from features._110_top3_subsector import _110_Top3Subsector
from features._201_aggregate import _201_Aggregate
from features._202_sub_aggregate import _202_SubAggregate
from features._208_purchase_date import _208_PurchaseDate
from features._209_special_date import _209_SpecialDate
from features._210_top3_subsector import _210_Top3Subsector
from features._301_train_test import _301_TrainTest
from features._302_first_active_month import _302_FirstActiveMonth
from features._401_aggregate import _401_Aggregate
from features._501_aggregate import _501_Aggregate
from features._601_new_hist import _601_NewHist
from features._602_recent_aggregate import _602_Recent5Aggregate
from features._602_recent_aggregate import _602_Recent30Aggregate
from features._701_merchant import _701_Merchant

MAPPER = {
    "_101_aggregate": _101_Aggregate,
    "_102_aggregate_authorized": _102_AggregateAuthorized,
    "_103_aggregate_rejected": _103_AggregateRejected,
    "_104_aggregate_per_month": _104_AggregatePerMonth,
    "_105_aggregate_authorized_per_month": _105_AggregateAuthorizedPerMonth,
    "_106_aggregate_rejected_per_month": _106_AggregateRejectedPerMonth,
    "_107_aggregate_per_hour": _107_AggregatePerHour,
    "_108_purchase_date": _108_PurchaseDate,
    "_109_special_date": _109_SpecialDate,
    "_110_top3_subsector": _110_Top3Subsector,
    "_201_aggregate": _201_Aggregate,
    "_202_sub_aggregate": _202_SubAggregate,
    "_208_purchase_date": _208_PurchaseDate,
    "_209_special_date": _209_SpecialDate,
    "_210_top3_subsector": _210_Top3Subsector,
    "_301_train_test": _301_TrainTest,
    "_302_first_active_month": _302_FirstActiveMonth,
    "_401_aggregate": _401_Aggregate,
    "_501_aggregate": _501_Aggregate,
    "_601_new_hist": _601_NewHist,
    "_602_recent5_aggregate": _602_Recent5Aggregate,
    "_602_recent30_aggregate": _602_Recent30Aggregate,
    "_701_merchant": _701_Merchant,
}
