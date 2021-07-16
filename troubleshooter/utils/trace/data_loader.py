# -*- coding: utf-8 -*-

import os
import time

import click
import pandas as pd
from elasticsearch import Elasticsearch
from loguru import logger
from tqdm import tqdm

ip_address = "*******"
user = "******"
pwd = "******"

history_time_range = 5 * 60  # s
now_time_range = 2 * 60


def timestamp2date(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def date2timestamp(format_time):
    return time.mktime(time.strptime(format_time, "%Y-%m-%d %H:%M:%S"))


def process_hits(hits):
    def get_pid(_references):
        for item in _references:
            if item["refType"] == "CHILD_OF":
                return item["spanID"]
        else:
            return None

    def get_tag(_tags, _key):
        for item in _tags:
            if item["key"] == _key:
                return item["value"]
        else:
            return None

    def parse(_item):
        _item = _item["_source"]
        _ret = {
            "trace_id": _item["traceID"],
            "timestamp": _item["startTimeMillis"],
            "latency": _item["duration"] / 1e3,
            "parent_span_id": get_pid(_item["references"]),
            "span_id": _item["spanID"],
            "service_name": _item["operationName"],
            "status_code": get_tag(_item["tags"], "status.code"),
        }
        # success: _ret["status_code"] == "0" || _ret["status_code"].lower() == "ok"
        if _ret["status_code"] is None:
            _ret["status_code"] = get_tag(_item["tags"], "otel.status_code")
        if _ret["status_code"] is None:
            _ret["status_code"] = get_tag(_item["tags"], "grpc.status_code")

        return _ret

    try:
        return list(map(parse, hits))
    except KeyError as e:
        logger.error(f"error: {e}")


def dump_es_data(anomaly_time, es_index):
    anomaly_timestamp = date2timestamp(anomaly_time)
    min_time = anomaly_timestamp - history_time_range
    max_time = anomaly_timestamp + now_time_range

    es = Elasticsearch(hosts=ip_address, http_auth=(user, pwd), timeout=30)

    body = {
        "query": {"bool": {"must": [], "must_not": [], "should": []}},
        "_source": {"includes": [], "excludes": []},
        "sort": ["_doc"],
        "aggs": {},
    }

    time_range_body = {"range": {"startTimeMillis": {}}}
    time_range_body["range"]["startTimeMillis"]["gte"] = int(min_time * 1000)  # ms
    time_range_body["range"]["startTimeMillis"]["lte"] = int(max_time * 1000)
    body["query"]["bool"]["must"].append(time_range_body)

    es_keys = [
        "traceID",
        "spanID",
        "references",
        "startTimeMillis",
        "duration",
        "tags",
        "process",
        "operationName",
    ]
    body["_source"]["includes"].extend(es_keys)

    logger.info(f"es search body: {body}")

    rsp = None
    while rsp is None:
        try:
            rsp = es.search(
                index=es_index, body=dict(**body, size=10000), scroll="2m", timeout="2m"
            )
        except Exception as e:
            logger.error(f"Exception in search: {e}. Retry")
            rsp = None

    total = rsp["hits"]["total"]["value"]
    # logger.info(f"total data size: {rsp['hits']['total']['value']}")

    sid = rsp["_scroll_id"]
    data = rsp["hits"]["hits"]
    # logger.info(f"es data sample: {data[0]}")
    data = process_hits(data)
    scroll_size = len(rsp["hits"]["hits"])
    total -= len(data)

    with tqdm(
        total=total,
        desc=f"pulling data from {timestamp2date(min_time)} to {timestamp2date(max_time)}",
    ) as pbar:
        while scroll_size > 0:
            try:
                _rsp = es.scroll(scroll_id=sid, scroll="2m")
            except Exception as e:
                logger.error(f"Exception in scroll: {e}. Retry")
                continue
            data.extend(process_hits(_rsp["hits"]["hits"]))
            sid = _rsp["_scroll_id"]
            scroll_size = len(_rsp["hits"]["hits"])

            pbar.update(scroll_size)

    return data


def process_data(data, anomaly_time):
    df = pd.DataFrame.from_records(data=data).drop_duplicates(subset=["span_id"])

    anomaly_timestamp = int(date2timestamp(anomaly_time) * 1000)  # ms

    span_id_service_map = dict(zip(df["span_id"], df["service_name"]))
    df["dst"] = df["service_name"]
    df["src"] = df["parent_span_id"].map(span_id_service_map)
    df = df.drop(["parent_span_id"], axis=1)
    df = df.dropna()

    df = df.astype(
        {
            "latency": float,
            "status_code": str,
            "dst": str,
            "src": str,
            "trace_id": str,
            "span_id": str,
        }
    )

    def is_success(_status_code):
        """ status code 0:成功 1:不成功 """
        if _status_code == "0" or _status_code.lower() == "ok":
            return 1.0
        else:
            return 0.0

    df["success"] = df["status_code"].apply(is_success)
    # print(df.size, df.loc[:, 'status_code'].value_counts())
    print(df.isna().sum())

    # init statistic
    train_trace_ids = set(df[df["timestamp"] <= anomaly_timestamp]["trace_id"])
    test_trace_ids = set(df[df["timestamp"] > anomaly_timestamp]["trace_id"])
    logger.info(f"train trace size: {len(train_trace_ids)}")
    logger.info(f"test trace size: {len(test_trace_ids)}")

    df = df[
        [
            "timestamp",
            "trace_id",
            "src",
            "dst",
            "span_id",
            "service_name",
            "status_code",
            "success",
            "latency",
        ]
    ]

    return df


@click.command()
@click.option("--anomaly_time", "-at", "anomaly_time", help="anomaly time:yyyy-mm-dd hh:mm:ss")
@click.option("--root_cause", "-rc", "root_cause", help="root cause")
def main(anomaly_time, root_cause):
    es_index = f"jaeger-span-{anomaly_time.split()[0]}"
    output_dir = "/Users/shenmengjia/Datasets/testbed"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Exception: {e}")

    data = dump_es_data(anomaly_time, es_index)
    logger.info(f"data size: {len(data)}")

    logger.info("processing data...")
    df = process_data(data, anomaly_time)
    logger.info(f"data size: {len(df)}")

    df.to_csv(
        f"{output_dir}/{es_index}-{root_cause}-{int(date2timestamp(anomaly_time)*1000)}.csv",
        index=False,
    )
    logger.info("data processing completed.")


if __name__ == "__main__":
    # anomaly_time = "2020-11-25 17:10:00"
    # root_cause = "paymentservice"
    main()
