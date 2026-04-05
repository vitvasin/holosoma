import sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2

ds = datastore.DataStore()
ds.open_for_scan(sys.argv[1])

for count in range(10):
    data = ds.scan_data()
    if data is None: break
    pb = wandb_internal_pb2.Record()
    pb.ParseFromString(data)
    if pb.history.item:
        for item in pb.history.item:
            print(f"Key: {item.key}, Val: {item.value_json}")
        break
