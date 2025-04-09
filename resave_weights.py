'''Utility script re-save weight file as "weights only"'''


import torch
import pathlib
import ultralytics

orig_path = pathlib.Path( __file__ ).parent / "models" / "ultralytics" / "yolo" / "v8n" / "yolov8n.pt"
new_path = orig_path.with_name( "yolov8n_just_weights.pt" )

assert orig_path.is_file()

# # Money patch
# _original_torch_load = torch.load

# # Define a new function that forces weights_only=False
# def custom_torch_load(*args, **kwargs):
#     if "weights_only" not in kwargs:
#         kwargs["weights_only"] = False
#     return _original_torch_load(*args, **kwargs)

safe = [
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.Conv,
    ultralytics.nn.modules.C2f
]

with torch.serialization.safe_globals(safe):

    model = torch.load(
        str( orig_path ),
        weights_only = True
    )

    print( model )

    # torch.save(
    #     model,
    #     str( new_path )
    # )
