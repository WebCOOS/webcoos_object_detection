'''Utility script to enumerate the existing models and print out the detection
class names available to the models.'''


from yolo_processing import YOLO_MODELS

print( "YOLO Models:" )

for ( model_name, _ ) in YOLO_MODELS.items():

    for ( model_version, the_model ) in YOLO_MODELS[model_name].items():

        print(
            f" - {model_name}/{model_version}"
        )

        for n in the_model.names:
            print( f"    - {the_model.names[n]} ({n})" )
