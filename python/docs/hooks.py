import warnings

def on_startup(**kwargs):
    warnings.filterwarnings(
        "ignore",
        message="Importing from obspec_utils",
    )
    warnings.filterwarnings(
        "ignore",
        message="Numcodecs codecs are not in the Zarr version 3 specification",
    )
