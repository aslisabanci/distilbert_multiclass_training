import Algorithmia
import json
import os.path
from pathlib import Path
import joblib
import hashlib


client = Algorithmia.client()


def load_model_manifest(rel_path="model_manifest.json"):
    """Loads the model manifest file as a dict.
    A manifest file has the following structure:
    {
      "model_filepath": Uploaded model path on Algorithmia data collection
      "model_md5_hash": MD5 hash of the uploaded model file
      "model_origin_repo": Model development repository with the Github CI workflow
      "model_origin_ref": Branch of the model development repository related to the trigger of the CI workflow,
      "model_origin_commit_SHA": Commit SHA related to the trigger of the CI workflow
      "model_origin_commit_msg": Commit message related to the trigger of the CI workflow
      "model_uploaded_utc": UTC timestamp of the automated model upload
    }
    """
    manifest = []
    manifest_path = "{}/{}".format(Path(__file__).parents[1], rel_path)
    if os.path.exists(manifest_path):
        with open(manifest_path) as json_file:
            manifest = json.load(json_file)
    return manifest


def load_model(manifest):
    """Loads the model object from the file at model_filepath key in config dict"""
    model_path = manifest["model_filepath"]
    if __name__ == "__main__":
        model_file = model_path
    else:
        model_file = client.file(model_path).getFile().name
        assert_model_md5(model_file)
    model_obj = joblib.load(model_file)
    return model_obj


def assert_model_md5(model_file):
    """
    Calculates the loaded model file's MD5 and compares the actual file hash with the hash on the model manifest
    """
    md5_hash = None
    DIGEST_BLOCK_SIZE = 128 * 64
    with open(model_file, "rb") as f:
        hasher = hashlib.md5()
        buf = f.read(DIGEST_BLOCK_SIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(DIGEST_BLOCK_SIZE)
        md5_hash = hasher.hexdigest()
    assert manifest["model_md5_hash"] == md5_hash
    print("Model file's runtime MD5 hash equals to the upload time hash, great!")


manifest = load_model_manifest()
model = load_model(manifest)


# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    # You can access the model object from this scope
    return f"Echoing back input: {input}"


if __name__ == "__main__":
    # Now the apply() function will be able to access the locally loaded model
    algo_result = apply("Test input")
    print(algo_result)
