import Algorithmia
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import json
import os.path
from pathlib import Path
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
    checkpoints_path = manifest["model_filepath"]
    if __name__ == "__main__":
        checkpoints = checkpoints_path
    else:
        checkpoints = client.file(checkpoints_path).getFile().name
        assert_model_md5(checkpoints)

    class_mapping = {
        0: "Movies_Negative",
        1: "Movies_Positive",
        2: "Food_Negative",
        3: "Food_Positive",
        4: "Clothing_Negative",
        5: "Clothing_Positive",
    }
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(class_mapping),
        output_attentions=False,
        output_hidden_states=False,
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load(checkpoints, map_location=torch.device("cpu")))
    return model, tokenizer, class_mapping


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


def evaluate_single(dataloader_val):
    model.eval()
    predictions, probs = [], []

    for batch in dataloader_val:
        # batch = tuple(b.to("cuda") for b in batch)
        batch = tuple(b for b in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs[0]
        predictions.append(logits.detach().cpu().numpy())
        softmax = torch.nn.Softmax(dim=1)
        probs.append(softmax(logits).detach().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    probs = np.concatenate(probs, axis=0)
    return predictions, probs


def predict_single(input_texts):
    encoded_inputs = tokenizer.batch_encode_plus(
        input_texts,
        max_length=250,
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    )

    input_ids = torch.tensor(encoded_inputs["input_ids"])
    attention_masks = torch.tensor(encoded_inputs["attention_mask"])

    tensor_ds = TensorDataset(input_ids, attention_masks)
    data_loader = DataLoader(
        tensor_ds, sampler=SequentialSampler(tensor_ds), batch_size=32
    )
    classes, probs = evaluate_single(data_loader)

    results = []

    for predicted_class, prob in zip(classes, probs):
        predicted_class = class_mapping[int(np.argmax(predicted_class))]
        confidence = float(np.max(prob))
        results.append(
            {"class": predicted_class, "confidence": confidence,}
        )
        client.report_insights({"model_confidence": confidence})
    return results


manifest = load_model_manifest()
model, tokenizer, class_mapping = load_model(manifest)


def apply(input):
    results = predict_single(input)
    return results


if __name__ == "__main__":
    # Now the apply() function will be able to access the locally loaded model
    test_input = [
        "I have two Corgis and they have mixed feelings about it, they seem okay well who knows",
        "These are fantastic tasting and really work when you add ANY meat to it",
    ]
    algo_result = apply(test_input)
    print(algo_result)
