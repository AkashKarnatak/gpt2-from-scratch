import torch
from gpt2 import GPT2Model
from bert import BertModel
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    BertTokenizer,
    BertModel as BertModelHF,
)


def test_gpt2():
    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # load custom gpt2 model
    gpt = GPT2Model.from_pretrained("gpt2")
    # load gpt2 model from HF
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # test
    text = "Hi, I am a language model,"
    encoded_input = tokenizer(text, return_tensors="pt")

    hf_out = model(encoded_input["input_ids"]).logits
    custom_out = gpt(encoded_input["input_ids"])
    print("HF GPT2 output:\n", hf_out)
    print("Custom GPT2 output:\n", custom_out)
    assert (
        torch.abs(hf_out - custom_out).max() < 1e-5
    ), "Sanity check for GPT2Model failed"
    print("Sanity check for GPT2Model passed")


def test_bert():
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # load custom gpt2 model
    bert = BertModel.from_pretrained("bert-base-uncased")
    # load gpt2 model from HF
    model = BertModelHF.from_pretrained("bert-base-uncased")
    model.eval()

    # test
    text = "Hi, I am a language model,"
    encoded_input = tokenizer(text, return_tensors="pt")

    hf_model_out = model(encoded_input["input_ids"])
    hf_out, hf_pool = hf_model_out.last_hidden_state, hf_model_out.pooler_output
    out, pool = bert(encoded_input["input_ids"])
    print("HF GPT2 output:\n", hf_out)
    print("Custom GPT2 output:\n", out)
    assert (
        torch.abs(hf_out - out).max() < 1e-5 and torch.abs(hf_pool - pool).max() < 1e-5
    ), "Sanity check for BertModel failed"
    print("Sanity check for BertModel passed")


test_gpt2()
test_bert()
