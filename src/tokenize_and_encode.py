import Algorithmia
from joblib import load

from tokenizer import DistilBertTokenizer
from encoder import DistilBertEncoder


client = Algorithmia.client()

# Tokenizer
tokenizer = load(client.file(
    "data://tokenize_and_encode/distilmbert_tokenizer/").getFile().name)

# Encoder
encoder = load(client.file(
    "data://tokenize_and_encode/distilmbert/").getFile().name)


def apply(input):
    ##########################
    # Processing input data #
    ##########################


    ###########################################################
    # Tokenization and Encoding #
    ###########################################################
    tokenized_messages = tokenizer.forward(input)
    encodings = encoder.forward(tokenized_messages)

    ##########################################
    # Return output to consuming application #
    ##########################################
    return {"risk_score": risk_score, "approved": approved}