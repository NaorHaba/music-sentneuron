import os
import json
import argparse
import numpy as np
import tensorflow as tf
import midi_encoder as me

from train_generative import build_generative_model
from train_classifier import preprocess_sentence, get_states_from_layers

GENERATED_DIR = './generated'


def update_states_for_layers(model, layer_idxs, h_state, c_state):
    for layer_idx in layer_idxs:
        units = model.get_layer(index=layer_idx).units
        model.get_layer(index=layer_idx).states = (tf.Variable(h_state[(layer_idx - 1) * units: layer_idx * units].reshape(1, units)),
                                                   tf.Variable(c_state[(layer_idx - 1) * units: layer_idx * units].reshape(1, units)))
    return model


def override_neurons(model, layer_idxs, override):
    h_state, c_state = get_states_from_layers(model, layer_idxs)

    for neuron, value in override.items():
        # why is this int(value) - its making everything zero
        c_state[int(neuron)] = value

    model = update_states_for_layers(model, layer_idxs, h_state, c_state)


def sample_next(predictions, k):
    # Sample using a categorical distribution over the top k midi chars
    top_k = tf.math.top_k(predictions, k)
    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values = top_k[0].numpy().squeeze()

    if np.random.uniform(0, 1) < .5:
        predicted_id = top_k_choices[0]
    else:
        p_choices = tf.math.softmax(top_k_values[1:]).numpy()
        predicted_id = np.random.choice(top_k_choices[1:], 1, p=p_choices)[0]

    return predicted_id


def process_init_text(model, init_text, char2idx, layer_idxs, override):
    model.reset_states()

    for c in init_text.split(" "):
        # Run a forward pass
        try:
            input_eval = tf.expand_dims([char2idx[c]], 0)

            # override sentiment neurons
            override_neurons(model, layer_idxs, override)

            predictions = model(input_eval)
        except KeyError:
            if c != "":
                print("Can't process char", c)

    return predictions


def generate_midi(model, char2idx, idx2char, init_text="", seq_len=256, k=3, layer_idxs=(-2, ), override={}):
    """
    Generates a midi using the model.
    if override is not empty it's indexes are neuron indexes and its values are the activation values in [-1,1]
    while predicting the neurons in the c_state that are in override stay the same, and the other chang according to what
    is seen.
    this way we can keep the sentiment the same while still changing the song.
    """

    # Add front and end pad to the initial text
    init_text = preprocess_sentence(init_text)

    # Empty midi to store our results
    midi_generated = []

    # Process initial text
    predictions = process_init_text(model, init_text, char2idx, layer_idxs, override)

    # Here batch size == 1
    model.reset_states()
    for i in range(seq_len):
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0).numpy()

        # Sample using a categorical distribution over the top k midi chars
        predicted_id = sample_next(predictions, k)

        # Append it to generated midi
        midi_generated.append(idx2char[predicted_id])

        # override sentiment neurons
        override_neurons(model, layer_idxs, override)

        # Run a new forward pass
        input_eval = tf.expand_dims([predicted_id], 0)
        predictions = model(input_eval)

    return init_text + " " + " ".join(midi_generated)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--seqinit', type=str, default="\n", help="Sequence init.")
    parser.add_argument('--seqlen', type=int, default=256, help="Sequence length.")
    parser.add_argument('--cellix', nargs='+', type=int, default=-2, help="LSTM layer to use as encoder.")
    parser.add_argument('--override', type=str, default="", help="JSON file with neuron values to override.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Load override dict from json file
    override = {}

    try:
        with open(opt.override) as f:
            override = json.load(f)
    except FileNotFoundError:
        print("Override JSON file not provided.")

    # Create idx2char from char2idx dict
    idx2char = {idx: char for char, idx in char2idx.items()}

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild model from checkpoint
    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(opt.model))
    model.build(tf.TensorShape([1, None]))

    # Generate a midi as text
    midi_txt = generate_midi(model, char2idx, idx2char, opt.seqinit, opt.seqlen, layer_idxs=opt.cellix,
                             override=override)
    print(midi_txt)

    me.write(midi_txt, os.path.join(GENERATED_DIR, "generated.mid"))
