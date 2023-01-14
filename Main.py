#################################################################
# FILE : Main.py
# WRITERS :  Linoy Bushari, Yuval Gabbay, Noa Goldenberg, Bar Melinarskiy, and Yuval Roditi
# EXERCISE : Algorithms in Computational Biology - 76558 - Hackathon
# DESCRIPTION: HMM parameter learning, CpG islands example
#################################################################

from HMM import *
from Trees import *
from PCA import *

import argparse
from itertools import groupby
import pandas as pd
import numpy as np
from hmmlearn import hmm
import gzip as gz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

NUMBER_OF_COMPONENTS = 8
MAX_ITERATIONS = 100
DIC_STATES = {"A": 0, "C": 1, "G": 2, "T": 3}
DIC_BASES = {"A": [0, 4], "C": [1, 5], "G": [2, 6], "T": [3, 7]}
CORRECT_THRESHOLD = 0.7

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'


def fasta_read(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    if is_gz_file(fasta_name):  # check if given file is gzip or not, we need to open is accordingly
        file = gz.open(fasta_name)
    else:
        file = open(fasta_name, 'rb')  # open in bytes mode just like when we open the gzip
    faiter = (x[1] for x in groupby(file, lambda line: line.decode().startswith(">")))
    for header in faiter:
        header = next(header)[1:].decode().strip()
        seq = "".join(s.decode().strip() for s in next(faiter))
        yield header, seq


def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. seqs_ATTA.fasta)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    parser.add_argument('decodeAlg',
                        help='The algorithm name for predicting/decoding hidden states. Can be "posterior" or "viterbi".')
    parser.add_argument('outputPrefix', help='The prefix of output.')
    return parser.parse_args()


def preprocess_fasta(fasta):
    """
    We need to change the DNA bases into an input the model knows how to process
    Coding works as follows:
    A = 1000
    C = 0100
    G = 0010
    T = 0001
    @return two arrays: X - [4, #total_number_of_bases], sequences_length - [#_total_number_sequences]
    """
    sequences = []
    sequences_lengths = []
    for fasta_header, fasta_seq in fasta_read(fasta):
        for c in fasta_seq:
            sequences.append([DIC_STATES[c.upper()]])
        sequences_lengths.append(len(fasta_seq))
    sequences = np.array(sequences)
    sequences_lengths = np.array(sequences_lengths, dtype=int)
    return sequences, sequences_lengths


def print_model_parameters(model, title, learned_params_output_file=None):
    """
    Print the model's startprob_, transmat_ and emissionprob_ tables
    """
    print(Colors.OKBLUE + title + Colors.ENDC, '\n')
    print(Colors.OKGREEN + "start probabilities:" + Colors.ENDC)
    headers = ['A+', 'C+', 'G+', 'T+', 'A-', 'C-', 'G-', 'T-']
    print(pd.DataFrame(np.atleast_2d(model.startprob_), [""], headers))
    print('\n')
    print(Colors.OKGREEN + "transition probabilities:" + Colors.ENDC)
    df = pd.DataFrame(model.transmat_, headers, headers)
    print(df)
    print()
    if learned_params_output_file:  # print the learned transition probabilities to output file
        file = open(learned_params_output_file, 'w')
        if file:
            file.write(df.to_string(header=False, index=False))
            file.close()

    print(Colors.OKGREEN + "emission probabilities:" + Colors.ENDC)
    print(pd.DataFrame(model.emissionprob_, headers, ['A', 'C', 'G', 'T']))
    print(Colors.BOLD + "*" * 150 + Colors.ENDC)


def train(convergence_thr, algo, learned_params_output_file):
    """
    Train the CpG island model
    We always train on the 2 given train files: train_background.fa.gz, train_cpg_island.fa.gz
    merged together as one big X given to the fit method
    """
    # we train the model on the 2 fasta files
    train_files = ["train_background.fa.gz", "train_cpg_island.fa.gz", "1000_cpg_island.train.padded.fa.gz"]
    cpg_train_seq, cpg_train_lengths = preprocess_fasta(train_files[0])
    bgd_train_seq, bgd_train_lengths = preprocess_fasta(train_files[1])
    pad_cpg_train_seq, pad_cpg_train_lengths = preprocess_fasta(train_files[2])
    sequences = np.concatenate((cpg_train_seq, bgd_train_seq, pad_cpg_train_seq), axis=0)
    sequences_lengths = np.concatenate((cpg_train_lengths, bgd_train_lengths, pad_cpg_train_lengths))
    # Initialized the model's parameters
    start_probability = np.array([2 / 600, 4/600, 4/600, 2 / 600, 0.245, 0.245, 0.245, 0.245])

    transition_probability = np.array([
        [0.1772237, 0.2682517, 0.4170629, 0.1174825, 0.0035964, 0.0054745, 0.0085104, 0.0023976],
        [0.1682435, 0.3599201, 0.267984, 0.1838722, 0.0034131, 0.0073453, 0.005469, 0.0037524],
        [0.1586223, 0.3318881, 0.3671328, 0.1223776, 0.0032167, 0.0067732, 0.0074915, 0.0024975],
        [0.0783426, 0.3475514, 0.375944, 0.1781818, 0.0015784, 0.0070929, 0.0076723, 0.0036363],
        [0.0012997, 0.0002047, 0.0002837, 0.0002097, 0.2994005, 0.2045904, 0.2844305, 0.2095804],
        [0.0013216, 0.0002977, 0.0000769, 0.0003016, 0.3213566, 0.2974045, 0.0778441, 0.3013966],
        [0.0011768, 0.000238, 0.0002917, 0.0002917, 0.1766463, 0.2385224, 0.2914165, 0.2914155],
        [0.0012477, 0.0002457, 0.0002977, 0.0002077, 0.2475044, 0.2455084, 0.2974035, 0.2075844]
    ])

    emission_probability = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    model = hmm.CategoricalHMM(n_components=NUMBER_OF_COMPONENTS, algorithm=algo, n_iter=MAX_ITERATIONS,
                               tol=convergence_thr, init_params="cm")
    model.emissionprob_ = emission_probability
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    # Print the model's parameters before we train
    print(Colors.OKCYAN + "Running 1. HMM Description" + Colors.ENDC)
    print(Colors.BOLD + "*" * 150 + Colors.ENDC)
    print_model_parameters(model, "Model's parameters initialized")
    logprob, state_sequence = model.decode(sequences, lengths=sequences_lengths, algorithm=algo)
    print(Colors.OKCYAN + "Loglikehood of the training set before training the model :" + str(logprob) + Colors.ENDC)
    training_message = "Training the model n_iter = " + str(MAX_ITERATIONS) + ", convergence tolerance = " + str(
        convergence_thr)
    print(Colors.OKCYAN + training_message + Colors.ENDC)
    # Train the model with the given files data
    model.fit(sequences, sequences_lengths)
    # Print the model's parameters after we trained
    logprob, state_sequence = model.decode(sequences, lengths=sequences_lengths, algorithm=algo)
    print(Colors.OKCYAN + "Loglikehood of the training set after training the model:" + str(logprob) + Colors.ENDC)
    print(Colors.BOLD + "*" * 150 + Colors.ENDC)
    print_model_parameters(model, "Model's parameters after learning", learned_params_output_file)
    return model


def predict(model, fasta, algo):
    logprobs, state_sequences, hidden_states_sequences = [], [], []
    total_number_of_bases = 0
    for fasta_header, fasta_seq in fasta_read(fasta):
        sequence = []
        for c in fasta_seq:
            sequence.append([DIC_STATES[c.upper()]])
        sequence = np.array(sequence)
        sequence_length = np.array([len(fasta_seq)], dtype=int)
        logprob, state_sequence = model.decode(sequence, lengths=sequence_length, algorithm=algo)
        total_number_of_bases += len(state_sequence)
        logprobs.append(logprob)
        hidden_states = np.where(state_sequence >= 4, 'N', 'I')  # states >=4 are not inside CpG islands
        hidden_states_sequences.append(hidden_states)
        state_sequences.append(state_sequence)

    return np.array(logprobs), np.array(state_sequences, dtype=object), np.array(hidden_states_sequences,
                                                                                 dtype=object), total_number_of_bases


def compute_accuracy(arr_predicted_states, arr_hidden_states_sequences, file_labels):
    index = 0
    accuracies_bases, accuracies_cpg = np.ones((len(arr_predicted_states), 4)), []
    accuracies_bases_cpg, accuracies_bases_non_cpg = accuracies_bases.copy(), accuracies_bases.copy()
    total_bases = 0
    sum_of_hits = 0
    for fasta_header, fasta_labels in fasta_read(file_labels):
        y_pred_bases = arr_predicted_states[index]
        y_pred_cpg = arr_hidden_states_sequences[index]
        y_true_bases = np.array([*str(fasta_labels)], dtype=int)
        base_index = 0
        for base, base_values in DIC_BASES.items():  # check accuracy per DNA base
            # get only the indexes where the current base is
            base_indexes = np.argwhere(np.isin(y_true_bases, base_values)).ravel()
            if len(base_indexes) > 0:
                y_true_base = y_true_bases[base_indexes]
                y_pred_base = y_pred_bases[base_indexes]
                # overall accuracy of current base
                score_base = accuracy_score(y_true_base, y_pred_base, normalize=True)
                accuracies_bases[index][base_index] = score_base
                # cpg accuracy of current base
                cpg_indexes = np.argwhere(y_true_base < 4).ravel()
                if len(cpg_indexes) > 0:
                    score_base_cpg = accuracy_score(y_true_base[cpg_indexes], y_pred_base[cpg_indexes], normalize=True)

                    accuracies_bases_cpg[index][base_index] = score_base_cpg
                non_cpg_indexes = np.argwhere(y_true_base >= 4).ravel()
                # non-cpg accuracy of current base
                if len(non_cpg_indexes) > 0:
                    score_base_non_cpg = accuracy_score(y_true_base[non_cpg_indexes], y_pred_base[non_cpg_indexes],
                                                        normalize=True)
                    accuracies_bases_non_cpg[index][base_index] = score_base_non_cpg

            base_index += 1
        y_true_cpg = np.where(y_true_bases >= 4, 'N', 'I')  # states >=4 are not inside CpG islands
        score_cpg = accuracy_score(y_true_cpg, y_pred_cpg, normalize=True)
        total_hits = accuracy_score(y_true_cpg, y_pred_cpg, normalize=False)
        total_bases += len(y_pred_cpg)
        sum_of_hits += total_hits
        accuracies_cpg.append(score_cpg)
        index += 1
    total_accuracy = sum_of_hits / total_bases
    return accuracies_bases.mean(axis=0), accuracies_bases_cpg.mean(axis=0), \
           accuracies_bases_non_cpg.mean(axis=0), np.array(accuracies_cpg),total_accuracy


def analyze_model(model, algo, output_path):
    sequences_files = ["30_cpg_island.padded.fa.gz", "70_non_cpg_island.padded.fa.gz", "1000_cpg_island.train.padded.fa.gz"]
    labels_files = ["30_cpg_island.label.fa.gz", "70_non_cpg_island.label.fa.gz", "1000_cpg_island.train.padded.label.fa.gz"]

    for file_sequences, file_labels in zip(sequences_files, labels_files):
        logprobs, state_sequences, hidden_states_sequences, total_number_of_bases = predict(model, file_sequences, algo)
        """
            Code for question 2  
            Calculate the average likelihood per base (use a geometric mean by dividing the
            log-likelihood by the total number of bases and then exponentiation).
            """
        print(Colors.OKCYAN + "Running 2. Per-base likelihood for file: " + str(file_sequences) + Colors.ENDC)
        average_likelihood = np.exp(logprobs.sum() / total_number_of_bases)
        print("Average likelihood per base: ", str(average_likelihood))
        accuracies_bases, accuracies_bases_cpg, accuracies_bases_non_cpg, accuracies_cpg, total_accuracy = compute_accuracy(
            state_sequences, hidden_states_sequences, file_labels)
        percentage_of_correct_sequences = (accuracies_cpg >= CORRECT_THRESHOLD).sum() / len(state_sequences) * 100
        print("Sequences with accuracy of >= ", str(CORRECT_THRESHOLD), "are considered correct")
        print("Percentage of correct sequences: ", str(percentage_of_correct_sequences))
        print(Colors.BOLD + "*" * 150 + Colors.ENDC)
        # Create plot
        x_indexes = np.arange(len(state_sequences))
        x_bases = list(DIC_BASES.keys())
        avg_accuracy = accuracies_cpg.mean()
        title = "Model analysis using the file: " + str(file_sequences)
        sub_titles = ("Average DNA base accuracy of prediction",
                      "CpG islands accuracy of prediction per sequence, Total Accuracy = " + str(total_accuracy))
        fig = make_subplots(rows=2, cols=1, y_title="Accuracy", subplot_titles=sub_titles)
        fig.append_trace(go.Bar(x=x_bases, y=accuracies_bases, name="Overall Base Accuracy"), row=1, col=1)
        fig.append_trace(go.Bar(x=x_bases, y=accuracies_bases_cpg, name="CpG Base Accuracy"), row=1, col=1)
        fig.append_trace(go.Bar(x=x_bases, y=accuracies_bases_non_cpg, name="NonCpG Base Accuracy"), row=1, col=1)
        fig.append_trace(go.Scatter(x=x_indexes, y=accuracies_cpg, name="Per Sequence CpG Islands Accuracy"), row=2, col=1)
        fig.update_layout(title_text=title, title_x=0.5)
        fig.update_layout(yaxis_range=[0, 1])
        # Update xaxis properties
        fig.update_xaxes(title_text="Bases", row=1, col=1)
        fig.update_xaxes(title_text="Sequence Index", row=2, col=1)

        fig.show()
        path = output_path + "analysis_" + str(file_sequences) + ".png"
        fig.write_image(path, width=1080, height=1080)


def main():
    args = parse_args()
    if len(args.outputPrefix) > 0 and args.outputPrefix[-1] != '/':
        args.outputPrefix += '/'
    prediction_output_file = args.outputPrefix + "cpg_island_predictions.txt"
    likelihood_output_file = args.outputPrefix + "likelihood.txt"
    learned_params_output_file = args.outputPrefix + "params.txt"
    convergence_thr = args.convergenceThr
    algo = args.decodeAlg
    if algo == "posterior":  # the hmm calls
        algo = "map"

    model = train(convergence_thr, algo, learned_params_output_file)
    logprobs, state_sequences, hidden_states_sequences, total_number_of_bases = predict(model, args.fasta, algo)

    # Create likelihood_output_file.txt file
    file = open(likelihood_output_file, 'w')
    if file:
        file.writelines('\n'.join(np.array(logprobs, dtype=str)))
        file.close()
    # Create prediction_output_file.txt file
    file = open(prediction_output_file, 'w')
    if file:
        for line in hidden_states_sequences:
            file.write(''.join(line) + "\n")
        file.close()

    """
    Code for question 3
    You received as part of the HW 6 fasta files. These included sequnces to be used
    for training the HMM model and sequences to be used to analyze performance.
    Provide measures of performance of the HMM model you trained at predicting
    CpG islands. Provide the average DNA base accuracy of predicting, for each
    base, whether it is a CpG island or not. Describe a way to measure average
    performance over a set of sequences and measure this metric on your model
    on the non-training data provided. This metric should provide a measure for
    what proportion of sequences you are able to predict correctly. You should
    determine a measure of correctness for each sequence
    """
    print(Colors.OKCYAN + "Running 3. HMM performance" + Colors.ENDC)
    analyze_model(model, algo, args.outputPrefix)

    print("This is the end!")


if __name__ == '__main__':
    main()
