#################################################################
# FILE : HMM.py
# WRITERS :  Linoy Bushari, Yuval Gabbay, Noa Goldenberg, Bar Melinarskiy, and Yuval Roditi
# EXERCISE : Algorithms in Computational Biology - 76558 - Hackathon
# DESCRIPTION: HMM
#################################################################

import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import scipy.stats as stats
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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


class HMMmodel:
    NUMBER_OF_COMPONENTS = 3
    MAX_ITERATIONS = 500  # todo increase to 10000
    CORRECT_THRESHOLD = 0.6

    def __init__(self):
        self.healthy_model = None
        self.diagnosed_model = None

    def print_model_parameters(self, model, title,name, learned_params_output_file=None):
        """
        Print the model's startprob_, transmat_ and emissionprob_ tables
        """
        print(Colors.OKBLUE + title + Colors.ENDC, '\n')
        print(Colors.OKGREEN + "start probabilities:" + Colors.ENDC)
        print(pd.DataFrame(np.atleast_2d(model.startprob_), [""]))
        print('\n')
        print(Colors.OKGREEN + "transition probabilities:" + Colors.ENDC)
        df = pd.DataFrame(model.transmat_)
        print(df)
        print()

        print(Colors.OKGREEN + "means:" + Colors.ENDC)
        print(model.means_)
        print()
        print(Colors.OKGREEN + "covars:" + Colors.ENDC)
        print(model.covars_)
        print()




        fig = go.Figure()
        # sigma = math.sqrt(model.covars_[0])
        # x = np.linspace(model.means_ - 3 * sigma, model.means_ + 3 * sigma, 100)
        # fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, model.means_, sigma),
        #                          mode='lines',
        #                          name=title))
        x = np.linspace(-1500, 1500, 100)
        for mu, cov, title in zip(model.means_, model.covars_, ["0", "1", "2"]):
            sigma = math.sqrt(cov)
            #x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, mu, sigma),
                                     mode='lines',
                                     name=title))
        fig.update_layout(title="Normal Distributions- "+str(name), title_x=0.5)


        fig.show()
        path = learned_params_output_file + "dis.png"
        fig.write_image(path, width=1080, height=1080)

        if learned_params_output_file:  # print the learned transition probabilities to output file
            file = open(learned_params_output_file, 'w')
            if file:
                file.write(df.to_string(header=False, index=False))
                file.close()

    def train(self, train_healthy, healthy_lengths, train_diagnosed, diagnosed_lengths,
              convergence_thr,
              learned_params_output_file, algo="viterbi"):
        """
        Train the CpG models
        train_healthy:
        healthy_lengths:
        train_diagnosed:
        diagnosed_lengths:
        convergence_thr:
        learned_params_output_file:
        algo:
        """
        # Initialized the model's parameters
        self.healthy_model = hmm.GaussianHMM(n_components=HMMmodel.NUMBER_OF_COMPONENTS, algorithm=algo,
                                             n_iter=HMMmodel.MAX_ITERATIONS, tol=convergence_thr)
        self.diagnosed_model = hmm.GaussianHMM(n_components=HMMmodel.NUMBER_OF_COMPONENTS, algorithm=algo,
                                               n_iter=HMMmodel.MAX_ITERATIONS, tol=convergence_thr)
        # Train the model with the given files data
        # self.healthy_model.means_ = []
        # self.healthy_model.covars_ = []
        self.healthy_model.fit(train_healthy, healthy_lengths)
        self.diagnosed_model.fit(train_diagnosed, diagnosed_lengths)
        title = "Healthy Model's parameters after learning"
        self.print_model_parameters(self.healthy_model, title, "Healthy", learned_params_output_file + "hmmHealthy.txt")
        title = "Diagnosed Model's parameters after learning"
        self.print_model_parameters(self.diagnosed_model, title, "Diagnosed", learned_params_output_file + "hmmDiagnosed.txt")

    def predict(self, patient_data, patient_length=None, algo="viterbi"):
        """
        Predict on the given sample
        @return True if we predict that the given patient is healthy, False otherwise
        also returns the log probability of the healthy and it's hidden states,
        the log probability of the diagnosed and it's hidden states
        """
        logprob_healthy, states_healthy = self.healthy_model.decode(patient_data,
                                                                    lengths=patient_length, algorithm=algo)
        logprob_diagnosed, states_diagnosed = self.diagnosed_model.decode(patient_data,
                                                                          lengths=patient_length, algorithm=algo)
        is_healthy = logprob_healthy > logprob_diagnosed  # If the given patient is healthy
        return is_healthy, logprob_healthy, states_healthy, logprob_diagnosed, states_diagnosed


def read(path):
    data = pd.read_csv(path, compression='gzip', delimiter='\t')
    location_ids = data.to_numpy().transpose()[0]
    # samples = data.to_numpy().transpose()[1:96]
    samples = data.to_numpy().transpose()[1:]
    number_of_locations = len(location_ids)
    train, test = train_test_split(samples, train_size=0.85)
    train_size = train.shape[0]
    test_size = test.shape[0]
    return train.reshape(-1, 1), test.reshape(-1, 1), \
           np.array([number_of_locations] * train_size), np.array([number_of_locations] * test_size)


def create_model_accuracy_figure(hmmModel, test_healthy, test_lengths_healthy, test_diagnosed, test_lengths_diagnosed,
                                 output_path):
    score_healthy, score_diagnosed = 0, 0
    len_locations = test_lengths_healthy[0]
    states_healthy_for_plot = []
    states_diagnosed_for_plot = []

    # Predict on all the healthy patients
    for i in range(0, len(test_healthy), len_locations):
        is_healthy, logprob_healthy, states_healthy, logprob_diagnosed, states_diagnosed = hmmModel.predict(
            patient_data=test_healthy[i:i + len_locations], patient_length=len_locations)
        states_healthy_for_plot.append(states_healthy)
        if is_healthy:
            score_healthy += 1
    # Predict on all the diagnosed patients
    len_locations = test_lengths_diagnosed[0]
    for i in range(0, len(test_diagnosed), len_locations):
        is_healthy, logprob_healthy, states_healthy, logprob_diagnosed, states_diagnosed = hmmModel.predict(
            patient_data=test_diagnosed[i:i + len_locations], patient_length=len_locations)
        states_diagnosed_for_plot.append(states_diagnosed)
        if not is_healthy:
            score_diagnosed += 1
    states_healthy_for_plot = np.array(states_healthy_for_plot)
    states_diagnosed_for_plot = np.array(states_diagnosed_for_plot)
    total_score_norm = (score_healthy + score_diagnosed) / (len(test_lengths_healthy) + len(test_lengths_diagnosed))
    score_healthy_norm = score_healthy / len(test_lengths_healthy)
    score_diagnosed_norm = score_diagnosed / len(test_lengths_diagnosed)

    titles = ["HMM accuracy per group, overall model accuracy = " + str(total_score_norm),
              "Methylation Patterns Diagnosed", "Methylation Patterns Healthy"]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=["Healthy", "Diagnosed"])
    # fig.add_trace(go.Bar(x=["Healthy", "Diagnosed"], y=[score_healthy_norm, score_diagnosed_norm], name="Accuracy"))
    colors = ['royalblue', 'red']
    fig.append_trace(go.Pie(labels=["Healthy Correct", "Healthy Incorrect"],
                            values=[score_healthy_norm, 1 - score_healthy_norm],
                            name="Accuracy Healthy",
                            ), row=1, col=1)
    fig.append_trace(go.Pie(labels=["Diagnosed Correct", "Diagnosed Incorrect"],
                            values=[score_diagnosed_norm, 1 - score_diagnosed_norm], name="Accuracy Diagnosed"), row=1,
                     col=2)
    # fig.update_xaxes(title_text="Group")
    # fig.update_yaxes(title_text="Accuracy")
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.update_layout(title=titles[0], title_x=0.5)
    # fig.update_layout(yaxis_range=[0, 1])
    fig.show()
    path = output_path + "hmm_accuracy_analysis.png"
    fig.write_image(path, width=1080, height=1080)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=np.array(range(len(test_lengths_diagnosed))), y=np.array(range(len_locations)),
                             z=states_diagnosed_for_plot))
    fig.update_xaxes(title_text="Locations")
    fig.update_yaxes(title_text="Patients")
    fig.update_layout(title=titles[1], title_x=0.5)
    fig.show()
    path = output_path + titles[1] + ".png"
    fig.write_image(path, width=1080, height=1080)

    fig = go.Figure()
    df = pd.DataFrame(states_healthy_for_plot, dtype="category")
    fig.add_trace(go.Heatmap(x=np.array(range(len(test_lengths_healthy))), y=np.array(range(len_locations)),
                             z=df))
    fig.update_xaxes(title_text="Locations")
    fig.update_yaxes(title_text="Patients")
    fig.update_layout(title=titles[2], title_x=0.5)
    fig.show()
    path = output_path + titles[2] + ".png"
    fig.write_image(path, width=1080, height=1080)


if __name__ == '__main__':
    file_healthy_path = "Data/BRCA_Solid_Tissue_Normal.chr19.tsv.gz"
    file_diagnosed_path = "Data/BRCA_Primary_Tumor.chr19.tsv.gz"
    train_healthy, test_healthy, train_lengths_healthy, test_lengths_healthy = read(file_healthy_path)
    train_diagnosed, test_diagnosed, train_lengths_diagnosed, test_lengths_diagnosed = read(file_diagnosed_path)
    output_path = "Output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    hmmModel = HMMmodel()
    hmmModel.train(train_healthy=train_healthy, healthy_lengths=train_lengths_healthy,
                   train_diagnosed=train_diagnosed, diagnosed_lengths=train_lengths_diagnosed,
                   convergence_thr=HMMmodel.CORRECT_THRESHOLD,
                   learned_params_output_file=output_path)

    create_model_accuracy_figure(hmmModel, test_healthy, test_lengths_healthy, test_diagnosed, test_lengths_diagnosed,
                                 output_path)
    print("Were are Done!!!")
