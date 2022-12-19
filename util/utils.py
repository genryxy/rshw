import pickle


def load_model():
    model = pickle.load(open('../model/conf/decision_tree.pkl', 'rb'))
    return model
