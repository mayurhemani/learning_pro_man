import numpy as np
from typing import Tuple

states = {
    "qualified lead": 0,
    "aware": 1,
    "considering": 1,
    "evaluating": 1,
    "purchased": 1,
    "onboarded": 1,
    "engaged": 1,
    "to-be-retained": 1,
    "advocating": 1,
    "annoyed": -1,
    "about-to-leave": -1,
    "exited": -1
}

def csv_iterator(filepath: str):
    with open(filepath) as fd:
        lines = fd.read().splitlines()
        for l in lines:
            yield l.split(",")

class Transitions:
    def __init__(self, data_file_path: str) -> None:
        desirable_states = {k: v for k, v in states.items() if v > 0}
        undesirable_states = {k: v for k, v in states.items() if v < 0}
        self.p_good = {}
        self.p_bad = {}
        for ks in csv_iterator(data_file_path):
            # persona-action-state1-state2-transition-prob
            target_state = ks[3]
            prob = float(ks[4])
            target_dict = None
            if target_state in desirable_states:
                target_dict = self.p_good
            elif target_state in undesirable_states:
                target_dict = self.p_bad
            
            if not(target_dict is None):
                key = f"{ks[0]}-{ks[1]}-{ks[2]}"
                if not key in target_dict:
                    target_dict[key] = prob
                else:
                    target_dict[key] += prob

    def transition_probabilities(self, persona: str, action: str, src_state: str) -> Tuple[float, float]:
        key = f"{persona}-{action}-{src_state}"
        if (key in self.p_good) and (key in self.p_bad):
            return (self.p_good[key], self.p_bad[key])
        return (0., 0.)

class JourneyData:
    def __init__(self, data_file_path: str) -> None:
        self.pop = {}
        for ks in csv_iterator(data_file_path):
            key = f"{ks[0]}-{ks[1]}"
            self.pop[key] = int(ks[2])
        
    def population(self, state: str, persona: str) -> int:
        key = f"{state}-{persona}"
        return self.pop.get(key, 0)

class ActionCosts:
    def __init__(self, data_file_path: str) -> None:
        self.actions = {}
        for ks in csv_iterator(data_file_path):
            self.actions[ks[0]] = float(ks[1])

    def amortized_cost_fraction(self, action: str) -> float:
        return self.actions.get(action, 0)

class Personae:
    def __init__(self, data_file_path: str) -> None:
        self.personae = {}
        for ks in csv_iterator(data_file_path):
            self.personae[ks[0]] = float(ks[1])
        
    def opportunity_cost(self, persona:str) -> float:
        return self.personae.get(persona, 0.)

class ModelData:
    def __init__(
        self, 
        persona_csv: str, 
        journey_csv: str,
        transitions_csv: str,
        actions_csv: str):
        self.personae = Personae(persona_csv)
        self.journeys = JourneyData(journey_csv)
        self.transitions = Transitions(transitions_csv)
        self.actions = ActionCosts(actions_csv)

    def score_intervention(self, persona: str, action: str, state: str) -> float:
        good_prob, bad_prob = self.transitions.transition_probabilities(persona = persona, action = action, src_state = state)
        assert(0 <= good_prob <= 1.0)
        assert(0 <= bad_prob <= 1.0)
        raw_benefit = good_prob - bad_prob
        amo_cost = self.actions.amortized_cost_fraction(action)
        opp_cost = self.personae.opportunity_cost(persona)
        n_users = self.journeys.population(state = state, persona = persona)

        score = n_users * (raw_benefit + (1. - bad_prob) * opp_cost) / np.exp(amo_cost)
        return score



if __name__ == "__main__":

    model = ModelData(
            persona_csv = "sampledata/personae.csv",
            journey_csv = "sampledata/journeydata.csv",
            transitions_csv = "sampledata/transitions.csv",
            actions_csv = "sampledata/actioncosts.csv"
        )
    interv_scores = [(persona, state, action, model.score_intervention(persona=persona, action=action, state=state)) \
                        for persona, state, action in csv_iterator("sampledata/candidates.csv")]
    
    print("Top-5 interventions")
    for interv in sorted(interv_scores, key = lambda k: -k[-1])[:5]:
        print(f"Persona: {interv[0]}, State: {interv[1]}, Action: {interv[2]}, Score: {interv[3]}")
        