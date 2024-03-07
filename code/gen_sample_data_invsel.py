import numpy as np
import itertools

# Hypothetical data

states = ["qualified lead", "aware", "considering", "evaluating", "purchased", "onboarded", "engaged", "to-be-retained", "advocating", "annoyed", "about-to-leave", "exited"]
actions = ["ad1", "ad2", "email1", "email2", "call1", "call2"]
ps = ["John Brown", "Jane Doe", "Jack Reacher", "Jack Ryan"]
nc = len(states)
# change this if you want to test more or less
N_CANDIDATE_INTERVENTIONS = 10

with open("sampledata/journeydata.csv", "w") as fd:
	for s in states:
		for p in ps:
			n = np.random.randint(100, 10000)
			fd.write(f"{s},{p},{n}\n")

with open("sampledata/transitions.csv", "w") as fd:
	for p in ps:
		for ss in states:
			for a in actions:
				ns = np.random.uniform(size = nc)
				ns = ns / np.sum(ns)
				for st, prob in zip(states, ns):
					fd.write(f"{p},{a},{ss},{st},{prob}\n")

with open("sampledata/personae.csv", "w") as fd:
	for p in ps:
		n = np.random.random()
		fd.write(f"{p},{n}\n")

with open("sampledata/actioncosts.csv", "w") as fd:
    costs = np.random.random(size = len(actions))
    costs = costs / np.sum(costs)
    for a, c in zip(actions, costs):
        fd.write(f"{a},{c}\n")

with open("sampledata/candidates.csv", "w") as fd:
	n_ps = len(ps)
	n_states = len(states)
	n_actions = len(actions)
	n_choices = n_ps * n_states * n_actions
	all_interventions = list(itertools.product(*[ps, states, actions]))
	for idx in np.random.choice(np.arange(0, n_choices), N_CANDIDATE_INTERVENTIONS, replace = False):
		p, s, a = all_interventions[idx]
		fd.write(f"{p},{s},{a}\n")	

