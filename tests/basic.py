import numpy as np


radical_members = 2
members = 1000

prob_radical_left = (radical_members / members) / 2
prob_radical_right = (radical_members / members) / 2
prob_unaffiliated = 1 - prob_radical_left - prob_radical_right

affiliation = np.random.choice(['far-left', 'far-right', None], p=[prob_radical_left,
                                                                   prob_radical_right,
                                                                   prob_unaffiliated])


print(affiliation)