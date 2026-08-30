def p_update(prior,likelihood,false_positive):
    # P(E) = (PH/E * P(H) + (1-P(H/E))*P('H))
    evidence = (prior*likelihood + (1-prior)*false_positive)

    # P(H) = 0.0001, P(H/SICK)=0.99, P(H')=0.0999

    # P(E/H) = (PH/E)*P(H)/P(E)
    answer = (prior*likelihood)/evidence
    return answer 


prior_1 = p_update(
    0.0001,
    0.99,
    0.01#Falsse positive = 1-0.99 = 0.01 wrongly identifiess healthy ass sick
)

print(f"Prior1: {prior_1:.4f}")


prior_2 = p_update(
    prior_1,
    0.99,
    0.01
)

print(f"Prior 2 : {prior_2:.4f}")
