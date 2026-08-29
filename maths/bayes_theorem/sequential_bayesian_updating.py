def update(prior,likelihood,false_positive):
    # P(E) = P(E/H)*P(H) + P(E/H')*P(H')
    evidence = (prior*likelihood + false_positive*(1-prior))

    # P(H/E) = P(E/H)*P(H)/P(E)
    posterior = likelihood*prior/evidence 
    return posterior


prior = 0.1

posterior = update(
    prior,
    likelihood=0.99,
    false_positive=0.05
)

posterior2 = update(
    posterior,
    likelihood=0.99,
    false_positive=0.05
)

print(f"Posterior 1 : {posterior}")
print(f"Posterior 2 : {posterior2}")