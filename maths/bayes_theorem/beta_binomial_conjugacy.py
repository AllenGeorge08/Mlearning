def update_beta(alpha,beta,succeses,failure):
    alpha += succeses
    beta + failure 
    return alpha,beta 


alpha,beta = update_beta(
    1,
    1,
    7,
    3
)

print(alpha,beta)
print(f"Mean : {alpha/(alpha+beta)}")

alpha_one,beta_one = update_beta(
    alpha,
    beta,
    5,
    5
)

print(alpha_one,beta_one)
print(f"Mean: {alpha_one/(alpha_one+beta_one)}")