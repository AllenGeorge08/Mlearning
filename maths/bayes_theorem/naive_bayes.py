classes = {
    "spam": 0.3,
    "ham": 0.7
}

word_probs = {
    "spam": {
        "lottery": 0.05,
        "free": 0.10,
        "money": 0.08
    },
    "ham": {
        "lottery": 0.001,
        "free": 0.01,
        "money": 0.005
    }
}


message = ["lottery","free","money"]

scores = {}

for cls in classes:
    score = classes[cls] 

    for word in message:
        score *= word_probs[cls][word]

    scores[cls] = score 



print(f"The scores are : {scores}")
print(f"Prediction : {max(scores)}")


