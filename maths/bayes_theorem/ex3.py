# Smoothing impact. Run the spam classifier with smoothing values of 0.01, 0.1, 1.0, and 10.0. How do the top word probabilities change? What happens with smoothing=0 and a word that appears only in ham?


from naive_bayes_classifier import NaiveBayes, train_docs,train_labels,test_messages


classifier_1 = NaiveBayes(smoothing=0.01)
classifier_2 =NaiveBayes(smoothing=0.1)
classifier_3 = NaiveBayes(smoothing=1.0)
classifier_4 = NaiveBayes(smoothing=10.0)

classifier_1.train(train_docs,train_labels)
classifier_2.train(train_docs,train_labels)
classifier_3.train(train_docs,train_labels)
classifier_4.train(train_docs,train_labels)

print("Predictions from classifier 1 ")
for msg  in test_messages:
    print(msg,"->",classifier_1.predict(msg))

print("-"*20)
print("Predictions from classifier 2")
for msg in test_messages:
    print(msg,"->",classifier_2.predict(msg))


print("-"*20)
print("Predictions from classifier 3")
for msg in test_messages:
    print(msg,"->",classifier_3.predict(msg))


print("-"*20)
print("Predictions from classifier 4")
for msg in test_messages:
    print(msg,"->",classifier_4.predict(msg))


# Smoothing value of 0 gives an Value error 
# print("-"*20)
# print("Predictions for smoothing value = 0")
# classifier_5 = NaiveBayes(smoothing=0)
# classifier_5.train(train_docs,train_labels)

# for msg in test_messages:
#     print(msg,"Classifier 5 : ->",classifier_5.predict(msg))

