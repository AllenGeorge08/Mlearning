import math 
from collections import defaultdict



# Tiny example

# Suppose spam contains 10 total words and our vocabulary contains 5 words.

# For a word never seen in spam:

# $$ P(word|spam) = \frac{0+1}{10+5} = \frac1{15} $$

# Not zero anymore.


# How many documents/emails belong to each class
# class_counts = {
    # "spam": 2,
    # "ham": 1
# }

#How many times did each word appear inside aach class
# self.word_counts = defaultdict(lambda: defaultdict(int))

#How many total word are there in each class
#How many total words are there in each class? 

# This simply stores every unique word we've seen. self.vocab = set()


class NaiveBayes:
    def __init__(self,smoothing=1.0):
        self.smoothing = smoothing 
        self.class_counts = defaultdict(int) #How many documents belong to each class...
        self.word_counts = defaultdict(lambda: defaultdict(int))#How many times each word appears in each class.
        self.class_word_totals = defaultdict(int) #Total number of words in each class.
        self.vocab = set()  #contains every unique word we've encountered.

    
    def train(self,documents,labels):
        for doc,label in zip(documents,labels):
            self.class_counts[label] += 1

            words = doc.lower().split()

            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    
    def predict(self,document):
        words = document.lower().split()

        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)

        best_class = None 
        best_score = float('-inf')

        for cls in self.class_counts:
            score = math.log(self.class_counts[cls]/total_docs)

            for word in words:
                #our count
                count = self.word_counts[cls].get(word,0)
                total = self.class_word_totals[cls]#total count of words
                probability = (count+self.smoothing)/(total+self.smoothing*vocab_size)  #the num is laplace ssmoothing
                score += math.log(probability)  #We're adding log probabilities instead of multiplying probabilities.

            
            if score>best_score:
                best_score = score 
                best_class = cls 


        return best_class


# Score(class) = logP(class) + Summaition(LogP(word/class))
# P(word/class) = count(word,class)+1/ (total_words+ )


train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",

    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham"
]


classifier = NaiveBayes()
classifier.train(train_docs,train_labels)

test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]

for msg in test_messages:
    print(msg, "->", classifier.predict(msg))



