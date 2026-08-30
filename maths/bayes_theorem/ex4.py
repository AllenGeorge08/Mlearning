from collections import defaultdict
import math 
class NaiveBayes:
    def __init__(self,smoothing=1.0):
        self.smoothing = smoothing 
        self.class_counts = defaultdict(int) #How many documents belong to each class...
        self.word_counts = defaultdict(lambda: defaultdict(int))#How many times each word appears in each class.
        self.class_word_totals = defaultdict(int) #Total number of words in each class.
        self.vocab = set()  #contains every unique word we've encountered.
        self.length_counts = defaultdict(lambda: defaultdict(int))  #count of spam in short

    
    def train(self,documents,labels):
        for doc,label in zip(documents,labels):
            self.class_counts[label] += 1

            words = doc.lower().split()

            length_category = "long" if len(words) > 4 else "short"
            self.length_counts[label][length_category] += 1  #count of spam/ham in short/long


            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

            

    
    def predict(self,document):
        words = document.lower().split()

        length_category = "long" if len(words) > 4 else "short"

        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)

        best_class = None 
        best_score = float('-inf')

       
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls]/total_docs)
            
            # Adding pP(lENGHT/Class)
            # Vocab size for length is 2 (short,loong)
            length_count = self.length_counts[cls].get(length_category,0)  #total spam/ham in length_category if existss cool else 0
            total_cls_docs = self.class_counts[cls] # total spam\ham across long+short
            length_prob = (length_count+self.smoothing)/(total_cls_docs+self.smoothing*2)

            score += math.log(length_prob)

            for word in words:
                #our count
                count = self.word_counts[cls].get(word,0)
                total = self.class_word_totals[cls]#total count of words
                probability = (count+self.smoothing)/(total+self.smoothing*vocab_size)  #the num is laplace ssmoothing
                score += math.log(probability)  #We're adding log probabilities instead of multiplying probabilities.
                

            if score>best_score:
                best_score = score 
                best_class = cls 


        return best_class,best_score




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


model = NaiveBayes()
model.train(train_docs,train_labels)

# Extract P(short|spam)
short_spam = model.length_counts["spam"]["short"]
total_spam = model.class_counts["spam"]
p_short_spam = (short_spam+model.smoothing)/(total_spam+model.smoothing*2)  #

# p(SHORt|pam)
short_ham = model.length_counts["ham"]["short"] #total_shorts in ham
total_ham = model.class_counts["ham"]
p_short_ham = (short_ham+model.smoothing)/(total_ham+model.smoothing*2)  #it's count(feature,class), total_ham, a is smoothing factor and k is possible categories..


print(f"P(Short|Spam) = {p_short_spam:.3f}")
print(f"P(Short|Ham) = {p_short_ham:.3f}")
