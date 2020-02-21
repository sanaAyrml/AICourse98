import pandas as pd

def cleaning(word):
    cleaned = ['.','?','!','{', '}', ',','(', ')', ':', ';', '&',  '*', '/','+', '=']
    while True:
        if len(word) == 0 or word.startswith('http://') or word.startswith('#') :
            return ''
        if cleaned.__contains__(word[-1]):
            word = word[:-1]
        elif cleaned.__contains__(word[0]):
            word = word[1:]
        else:
            return word.lower()

df = pd.read_csv('/Users/sana/Downloads/ExtractedTweets.csv')
df = df.sample(frac=1).reset_index(drop=True)
df_train = df[:60000]
df_test= df[60000:]
pos_dic = dict()
total_pos = 0
tweet_pos = 0
neg_dic = dict()
total_neg = 0
tweet_neg = 0
number_of_words = 0
for label, tweet in df_train.values:
    words = tweet.split()
    if label == 1:
        tweet_pos += 1
        for word in words:
            word = cleaning(word)
            if pos_dic.__contains__(word):
                pos_dic[word] += 1
            else:
                if not neg_dic.__contains__(word):
                    number_of_words += 1
                pos_dic[word] = 1
            total_pos += 1
    elif label == 0:
        tweet_neg += 1
        for word in words:
            word = cleaning(word)
            if neg_dic.__contains__(word):
                neg_dic[word] += 1
            else:
                if not pos_dic.__contains__(word):
                    number_of_words += 1
                neg_dic[word] = 1
            total_neg += 1
true = 0
total = 0
for label, tweet in df_test.values:
    words = tweet.split()
    prob_pos = float(tweet_pos/(tweet_neg+tweet_pos))
    prob_neg = float(tweet_neg/(tweet_neg+tweet_pos))
    for word in words:
        word = cleaning(word)
        if pos_dic.__contains__(word):
            prob_pos = prob_pos * float(pos_dic[word]+1/(total_pos+number_of_words))
        else:
            prob_pos = prob_pos * float(1 / (total_pos + number_of_words))
        if neg_dic.__contains__(word):
            prob_neg = prob_neg * float(neg_dic[word]+1/(total_neg+number_of_words))
        else:
            prob_neg = prob_neg * float(1 / (total_neg + number_of_words))
    if prob_neg < prob_pos:
        if label == 1:
            true += 1
    if prob_neg > prob_pos:
        if label == 0:
            true += 1
    total += 1
print(true/total)

