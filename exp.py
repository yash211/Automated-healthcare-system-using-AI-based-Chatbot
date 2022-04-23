import nltk
s = input("Enter sentence: ")

words_in_sent=nltk.word_tokenize(s) 
print(words_in_sent) 
print(nltk.pos_tag(words_in_sent))
