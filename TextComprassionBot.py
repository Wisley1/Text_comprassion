import telebot
from telebot import types


import textwrap
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import nltk
nltk.download('stopwords')

token = '6190362939:AAH3l6OO1JrmqY4W2MYWeax8TKy020OqFQg'
bot = telebot.TeleBot(token)

@bot.message_handler(commands = ['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет! Я могу помочь тебе сделать краткое содержание твоего текста. Для этого просто отправь мне его!\n Я работаю только с большим количеством предложений. Убедись, что в твоем тексте их не меньше 10.')


# Смысловая часть кода. Начало.
def preprocess_text(text):
    sentences = text.split(". ")
    processed_sentences = []

    for sentence in sentences:
        processed_sentence = sentence.replace("[^a-zA-Z]", " ").split(" ")
        processed_sentences.append(processed_sentence)

    return processed_sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)



def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

@bot.message_handler(func=lambda message: True)
def generate_summary(message):

    stop_words = stopwords.words('russian')
    summarize_text = []

    # Step 1 - Read text anc split it
    text = message.text
    sentences = preprocess_text(text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    dot_count = 0
    for char in text:
        if char == '.':
            dot_count += 1

    top_n = int(dot_count)//5
    top_n = min(top_n, len(ranked_sentence))

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    summary = "\n".join(summarize_text)
    bot.reply_to(message, f"Ваш текст:\n\n{summary}\n\nМожешь отправить мне еще один текст!")


#Смысловая часть кода. Конец.



bot.polling()