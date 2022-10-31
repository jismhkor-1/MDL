import random


templates = ["the {category} is {sentiment}",
             "the opinion is {sentiment} for {category}",
             "people feel {sentiment} to {category}",
             "the sentiment polarity of {category} is {sentiment}",
             "the {category} category is rated as {sentiment}",
             "it is {sentiment} for {category}",
             "in terms of {category}, people feel {sentiment}",
             "the {category} category comes with a {sentiment} comment",
             "the opinion for {category} is {sentiment}",
             "it is {sentiment} when talking about the {category} category"]

sentiment_words = {"positive": ["good", "nice", "well", "excellent", "great", "the best"],
                   "negative": ["bad", "terrible", "awful", "dreadful", "horrid"],
                   "neutral": ["not bad", "just so", "okay", "so-so"]}

end_case = '.'
conjunction_case = {"non-flip": ["and", ","], "flip": ["but", "while"]}


def surrogate_for_one_pair(category_sentiment):
    template = random.choice(templates)
    c = category_sentiment['category']
    s = category_sentiment['polarity']
    sentiment_word = random.choice(sentiment_words[s])
    surrogate = template.format(category=c, sentiment=sentiment_word)
    return surrogate


def generate_combine_template(category_sentiments):
    t = []
    sentiment_list = [item['polarity'] for item in category_sentiments]
    for i in range(len(sentiment_list)-1):
        if sentiment_list[i+1] == sentiment_list[i]:
            t.append(random.choice(conjunction_case["non-flip"]))
        else:
            t.append(random.choice(conjunction_case["flip"]))
    combine_template = "{} " + " {} ".join(t) + " {}" + end_case
    return combine_template


def surrogate_generation(category_sentiments):
    surrogates = map(surrogate_for_one_pair, category_sentiments)
    combine_template = generate_combine_template(category_sentiments)
    combined_surrogates = combine_template.format(*surrogates)
    return combined_surrogates
