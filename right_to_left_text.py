import matplotlib.pyplot as plt
import re 

text = "Python اور matplotlib میں اردو؟"  # Urdu sentence that starts with an English word

#checks if the word contains any arabic / urdu characters
def contains_arabic(word):
    return bool(re.search(r'[\u0600-\u06FF]', word))

def reverse_mixed_text(text):
    words = text.split()
    groups = []
    arabic_phrase = []

    for word in words:
        if contains_arabic(word):
            #keeps consecutive Arabic/Urdu words together 
            arabic_phrase.append(word)
        else:
            #finish the arabic/urdu phrase (if there is one)
            if arabic_phrase:
                groups.append(" ".join(arabic_phrase))
                arabic_phrase = []

            groups.append(word)

    # adds arabic/urdu phrase if the string ended with one
    if arabic_phrase:
        groups.append(" ".join(arabic_phrase))

    #reverse the groups
    groups.reverse()

    return " ".join(groups)

#creating a function that user can call to explicitly decide if 
#text should be read left to right (ltr) or right to left (rtl)
def text_direction(direction, text):
    if direction == 'ltr':
        return text
    if direction == 'rtl':
        return reverse_mixed_text(text)


fig = plt.figure(figsize=(6, 1.6))
fig.text(0.95, 0.70, text, ha="right", fontsize=18, language="ur")
fig.text(0.95, 0.25, "‏" + text, ha="right", fontsize=18, language="ur")  # RLM prefix
fig.savefig("repro_direction.png", dpi=100)
