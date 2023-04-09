import Levenshtein
import spacy
from fuzzywuzzy import fuzz
import pandas as pd

def isnull(str):
    return pd.isnull(str) or str == "" 


def compare_strings_Levenshtein(string1, string2):
    distance = Levenshtein.distance(string1, string2)
    similarity = 1 - (distance / len(string1))
    return similarity

def compare_strings_spacy(string1, string2):
    nlp = spacy.load('en_core_web_md')
    doc1 = nlp(string1)
    doc2 = nlp(string2)

    similarity_score = doc1.similarity(doc2)
    return similarity_score

def compare_strings_fuzz(string1, string2):
    similarity_score = fuzz.token_sort_ratio(string1, string2)
    return similarity_score / 100


def feature_address(string1, string2):
    if isnull(string1) or isnull(string2):
        return False
    return compare_strings_spacy(string1, string2) >= 0.7

def feature_author(string1, string2):
    if isnull(string1) or isnull(string2):
        return False
    return compare_strings_Levenshtein(string1, string2) >= 0.7

def feature_page(string1, string2):
    if isnull(string1) or isnull(string2):
        return False
    return compare_strings_spacy(string1, string2) >= 0.7

def feature_publisher(string1, string2):
    if isnull(string1) or isnull(string2):
        return False
    return compare_strings_spacy(string1, string2) >= 0.7

def feature_title(string1, string2):
    if isnull(string1) or isnull(string2):
        return False
    return compare_strings_spacy(string1, string2) >= 0.7

def feature_venue(string1, string2):
    if isnull(string1) or isnull(string2):
        return False
    return compare_strings_Levenshtein(string1, string2) >= 0.7

def feature_year(string1, string2):
    if isnull(string1) or isnull(string2):
        return False
    return compare_strings_fuzz(string1, string2) >= 0.8