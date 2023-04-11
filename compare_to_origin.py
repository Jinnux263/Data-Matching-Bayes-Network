import json
import pandas as pd
from pybn import feature_address, feature_author, feature_page, feature_publisher, feature_title, feature_venue, feature_year

############################# Using the function to check entity matching
with open('model.json', 'r') as file:
  model = json.load(file)

threshold = 0.5

def is_match(entityA, entityB):
  # In the train data: 0 - match, 1 - unknown, 2 - notmatch
  # Setup Evident: 1 - match, 2 - unknown, 3 - notmatch
  AddressEvident = feature_address(entityA['address'], entityB['address']) + 1
  AuthorEvident = feature_author(entityA['author'], entityB['author']) + 1
  PageEvident = feature_page(entityA['pages'], entityB['pages']) + 1
  PublisherEvident = feature_publisher(entityA['publisher'], entityB['publisher']) + 1
  TitleEvident = feature_title(entityA['title'], entityB['title']) + 1
  VenueEvident = feature_venue(entityA['venue'], entityB['venue']) + 1
  YearEvident = feature_year(entityA['year'], entityB['year']) + 1

  a = (model['Address'][AddressEvident - 1] *  model['Author'][AuthorEvident - 1] * model['Page'][PageEvident - 1] * model['Publisher'][PublisherEvident - 1] * model['Title'][TitleEvident - 1] * model['Venue'][VenueEvident - 1] * model['Year'][YearEvident - 1]) * model['Match'][0]

  b = a + (model['Address'][AddressEvident + 2] * model['Author'][AuthorEvident + 2] * model['Page'][PageEvident + 2] * model['Publisher'][PublisherEvident + 2] * model['Title'][TitleEvident + 2] * model['Venue'][VenueEvident + 2] * model['Year'][YearEvident + 2]) * model['Match'][1]

  return (a / b) > threshold

def test_model():
  data = pd.read_csv("./data/train_data.csv", sep='|', engine='python', na_filter=False).astype(str)
  df = pd.read_csv("./data/cora.csv", sep='|', engine='python', na_filter=False).astype(str)
  del df['editor']
  del df['institution']
  del df['month']
  del df['note']
  del df['volume']
  del df['Unnamed: 13']

  check = []
  for i in range(0, len(data.index)):
  # for i in range(0, 10):
    if i % 1000 == 0:
      print(i, " rows done")

    indexA = int(data.loc[i]['EntityA'])
    indexB = int(data.loc[i]['EntityB'])

    entityA = {
      'address': df.iloc[indexA]['address'],
      'author': df.iloc[indexA]['author'],
      'pages': df.iloc[indexA]['pages'],
      'publisher': df.iloc[indexA]['publisher'],
      'title': df.iloc[indexA]['title'],
      'venue': df.iloc[indexA]['venue'],
      'year': df.iloc[indexA]['year']
    }
    entityB = {
      'address': df.iloc[indexB]['address'],
      'author': df.iloc[indexB]['author'],
      'pages': df.iloc[indexB]['pages'],
      'publisher': df.iloc[indexB]['publisher'],
      'title': df.iloc[indexB]['title'],
      'venue': df.iloc[indexB]['venue'],
      'year': df.iloc[indexB]['year']
    }
    check.append(1 if is_match(entityA, entityB) else 0)

  check_df = pd.DataFrame()
  check_df['origin'] = data.loc[:, 'Match']
  # check_df['origin'] = data.loc[0:9, 'Match']
  check_df['Predict'] = check

  comp_res = []
  # for i in range(0, len(check_df.index)):
  for i in range(0, len(check_df.index)):
    comp_res.append(int(check_df.iloc[i, 0]) == int(check_df.iloc[i, 1]))

  check_df['Check_Result'] = comp_res
  check_df.to_csv("data/check_train_data.csv", sep="|")

def main():
  entityA = {
    'address': '',
    'author': 'avrim blum, merrick furst, jeffrey jackson, michael kearns, yishay mansour, and steven rudich.',
    'pages': 'pages 253 - 262,',
    'publisher': '',
    'title': 'weakly learning dnf and characterizing statistical query learning using fourier analysis.',
    'venue': 'in t he 26 th annual acm symposium on t heory of computing,',
    'year': '1994'
  }
  entityB = {
    'address': '',
    'author': 'avrim blum, merrick furst, jeffery jackson, michael kearns, yishay mansour, and steven rudich.',
    'pages': '',
    'publisher': '',
    'title': 'weakly learning dnf and characterizing statistical query learning using fourier analysis.',
    'venue': 'in proceedings of twenty-sixth acm symposium on theory of computing,',
    'year': '(1994)'
  }

  print(is_match(entityA, entityB))

# This is the standard boilerplate that calls the main() function.
# if __name__ == '__main__':
#   main()
test_model()

