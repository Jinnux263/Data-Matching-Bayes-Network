from pybn import *
import pandas as pd
import json


threshold = 0.5

####################### This code is used for generate the train_data.csv file
# Check if two entity match or not in reality
def check_match_gt(gt, i, j):
  # Check by binary search i and sequence search j
  l = 0
  r = len(gt.index)

  while l < r:
    m = (l + r) // 2
    if gt.loc[m].iloc[0] < i:
      l = m + 1
    else:
      r = m

  if gt.loc[l].iloc[0] != i:
    return False
  while l < len(gt.index) and gt.loc[l].iloc[0] == i:
    if gt.loc[l].iloc[1] == j:
      return True
    l += 1
  return False;


def generate(df, gt, start, end):
  new_df = pd.DataFrame()
  new_df['EntityA'] = None
  new_df['EntityB'] = None
  data_top = df.head() 
  for header in data_top:
    new_df[header] = None
  del new_df['Entity Id']
  # Add Match column
  new_df['Match'] = None

  # Generate the data
  prev = 0
  for i in range(start, end):
    print("State ", i, " done.")
    for j in range(i + 1, end):
      # print("\tStep ", j, " done.")
      isMatch = check_match_gt(gt, i, j)
      new_df.loc[len(new_df.index)] =  [
        i,
        j,
        feature_address(df.loc[i]['address'], df.loc[j]['address']),
        feature_author(df.loc[i]['author'], df.loc[j]['author']),
        feature_page(df.loc[i]['pages'], df.loc[j]['pages']),
        feature_publisher(df.loc[i]['publisher'], df.loc[j]['publisher']),
        feature_title(df.loc[i]['title'], df.loc[j]['title']),
        feature_venue(df.loc[i]['venue'], df.loc[j]['venue']),
        feature_year(df.loc[i]['year'], df.loc[j]['year']),
        isMatch
      ]
    if i > 0 and i % 100 == 0:
      new_df.to_csv("data/train_data_" + str(prev) + "_" + str(i) +".csv", sep="|")
      prev = i
    elif i == end - 1:
      new_df.to_csv("data/train_data_" + str(prev) + "_" + str(i) +".csv", sep="|")
      prev = i
  
  new_df.to_csv("data/train_data.csv", sep="|")


def generate_data():
  df = pd.read_csv("./data/cora.csv", sep='|', engine='python', na_filter=False).astype(str)
  gt = pd.read_csv("./data/cora_gt.csv", sep=',', header=None, engine='python')
  del df['editor']
  del df['institution']
  del df['month']
  del df['note']
  del df['volume']
  del df['Unnamed: 13']

  # # Generate the data
  generate(df, gt, 0, len(df.index))

#generate the data
# generate_data()

####################### From here we have the training data
def validate_count(data_top, count):
  for header in data_top:
    maxVal = 3
    if header == "Match":
      maxVal = 2
    for val in range(0, maxVal):
      if val not in count['Match'][header]:
        count['Match'][header][val] = 0
      if val not in count['NotMatch'][header]:
        count['NotMatch'][header][val] = 0

def calculate_the_feature():
  count = {}
  count['Match'] = {}
  count['NotMatch'] = {}
  
  data = pd.read_csv("./data/train_data.csv", sep='|', engine='python', na_filter=False).astype(str)
  del data['EntityA'] 
  del data['EntityB'] 

  data_top = data.head() 

  for header in data_top:
    count['Match'][header] = {}
    count['NotMatch'][header] = {}

  validate_count(data_top, count)
  
  # for i in range(0, 10):
  for i in range(0, len(data.index)):
    if i % 1000 == 0:
      print(i, " rows done...")
    match_header = 'Match' if int(data.loc[i].loc['Match']) == 1 else 'NotMatch'
    for header in data_top:
      count[match_header][header][int(data.loc[i].loc[header])] += 1
      # if data.loc[i].loc[header] in count[match_header][header]:
      #   count[match_header][header][data.loc[i].loc[header]] += 1
      # else:
      #   count[match_header][header][data.loc[i].loc[header]] = 1
  if count['Match']['Match'][1] == 0:
    count['Match']['Match'][1] = 1
  if count['NotMatch']['Match'][0] == 0:
    count['NotMatch']['Match'][0] = 1

  count['total'] = len(data.index)

  return count


####################### Setup and used the Bayes Network
def setup_Probability_Distribution(Match, Address, Author, Page, Publisher, Title, Venue, Year):
  # count = calculate_the_feature()
  ## This code will generate orther way if run again cause I have not change code
  # total = count['total']

  # with open('count.json', 'w') as file:
  #   json.dump(count, file)

  with open('count.json', 'r') as file:
    my_dict = json.load(file)

  count = {}
  count['Match'] = {}
  count['NotMatch'] = {}
  
  data = pd.read_csv("./data/train_data.csv", sep='|', engine='python', na_filter=False).astype(str)
  del data['EntityA'] 
  del data['EntityB'] 

  data_top = data.head() 

  for header in data_top:
    count['Match'][header] = {}
    count['NotMatch'][header] = {}

  validate_count(data_top, count)

  for match_header in count:
    for header in count[match_header]:
      for val in count[match_header][header]:
        count[match_header][header][val] = my_dict[match_header][header][str(val)]

  total = count['Match']['Match'][1] + count['NotMatch']['Match'][0]

  # print("COUNT: ",count)
  # print("TOTAL: ", total)

  Match.setProbabilities([count['Match']['Match'][1] / total,
   1 - count['Match']['Match'][1] / total])
  
  Address.setProbabilities([
    count['Match']['address'][0] / count['Match']['Match'][1],
    count['Match']['address'][1] / count['Match']['Match'][1],
    count['Match']['address'][2] / count['Match']['Match'][1],
    count['NotMatch']['address'][0] / count['NotMatch']['Match'][0],
    count['NotMatch']['address'][1] / count['NotMatch']['Match'][0],
    count['NotMatch']['address'][2] / count['NotMatch']['Match'][0]
    ])
  
  Author.setProbabilities([
    count['Match']['author'][0] / count['Match']['Match'][1],
    count['Match']['author'][1] / count['Match']['Match'][1],
    count['Match']['author'][2] / count['Match']['Match'][1],
    count['NotMatch']['author'][0] / count['NotMatch']['Match'][0],
    count['NotMatch']['author'][1] / count['NotMatch']['Match'][0],
    count['NotMatch']['author'][2] / count['NotMatch']['Match'][0]
  ])
  
  Page.setProbabilities([
    count['Match']['pages'][0] / count['Match']['Match'][1],
    count['Match']['pages'][1] / count['Match']['Match'][1],
    count['Match']['pages'][2] / count['Match']['Match'][1],
    count['NotMatch']['pages'][0] / count['NotMatch']['Match'][0],
    count['NotMatch']['pages'][1] / count['NotMatch']['Match'][0],
    count['NotMatch']['pages'][2] / count['NotMatch']['Match'][0]
  ])
  
  Publisher.setProbabilities([
    count['Match']['publisher'][0] / count['Match']['Match'][1],
    count['Match']['publisher'][1] / count['Match']['Match'][1],
    count['Match']['publisher'][2] / count['Match']['Match'][1],
    count['NotMatch']['publisher'][0] / count['NotMatch']['Match'][0],
    count['NotMatch']['publisher'][1] / count['NotMatch']['Match'][0],
    count['NotMatch']['publisher'][2] / count['NotMatch']['Match'][0]
  ])
  
  Title.setProbabilities([
    count['Match']['title'][0] / count['Match']['Match'][1],
    count['Match']['title'][1] / count['Match']['Match'][1],
    count['Match']['title'][2] / count['Match']['Match'][1],
    count['NotMatch']['title'][0] / count['NotMatch']['Match'][0],
    count['NotMatch']['title'][1] / count['NotMatch']['Match'][0],
    count['NotMatch']['title'][2] / count['NotMatch']['Match'][0]
  ])
  
  Venue.setProbabilities([
    count['Match']['venue'][0] / count['Match']['Match'][1],
    count['Match']['venue'][1] / count['Match']['Match'][1],
    count['Match']['venue'][2] / count['Match']['Match'][1],
    count['NotMatch']['venue'][0] / count['NotMatch']['Match'][0],
    count['NotMatch']['venue'][1] / count['NotMatch']['Match'][0],
    count['NotMatch']['venue'][2] / count['NotMatch']['Match'][0]
  ])
  
  Year.setProbabilities([
    count['Match']['year'][0] / count['Match']['Match'][1],
    count['Match']['year'][1] / count['Match']['Match'][1],
    count['Match']['year'][2] / count['Match']['Match'][1],
    count['NotMatch']['year'][0] / count['NotMatch']['Match'][0],
    count['NotMatch']['year'][1] / count['NotMatch']['Match'][0],
    count['NotMatch']['year'][2] / count['NotMatch']['Match'][0]
  ])


def setup_Network():
  # Create a Network
  Fnet = Network('CoraProblem')

  # Setup node
  FMatch = Node('Match')
  FMatch.addOutcomes(['match','notmatch'])

  FAddress = Node('Address')
  FAddress.addOutcomes(['match','unknown','notmatch'])

  FAuthor = Node('Author')
  FAuthor.addOutcomes(['match','unknown','notmatch'])

  FPage = Node('Page')
  FPage.addOutcomes(['match','unknown','notmatch'])

  FPublisher = Node('Publisher')
  FPublisher.addOutcomes(['match','unknown','notmatch'])

  FTitle = Node('Title')
  FTitle.addOutcomes(['match','unknown','notmatch'])

  FVenue = Node('Venue')
  FVenue.addOutcomes(['match','unknown','notmatch'])

  FYear = Node('Year')
  FYear.addOutcomes(['match','unknown','notmatch'])

  # Set up the graph
  arc_Match_Address = Arc(FMatch,FAddress)
  arc_Match_Author = Arc(FMatch,FAuthor)
  arc_Match_Page = Arc(FMatch,FPage)
  arc_Match_Publisher = Arc(FMatch,FPublisher)
  arc_Match_Title = Arc(FMatch,FTitle)
  arc_Match_Venue = Arc(FMatch,FVenue)
  arc_Match_Year = Arc(FMatch,FYear)

  # Check table size
  # print (FAddress.getTableSize())

  # Conditional distribution for node 'Match'
  setup_Probability_Distribution(FMatch, FAddress, FAuthor, FPage, FPublisher, FTitle, FVenue, FYear)

  # net.addNodes([Address, Match])
  Fnet.addNodes([FMatch, FAddress, FAuthor, FPage, FPublisher, FTitle, FVenue, FYear])
  return Fnet, FMatch, FAddress, FAuthor, FPage, FPublisher, FTitle, FVenue, FYear


def calculate_match(Inet, IAddress, IAuthor, IPage, IPublisher, ITitle, IVenue, IYear):
  Inet.setEvidence('Address', IAddress)
  Inet.setEvidence('Author', IAuthor)
  Inet.setEvidence('Page', IPage)
  Inet.setEvidence('Publisher', IPublisher)
  Inet.setEvidence('Title', ITitle)
  Inet.setEvidence('Venue', IVenue)
  Inet.setEvidence('Year', IYear)

  Inet.computeBeliefs()

# helper function display result
def display(IMatch, IAddress, IAuthor, IPage, IPublisher, ITitle, IVenue, IYear):
  # Print the results for each node
  print("RESULT:")
  print('\tMatch:\t\t', IMatch.getBeliefs())
  print('\tAddress:\t', IAddress.getBeliefs())
  print('\tAuthor:\t\t', IAuthor.getBeliefs())
  print('\tPage:\t\t', IPage.getBeliefs())
  print('\tPublisher:\t', IPublisher.getBeliefs())
  print('\tTitle:\t\t', ITitle.getBeliefs())
  print('\tVenue:\t\t', IVenue.getBeliefs())
  print('\tYear:\t\t', IYear.getBeliefs())



############################# Using the function to check entity matching
net, Match, Address, Author, Page, Publisher, Title, Venue, Year = setup_Network()
# Reset making bugs
# net.reset()
# # display(Match, Address, Author, Page, Publisher, Title, Venue, Year)

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

  # print('Match: ', AddressEvident == 1)
  # print('Match: ', AuthorEvident == 1)
  # print('Match: ', PageEvident == 1)
  # print('Match: ', PublisherEvident == 1)
  # print('Match: ', TitleEvident == 1)
  # print('Match: ', VenueEvident == 1)
  # print('Match: ', YearEvident == 1)

  calculate_match(Inet = net,
  IAddress = AddressEvident,
  IAuthor = AuthorEvident,
  IPage = PageEvident,
  IPublisher = PublisherEvident,
  ITitle = TitleEvident,
  IVenue =   VenueEvident,
  IYear = YearEvident)

  display(Match, Address, Author, Page, Publisher, Title, Venue, Year)

  return Match.getBeliefs()[0] > threshold


def main():
  entityA = {
    'address': '',
    'author': 'John.D',
    'pages': '100',
    'publisher': '',
    'title': 'A paper 2',
    'venue': '',
    'year': '2019'
  }
  entityB = {
    'address': '123 Main St',
    'author': 'John Doe',
    'pages': '1-10',
    'publisher': 'IEEE',
    'title': 'A paper',
    'venue': 'ICML',
    'year': '2018'
  }

  print(is_match(entityA, entityB))

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()

