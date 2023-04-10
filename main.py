from pybn import *
import pandas as pd
import json

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
  # Conditional distribution for node 'Match'
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

  print(count)

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
  net = Network('CoraProblem')

  # Setup node
  Match = Node('Match')
  Match.addOutcomes(['match','notmatch'])

  Address = Node('Address')
  Address.addOutcomes(['match','unknown','notmatch'])

  Author = Node('Author')
  Author.addOutcomes(['match','unknown','notmatch'])

  Page = Node('Page')
  Page.addOutcomes(['match','unknown','notmatch'])

  Publisher = Node('Publisher')
  Publisher.addOutcomes(['match','unknown','notmatch'])

  Title = Node('Title')
  Title.addOutcomes(['match','unknown','notmatch'])

  Venue = Node('Venue')
  Venue.addOutcomes(['match','unknown','notmatch'])

  Year = Node('Year')
  Year.addOutcomes(['match','unknown','notmatch'])

  # Set up the graph
  arc_Match_Address = Arc(Match,Address)
  arc_Match_Author = Arc(Match,Author)
  arc_Match_Page = Arc(Match,Page)
  arc_Match_Publisher = Arc(Match,Publisher)
  arc_Match_Title = Arc(Match,Title)
  arc_Match_Venue = Arc(Match,Venue)
  arc_Match_Year = Arc(Match,Year)

  # Check table size
  # print (Address.getTableSize())

  # Conditional distribution for node 'Match'
  setup_Probability_Distribution(Match, Address, Author, Page, Publisher, Title, Venue, Year)

  # net.addNodes([Address, Match])
  net.addNodes([Match, Address, Author, Page, Publisher, Title, Venue, Year])
  return net, Match, Address, Author, Page, Publisher, Title, Venue, Year


def calculate_match(net, Address, Author, Page, Publisher, Title, Venue, Year):
  net.setEvidence('Address', Address)
  net.setEvidence('Author', Author)
  net.setEvidence('Page', Page)
  net.setEvidence('Publisher', Publisher)
  net.setEvidence('Title', Title)
  net.setEvidence('Venue', Venue)
  net.setEvidence('Year', Year)

  net.computeBeliefs()

# helper function display result
def display(Match, Address, Author, Page, Publisher, Title, Venue, Year):
  # Print the results for each node
  print("RESULT:")
  print('\tMatch:\t\t', Match.getBeliefs())
  print('\tAddress:\t', Address.getBeliefs())
  print('\tAuthor:\t\t', Author.getBeliefs())
  print('\tPage:\t\t', Page.getBeliefs())
  print('\tPublisher:\t', Publisher.getBeliefs())
  print('\tTitle:\t\t', Title.getBeliefs())
  print('\tVenue:\t\t', Venue.getBeliefs())
  print('\tYear:\t\t', Year.getBeliefs())

def main():
  # TODO: Train the network
  net, Match, Address, Author, Page, Publisher, Title, Venue, Year = setup_Network()
  display(Match, Address, Author, Page, Publisher, Title, Venue, Year)


def is_match(entityA, entityB):
  net.reset()
  # Apply the network
  # Setup Evident: 1 - match, 2 - unknown, 3 - notmatch
  
        
  AddressEvident = feature_address(df.loc[i]['address'], df.loc[j]['address'])
  AuthorEvident = feature_author(df.loc[i]['author'], df.loc[j]['author'])
  PageEvident = feature_page(df.loc[i]['pages'], df.loc[j]['pages']),
  PublisherEvident = feature_publisher(df.loc[i]['publisher'], df.loc[j]['publisher']),
  TitleEvident = feature_title(df.loc[i]['title'], df.loc[j]['title']),
        feature_venue(df.loc[i]['venue'], df.loc[j]['venue']),
        feature_year(df.loc[i]['year'], df.loc[j]['year']),
  TitleEvident = 1
  VenueEvident = 2
  YearEvident = 3

  calculate_match(net,
  Address = AddressEvident,
  Author = AuthorEvident,
  Page = PageEvident,
  Publisher = PublisherEvident,
  Title = TitleEvident,
  Venue =   VenueEvident,
  Year = YearEvident)

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()

