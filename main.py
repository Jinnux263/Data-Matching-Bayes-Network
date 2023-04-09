from pybn import *
import pandas as pd

## Test the feature functions work properly
# df = pd.read_csv("./data/cora.csv", sep='|', engine='python', na_filter=False).astype(str)
# gt = pd.read_csv("./data/cora_gt.csv", sep='|', engine='python')
# del df['editor']
# del df['institution']
# del df['month']
# del df['note']
# del df['volume']

# r = 64
# com_r = 65
# rowA = df.loc[r]
# rowB = df.loc[com_r]

# print("TEST: ")
# print("\tAddress:\t", feature_address(rowA['address'], rowB['address']))
# print("\tAuthor:\t\t", feature_author(rowA['author'], rowB['author']))
# print("\tPage:\t\t", feature_page(rowA['pages'], rowB['pages']))
# print("\tPublisher:\t", feature_publisher(rowA['publisher'], rowB['publisher']))
# print("\tTitle:\t\t", feature_title(rowA['title'], rowB['title']))
# print("\tVenue:\t\t", feature_venue(rowA['venue'], rowB['venue']))
# print("\tYear:\t\t", feature_year(rowA['year'], rowB['year']))
# print("")

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


def generate_data():
  df = pd.read_csv("./data/cora.csv", sep='|', engine='python', na_filter=False).astype(str)
  gt = pd.read_csv("./data/cora_gt.csv", sep=',', engine='python')
  del df['editor']
  del df['institution']
  del df['month']
  del df['note']
  del df['volume']
  del df['Unnamed: 13']


  new_df = pd.DataFrame()
  
  new_df['EntityA'] = None
  new_df['EntityB'] = None
  data_top = df.head() 
  for header in data_top:
    new_df[header] = None
  del new_df['Entity Id']
  # Add Match column
  new_df['Match'] = None
  # print(new_df)

  # print(check_match_gt(gt, 64, 66))

  # # Generate the data
  for i in range(0, 10):
    for j in range(i + 1, 10):
  # for i in range(0, len(df.index) - 1):
  #   for i in range(i + 1, len(df.index) - 1):
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

  new_df.to_csv("data/train_data.csv", sep="|")



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
  Match.setProbabilities([0.0101526723,1 - 0.0101526723])
  Address.setProbabilities([0.5, 0.3, 0.2, 0.05, 0.3, 0.65])
  Author.setProbabilities([0.5, 0.3, 0.2, 0.05, 0.3, 0.65])
  Page.setProbabilities([0.5, 0.3, 0.2, 0.05, 0.3, 0.65])
  Publisher.setProbabilities([0.5, 0.3, 0.2, 0.05, 0.3, 0.65])
  Title.setProbabilities([0.5, 0.3, 0.2, 0.05, 0.3, 0.65])
  Venue.setProbabilities([0.5, 0.3, 0.2, 0.05, 0.3, 0.65])
  Year.setProbabilities([0.1, 0.4, 0.5, 0.03, 0.42, 0.55])

  # net.addNodes([Address, Match])
  net.addNodes([Match, Address, Author, Page, Publisher, Title, Venue, Year])
  return net, Match, Address, Author, Page, Publisher, Title, Venue, Year



def calculate_match(net, Address, Author, Page, Publisher, Title, Venue, Year):
  
  # For testing
  # net.setEvidence('Title', 1)
  # Set evidence
  net.setEvidence('Address', Address)
  net.setEvidence('Author', Author)
  net.setEvidence('Page', Page)
  net.setEvidence('Publisher', Publisher)
  net.setEvidence('Title', Title)
  net.setEvidence('Venue', Venue)
  net.setEvidence('Year', Year)

  net.computeBeliefs()


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


# Define a main() function.
def main():
  net, Match, Address, Author, Page, Publisher, Title, Venue, Year = setup_Network()


  # Setup Evident: 0 - match, 1 - unknown, 2 - notmatch
  # AddressEvident = 2
  # AuthorEvident = 1
  # PageEvident = 2
  # PublisherEvident = 2
  # TitleEvident = 1
  # VenueEvident = 3
  # YearEvident = 1


  # calculate_match(net,
  # Address = AddressEvident,
  # Author = AuthorEvident,
  # Page = PageEvident,
  # Publisher = PublisherEvident,
  # Title = TitleEvident,
  # Venue =   VenueEvident,
  # Year = YearEvident)

  # display(Match, Address, Author, Page, Publisher, Title, Venue, Year)

  generate_data()


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()

