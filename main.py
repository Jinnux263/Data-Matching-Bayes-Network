from pybn import *
import pandas as pd

df = pd.read_csv("./data/cora.csv", sep='|', engine='python', na_filter=False).astype(str)
gt = pd.read_csv("./data/cora_gt.csv", sep='|', engine='python')

del df['editor']
del df['institution']
del df['month']
del df['note']
del df['volume']


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


# Define a main() function.
def main():
  net, Match, Address, Author, Page, Publisher, Title, Venue, Year = setup_Network()


  # Setup Evident: 0 - match, 1 - unknown, 2 - notmatch
  AddressEvident = 2
  AuthorEvident = 1
  PageEvident = 2
  PublisherEvident = 2
  TitleEvident = 1
  VenueEvident = 3
  YearEvident = 1


  calculate_match(net,
  Address = AddressEvident,
  Author = AuthorEvident,
  Page = PageEvident,
  Publisher = PublisherEvident,
  Title = TitleEvident,
  Venue =   VenueEvident,
  Year = YearEvident)

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

  # This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()

