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
  # arc_Match_Author = Arc(Match,Author)
  # arc_Match_Page = Arc(Match,Page)
  # arc_Match_Publisher = Arc(Match,Publisher)
  # arc_Match_Title = Arc(Match,Title)
  # arc_Match_Venue = Arc(Match,Venue)
  # arc_Match_Year = Arc(Match,Year)

  # Check table size
  # print (Address.getTableSize())

  # Conditional distribution for node 'Match'
  Match.setProbabilities([0.02,0.98])
  Address.setProbabilities([0.96, 0.04, 0, 0.01, 0.3, 0.69])
  Author.setProbabilities([0.96, 0.04, 0])
  Page.setProbabilities([0.96, 0.04, 0])
  Publisher.setProbabilities([0.96, 0.04, 0])
  Title.setProbabilities([0.96, 0.04, 0])
  Venue.setProbabilities([0.96, 0.04, 0])
  Year.setProbabilities([0.96, 0.04, 0])

  # net.addNodes([Address, Match])
  net.addNodes([Match, Address, Author, Page, Publisher, Title, Venue, Year])
  return net, Match, Address, Author, Page, Publisher, Title, Venue, Year



def calculate_match(net, Address, Author, Page, Publisher, Title, Venue, Year):
  # Set evidence
  net.setEvidence('Address', Address)
  # net.setEvidence('Author', Address)
  # net.setEvidence('Page', Page)
  # net.setEvidence('Publisher', Publisher)
  # net.setEvidence('Title', Title)
  # net.setEvidence('Venue', Venue)
  # net.setEvidence('Year', Year)

  net.computeBeliefs()


# Define a main() function.
def main():
  net, Match, Address, Author, Page, Publisher, Title, Venue, Year = setup_Network()
  calculate_match(net, Address = 1, Author = 1, Page = 1, Publisher = 1, Title = 1, Venue = 1, Year = 1)

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

