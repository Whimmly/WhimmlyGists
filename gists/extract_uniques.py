from fuzzywuzzy import fuzz

def extract_uniques(tokens):
  """
  Extract the unique tokens out of a list of tokens.
  Examples of similar / duplicate tokens are:
  - guiness, guinnesses
  - hoppy, hops
  - citrusy, citrus
  Words that end with y are preferred (i.e. fruit vs. fruity)
  """
  uniques = set()
  tokens = set(tokens)
  token_graph = {token: {'neighbors': set(), 'visited': False}
                 for token in tokens}

  # Create a graph where similar words are connected
  for token in tokens:
    for other_token in tokens:
      if token != other_token and fuzz.ratio(token, other_token) >= 60:
        token_graph[token]['neighbors'].add(other_token)
        token_graph[other_token]['neighbors'].add(token)

  # DFS to extract the most connected word in each group of connected words
  for token in tokens:
    if token_graph[token]['visited'] is True:
      continue
    stack = [token]
    max_edges = 0
    max_node = token

    while len(stack) > 0:
      node = stack.pop()
      token_graph[node]['visited'] = True

      if len(token_graph[node]['neighbors']) > max_edges or \
         (len(token_graph[node]['neighbors']) == max_edges and
              node[-1] == 'y' and max_node[-1] != 'y'):
        max_edges = len(token_graph[node]['neighbors'])
        max_node = node

      for neighbor in token_graph[node]['neighbors']:
        if token_graph[neighbor]['visited'] is False:
          stack.append(neighbor)
    uniques.add(max_node)
  return list(uniques)