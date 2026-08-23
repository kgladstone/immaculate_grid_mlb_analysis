import pandas as pd

from app.services.baseball_network import BaseballNetwork


def _network():
    appearances = pd.DataFrame([
        {"playerID": "modern", "teamID": "A", "yearID": 2020},
        {"playerID": "bridge1", "teamID": "A", "yearID": 2020},
        {"playerID": "bridge1", "teamID": "B", "yearID": 2000},
        {"playerID": "bridge2", "teamID": "B", "yearID": 2000},
        {"playerID": "bridge2", "teamID": "C", "yearID": 1980},
        {"playerID": "old", "teamID": "C", "yearID": 1980},
        {"playerID": "old", "teamID": "D", "yearID": 1960},
        {"playerID": "ancient", "teamID": "D", "yearID": 1960},
        {"playerID": "isolated", "teamID": "Z", "yearID": 2020},
    ])
    teams = pd.DataFrame([
        {"teamID": "A", "yearID": 2020, "franchID": "FA", "name": "Alpha"},
        {"teamID": "B", "yearID": 2000, "franchID": "FB", "name": "Beta"},
        {"teamID": "C", "yearID": 1980, "franchID": "FC", "name": "Gamma"},
        {"teamID": "D", "yearID": 1960, "franchID": "FD", "name": "Delta"},
        {"teamID": "Z", "yearID": 2020, "franchID": "FZ", "name": "Zeta"},
    ])
    people = pd.DataFrame([
        {"playerID": player, "nameFirst": player.title(), "nameLast": "Player"}
        for player in ["modern", "bridge1", "bridge2", "old", "ancient", "isolated"]
    ])
    return BaseballNetwork(appearances, teams, people)


def test_teammates_preserve_team_year_evidence():
    network = _network()
    assert network.are_teammates("modern", "bridge1")
    assert network.shared_rosters("modern", "bridge1") == [("A", 2020)]


def test_shortest_path_across_generations_is_valid():
    network = _network()
    path = network.shortest_path("modern", "ancient")
    assert path is not None
    assert len(path) == 4
    for hop in path:
        assert (hop.team_id, hop.year) in network.shared_rosters(hop.from_player, hop.to_player)


def test_components_are_consistent():
    network = _network()
    _, sizes = network.connected_components()
    assert sorted(sizes.values()) == [1, 5]
    stats = network.statistics()
    assert stats["components"] == 2
    assert stats["largest_component"] == 5


def test_path_to_earliest_season():
    network = _network()
    path = network.path_to_earliest_baseball("modern")
    assert path is not None
    assert path[-1].to_player == "ancient"
    assert path[-1].year == 1960


def test_historical_relay_moves_to_earlier_career_starts():
    network = _network()
    path = network.historical_relay("modern")
    starts = [network.player_years["modern"][0]] + [network.player_years[hop.to_player][0] for hop in path]
    assert all(after < before for before, after in zip(starts, starts[1:]))


def test_oldest_player_walk_does_not_cycle():
    network = _network()
    oldest = pd.DataFrame([
        {"teamID": "A", "yearID": 2020, "playerID": "bridge1"},
        {"teamID": "B", "yearID": 2000, "playerID": "bridge2"},
        {"teamID": "C", "yearID": 1980, "playerID": "old"},
        {"teamID": "D", "yearID": 1960, "playerID": "ancient"},
    ])
    walk = network.oldest_player_walk(("A", 2020), oldest)
    rosters = [(row["teamID"], row["yearID"]) for row in walk]
    assert rosters == [("A", 2020), ("B", 2000), ("C", 1980), ("D", 1960)]
    assert len(rosters) == len(set(rosters))
