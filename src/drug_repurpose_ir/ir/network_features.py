import numpy as np
import pandas as pd
from sknetwork.ranking import PageRank, HITS
from sknetwork.data import from_edge_list
import gzip


class NetworkFeatures:
    """
    A class to help generate network features such as PageRank scores, HITS hub score and HITS authority scores.
    This class uses the scikit-network library https://scikit-network.readthedocs.io to calculate node ranking values.

    OPTIONAL reads
    1. PageRank: https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
    2. HITS: https://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture4/lecture4.html 
    """

    def load_network(self, network_filename: str, total_edges: int):
        """
        Loads the network from the specified file and returns the network. A network file 
        can be listed using a .csv or a .csv.gz file.

        Args:
            network_filename: The name of a .csv or .csv.gz file containing an edge list
            total_edges: The total number of edges in an edge list (please read the notes below)

        Returns:
            The loaded network from sknetwork
        """
        # TODO load the network edgelist dataset and return the scikit-network graph

        # NOTE: On a reference laptop, loading the network file's data took ~90 seconds
        # and constructing the network took ~75 seconds. We estimate that the entire
        # network construction memory requirement is under 5GB based on tests with
        # the reference implementation.

        # NOTE: Trying to create the network from a pandas dataframe will take lots of memory.
        # Although it works on Great Lakes, it is not memory efficient and may crash your system if you run it locally.
        # You'll need to read the documentation to figure out how to load in the network in the most memory-efficient way possible.

        # NOTE: total_edges is provided to help you with memory allocation. You may or may not need it depending on your implementation.
        # If you do use it, there are 92650947 edges in the big network we give you, pass this value in as the total_edges parameter for edgelist.csv.
        # If you do not use it, that's fine too. But please do not hardcode this value in your implementation nor delete total_edges from the function definition.

        # NOTE: your code should support reading both gzip and non-gzip formats.

        # NOTE: Please install scikit-network==0.31.0 to your virtual environment.
        
        # Instantiate an empty numpy array
        edge_list = np.empty((total_edges, 2), dtype=np.int64)

        open_func = gzip.open if network_filename.endswith(".gz") else open

        with open_func(network_filename, "rt") as f:

            # Skip header
            f.readline()

            for idx, line in enumerate(f):
                frm, to = line.split(",")
                edge_list[idx] = int(frm), int(to)
        
        graph = from_edge_list(
            edge_list,
            directed = True,
            weighted = False,
            reindex=True,
            matrix_only = False,
        )

        return graph


    def calculate_page_rank(self, graph, damping_factor=0.85, iterations=100) -> list[float]:
        """
        Calculates the PageRank scores for the provided network and
        returns the PageRank values for all nodes.

        Args:
            graph: A graph from sknetwork
            damping_factor: The complement of the teleport probability for the random walker
                For example, a damping factor of .8 has a .2 probability of jumping after each step.
            iterations: The maximum number of iterations to run when computing PageRank

        Returns:
            The PageRank scores for all nodes in the network (array-like)
        """
        # TODO: Use scikit-network to run Pagerank and return Pagerank scores
        
        model = PageRank(
            damping_factor=damping_factor, 
            n_iter=iterations,
        )

        scores = model.fit_predict(graph["adjacency"])
        return scores


    def calculate_hits(self, graph) -> tuple[list[float], list[float]]:
        """
        Calculates the hub scores and authority scores using the HITS algorithm
        for the provided network and returns the two lists of scores as a tuple.

        Args:
            graph: A graph from sknetwork

        Returns:
            The hub scores and authority scores (in that order) for all nodes in the network
        """
        # TODO: Use scikit-network to run HITS and return HITS hub scores and authority scores

        # NOTE: When returning the HITS scores, the returned tuple should have the hub scores in index 0 and
        #       authority score in index 1
        model = HITS()
        model = model.fit(graph["adjacency"])
        return model.scores_row_, model.scores_col_

    def get_all_network_statistics(self, graph) -> pd.DataFrame:
        """
        Calculates the PageRank and the hub scores and authority scores using the HITS algorithm
        for the provided network and returns a pandas DataFrame with columns: 
        'docid', 'pagerank', 'authority_score', and 'hub_score' containing the relevant values
        for all nodes in the network.

        Args:
            graph: A graph from sknetwork

        Returns:
            A pandas DataFrame with columns 'docid', 'pagerank', 'authority_score', and 'hub_score'
            containing the relevant values for all nodes in the network
        """

        # TODO: Calculate all the Pagerank and HITS scores for the network graph and store it in a dataframe

        # NOTE: We use a DataFrame here for efficient storage of the values on disk.
        # However, when you actually use these values, you'll convert this DataFrame
        # to another dictionary-based representation for faster lookup when making
        # the L2R features.

        # NOTE Return the dataframe
        pg = self.calculate_page_rank(graph)
        hub, authority = self.calculate_hits(graph)

        return pd.DataFrame({
            "docid": graph["names"],
            "pagerank": pg,
            "authority_score": authority,
            "hub_score": hub,
        })

# main function
# TODO: Save the dataframe obtained from NetworkFeatures.get_all_network_statistics() as a CSV or JSON
if __name__ == "__main__":
    
    features = NetworkFeatures()

    graph = features.load_network(
        network_filename="../edgelist.csv",
        total_edges=92650947,
    )

    df = features.get_all_network_statistics(graph)
    df.to_csv("../ranking_scores.csv", index=False)
