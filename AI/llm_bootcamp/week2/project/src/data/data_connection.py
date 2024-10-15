from datasets import load_dataset


class DataConnector:
 
    @staticmethod
    def get_data(path):
        # TODO: Implement the DataConnector.get_data method 
        # of the data_connection.py file. Use the load_dataset 
        # function from the datasets package and select split = "train" 
        # to make sure we only train on the training data.
        return load_dataset(
            path=path,
            split="train",
            trust_remote_code=True
        )
